#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
STRESS TEST: LLaMA-7B with LoRA, Tensor Parallelism, and Differential Privacy.

This stress test demonstrates:
- Large model: LLaMA-7B scale (4096d, 32 layers, ~7B parameters)
- LoRA adapters for memory-efficient training
- Large batches and long sequences
- Tensor parallelism across multiple GPUs
- Differential privacy with per-sample gradients
"""

import logging
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import LlamaConfig, LlamaForCausalLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# STRESS TEST CONFIGURATION
EMBEDDING_LENGTH = 32000
SEQ_LENGTH = 512  # Long sequences
BATCH_SIZE = 8  # Moderate batch size
LORA_RANK = 64  # LoRA rank for good capacity

# Model size: LLaMA-7B
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008
NUM_LAYERS = 32
NUM_HEADS = 32


def get_input_and_output(device):
    """Generate random input and labels."""
    input_ids = torch.randint(EMBEDDING_LENGTH, size=(BATCH_SIZE, SEQ_LENGTH)).to(
        device
    )
    labels = torch.randint(EMBEDDING_LENGTH, size=(BATCH_SIZE, SEQ_LENGTH)).to(device)
    return input_ids, labels


def profile_mem(f, arg1=None):
    """Profile memory usage of a function."""
    torch.cuda.reset_peak_memory_stats()
    m1_max = torch.cuda.max_memory_allocated() / 2**20
    m1 = torch.cuda.memory_allocated() / 2**20

    if arg1 is not None:
        ret = f(arg1)
    else:
        ret = f()

    m2_max_mem = torch.cuda.max_memory_allocated() / 2**20
    m2 = torch.cuda.memory_allocated() / 2**20
    print(f"Mem history: {m1:.2f} -> {m2_max_mem:.2f} -> {m2:.2f} MB")
    print(f"Max Mem difference: {m2_max_mem-m1:.2f} MB")
    return ret, m2_max_mem - m1


def model_parallel(rank, world_size):
    """Run LLaMA-7B model with LoRA, TP, and DP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12347"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] STRESS TEST STARTING")
    print(f"{'='*60}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Get input and labels
    input_ids, labels = get_input_and_output(device)
    print(f"[Rank {rank}] Input shape: {input_ids.shape}")
    print(f"[Rank {rank}] Total tokens: {BATCH_SIZE * SEQ_LENGTH:,}")

    # Define LLaMA-7B model
    print(f"[Rank {rank}] Creating LLaMA-7B model...")
    model_config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        max_position_embeddings=SEQ_LENGTH,
        vocab_size=EMBEDDING_LENGTH,
    )

    # Calculate model size
    total_params = (
        EMBEDDING_LENGTH * HIDDEN_SIZE  # embeddings
        + NUM_LAYERS
        * (
            4 * HIDDEN_SIZE * HIDDEN_SIZE  # attention
            + 3 * HIDDEN_SIZE * INTERMEDIATE_SIZE  # MLP
            + 2 * HIDDEN_SIZE  # norms
        )
        + HIDDEN_SIZE  # final norm
    )
    model_size_gb = total_params * 4 / (1024**3)  # 4 bytes per param (float32)
    print(
        f"[Rank {rank}] Model size: ~{model_size_gb:.1f} GB ({total_params/1e9:.1f}B parameters)"
    )

    torch.cuda.reset_peak_memory_stats()
    llama_model = LlamaForCausalLM(model_config).to(device)
    llama_model.train()

    m1 = torch.cuda.memory_allocated() / 2**20
    print(f"[Rank {rank}] Base model loaded: {m1:.0f} MB")

    # Add LoRA adapters
    print(f"[Rank {rank}] Adding LoRA adapters (rank={LORA_RANK})...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
    )
    llama_model = get_peft_model(llama_model, lora_config)

    # Count trainable params
    trainable_params = sum(
        p.numel() for p in llama_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in llama_model.parameters())
    print(
        f"[Rank {rank}] Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    m2 = torch.cuda.memory_allocated() / 2**20
    print(f"[Rank {rank}] After LoRA: {m2:.0f} MB")

    # Define tensor parallelization plan for LoRA adapters
    tp_mesh = init_device_mesh("cuda", (world_size,))

    print(f"[Rank {rank}] Applying tensor parallelism...")
    for layer_id, transformer_block in enumerate(
        llama_model.base_model.model.model.layers
    ):
        layer_tp_plan = {
            # Attention LoRA adapters
            "self_attn.q_proj.lora_A.default": ColwiseParallel(),
            "self_attn.q_proj.lora_B.default": RowwiseParallel(),
            "self_attn.k_proj.lora_A.default": ColwiseParallel(),
            "self_attn.k_proj.lora_B.default": RowwiseParallel(),
            "self_attn.v_proj.lora_A.default": ColwiseParallel(),
            "self_attn.v_proj.lora_B.default": RowwiseParallel(),
            "self_attn.o_proj.lora_A.default": ColwiseParallel(),
            "self_attn.o_proj.lora_B.default": RowwiseParallel(),
            # MLP LoRA adapters
            "mlp.gate_proj.lora_A.default": ColwiseParallel(),
            "mlp.gate_proj.lora_B.default": RowwiseParallel(),
            "mlp.up_proj.lora_A.default": ColwiseParallel(),
            "mlp.up_proj.lora_B.default": RowwiseParallel(),
            "mlp.down_proj.lora_A.default": ColwiseParallel(),
            "mlp.down_proj.lora_B.default": RowwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    m3 = torch.cuda.memory_allocated() / 2**20
    print(f"[Rank {rank}] After TP sharding: {m3:.0f} MB (saved {m2-m3:.0f} MB)")

    # Create optimizer
    optimizer = torch.optim.AdamW(llama_model.parameters(), lr=1e-4)

    # Create dummy data loader
    dummy_dataset = torch.utils.data.TensorDataset(input_ids, labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=BATCH_SIZE)

    # Setup privacy engine
    print(f"[Rank {rank}] Setting up privacy engine...")
    privacy_engine = PrivacyEngineGradSampleController()

    controller, optimizer, _ = privacy_engine.make_private(
        module=llama_model,
        optimizer=optimizer,
        data_loader=dummy_loader,
        noise_multiplier=0.5,  # Moderate noise
        max_grad_norm=1.0,
        grad_sample_mode="tp",
        poisson_sampling=False,
        return_controller=True,
    )

    print(f"[Rank {rank}] Setup complete! Starting training...")
    print(f"{'='*60}\n")

    # Forward pass
    print(f"[Rank {rank}] Forward pass...")
    forward_start = time.time()
    output, forward_mem = profile_mem(lambda: llama_model(input_ids))
    forward_time = time.time() - forward_start
    print(f"[Rank {rank}] Forward time: {forward_time:.2f}s")

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output.logits.view(-1, EMBEDDING_LENGTH), labels.view(-1))
    print(f"[Rank {rank}] Loss: {loss.item():.4f}")

    # Backward pass
    print(f"[Rank {rank}] Backward pass (computing per-sample gradients)...")
    backward_start = time.time()
    _, backward_mem = profile_mem(loss.backward)
    backward_time = time.time() - backward_start
    print(f"[Rank {rank}] Backward time: {backward_time:.2f}s")

    # Get per-sample norms
    per_sample_norms = controller.get_per_sample_norms()
    print(
        f"[Rank {rank}] Per-sample norms - min: {per_sample_norms.min():.2f}, "
        f"max: {per_sample_norms.max():.2f}, mean: {per_sample_norms.mean():.2f}"
    )

    # Optimizer step
    print(f"[Rank {rank}] Optimizer step...")
    step_start = time.time()
    _, step_mem = profile_mem(optimizer.step)
    step_time = time.time() - step_start
    print(f"[Rank {rank}] Step time: {step_time:.2f}s")

    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_sec = BATCH_SIZE * SEQ_LENGTH / total_time

    print(f"\n{'='*60}")
    print(f"[Rank {rank}] STRESS TEST COMPLETE!")
    print(f"{'='*60}")
    print(f"[Rank {rank}] Total time: {total_time:.2f}s")
    print(f"[Rank {rank}] Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(
        f"[Rank {rank}] Forward: {forward_time:.2f}s, Backward: {backward_time:.2f}s, Step: {step_time:.2f}s"
    )
    print(
        f"[Rank {rank}] Memory - Forward: +{forward_mem:.0f}MB, Backward: +{backward_mem:.0f}MB, Step: +{step_mem:.0f}MB"
    )
    print(f"{'='*60}\n")

    dist.destroy_process_group()


def main():
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*70}")
    print(f"LLAMA-7B + LORA + TP + DP STRESS TEST")
    print(f"{'='*70}")
    print(f"Available GPUs: {n_gpus}")

    assert n_gpus >= 2, f"Requires at least 2 GPUs, got {n_gpus}"

    # Use all 8 GPUs for maximum parallelism
    world_size = min(4, n_gpus)

    print(f"\nConfiguration:")
    print(f"  Model: LLaMA-7B (~7B params, {HIDDEN_SIZE}d, {NUM_LAYERS} layers)")
    print(
        f"  LoRA: rank={LORA_RANK}, ~{LORA_RANK * HIDDEN_SIZE * 14 * NUM_LAYERS / 1e6:.0f}M trainable params"
    )
    print(f"  Tensor Parallelism: {world_size} GPUs")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Total tokens per batch: {BATCH_SIZE * SEQ_LENGTH:,}")
    print(f"  Differential Privacy: enabled (noise_multiplier=0.5)")
    print(f"{'='*70}\n")

    try:
        mp.spawn(model_parallel, args=(world_size,), nprocs=world_size, join=True)
        print("\n" + "=" * 70)
        print("STRESS TEST PASSED - All ranks completed successfully!")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"STRESS TEST FAILED: {e}")
        print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    main()
