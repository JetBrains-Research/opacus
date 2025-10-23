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
LLaMA example demonstrating GradSampleControllerTP with tensor parallelism.

This example trains a full LLaMA model with tensor parallelism and differential privacy.
The model is sharded across GPUs using:
- Attention Q/K/V projections: ColwiseParallel
- Attention O projection: RowwiseParallel
- MLP gate/up projections: ColwiseParallel
- MLP down projection: RowwiseParallel

Note: Embedding layers are kept replicated for compatibility with the LLaMA architecture.
Embedding sharding (RowwiseParallel) is supported and tested in simpler models.

All grad samplers (Linear, Embedding, LayerNorm, etc.) support TP with implicit_replication.
"""

import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import LlamaConfig, LlamaForCausalLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_LENGTH = 32000
SEQ_LENGTH = 1
BATCH_SIZE = 1


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
    """Run LLaMA model with tensor parallelism using GradSampleControllerTP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"[Rank {rank}] Initializing...")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Get input and labels
    input_ids, labels = get_input_and_output(device)
    print(f"[Rank {rank}] Input shape: {input_ids.shape}")

    # Define LLaMA model with smaller config for testing
    model_config = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=SEQ_LENGTH,
        vocab_size=EMBEDDING_LENGTH,
    )

    torch.cuda.reset_peak_memory_stats()
    llama_model = LlamaForCausalLM(model_config).to(device)
    llama_model.train()

    m1 = torch.cuda.memory_allocated() / 2**20
    print(f"[Rank {rank}] Initial memory: {m1:.2f} MB")

    # Define the tensor parallelization plan
    tp_mesh = init_device_mesh("cuda", (world_size,))

    # Note: Embedding sharding is supported in simple models but has issues with
    # complex models like LLaMA when used with transformers. For now, keep embedding replicated.
    # TODO: Investigate why RowwiseParallel(input_layouts=Replicate()) on embed_tokens
    # causes scatter_add_ DTensor dispatch issues.

    # Parallelize each transformer layer - now training full model!
    # All grad samplers (Linear, Embedding, LayerNorm) support TP
    for layer_id, transformer_block in enumerate(llama_model.model.layers):
        layer_tp_plan = {
            # Attention projections - standard TP pattern
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            # MLP layers - standard TP pattern
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }

        # Apply parallelization to the transformer block
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    m2_max_mem = torch.cuda.max_memory_allocated() / 2**20
    m2 = torch.cuda.memory_allocated() / 2**20
    print(
        f"[Rank {rank}] After parallelization: {m1:.2f} -> {m2_max_mem:.2f} -> {m2:.2f} MB"
    )

    print(f"[Rank {rank}] Model parallelized (full model training, no LoRA!)")

    # Create optimizer - disable foreach to avoid mixed DTensor/Tensor issues
    optimizer = torch.optim.SGD(llama_model.parameters(), lr=0.01, foreach=False)

    # Create dummy data loader (required by make_private)
    dummy_dataset = torch.utils.data.TensorDataset(input_ids, labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=BATCH_SIZE)

    # Setup privacy engine with TP-aware controller
    privacy_engine = PrivacyEngineGradSampleController()

    # Make model private - TP will be auto-detected from DTensor parameters
    controller, optimizer, _ = privacy_engine.make_private(
        module=llama_model,
        optimizer=optimizer,
        data_loader=dummy_loader,
        noise_multiplier=0.0,  # No noise for testing
        max_grad_norm=1.0,
        poisson_sampling=False,
        return_controller=True,  # Get controller for inspection
    )

    print(f"[Rank {rank}] Privacy engine configured")

    # Forward pass
    print(f"[Rank {rank}] Forward pass, memory usage:")
    output, _ = profile_mem(lambda: llama_model(input_ids))

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output.logits.view(-1, EMBEDDING_LENGTH), labels.view(-1))
    print(f"[Rank {rank}] Loss: {loss.item():.4f}")

    # Backward pass
    print(f"[Rank {rank}] Backward pass, memory usage:")
    profile_mem(loss.backward)
    print(f"[Rank {rank}] Backward pass complete")

    # Optimizer step
    print(f"[Rank {rank}] Optimizer step, memory usage:")
    profile_mem(optimizer.step)

    # Print some gradient norms
    print(f"[Rank {rank}] Final parameter gradients:")
    for name, param in llama_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"[Rank {rank}]   {name}: grad_norm={grad_norm:.6f}")
            if grad_norm > 0:  # Only print a few non-zero gradients
                break

    print(f"[Rank {rank}] Training step complete!")
    dist.destroy_process_group()


def main():
    n_gpus = torch.cuda.device_count()
    print(f"Running with {n_gpus} GPUs")
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # Use 2 GPUs for this example
    world_size = 2

    try:
        mp.spawn(model_parallel, args=(world_size,), nprocs=world_size, join=True)
        print("\n=== All ranks completed successfully ===")
        print("=== Example completed successfully ===")
    except Exception as e:
        print(f"\n=== Example failed with error: {e} ===")
        raise


if __name__ == "__main__":
    main()
