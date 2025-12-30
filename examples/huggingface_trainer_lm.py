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
HuggingFace Trainer with Differential Privacy for Causal Language Modeling.

This example demonstrates Opacus integration with HuggingFace Trainer for
causal language modeling (text generation) tasks using LoRA for parameter-efficient
fine-tuning with differential privacy guarantees.

Supports all parallelism modes including Context Parallelism (CP) which requires
causal language modeling tasks.

Example Launch Commands
-----------------------

**Single GPU:**

    CUDA_VISIBLE_DEVICES=0 python huggingface_trainer_lm.py \\
        --train_samples 500 --epochs 1

**DDP (Distributed Data Parallel) with multiple GPUs:**

    accelerate launch --multi_gpu --num_processes 2 huggingface_trainer_lm.py \\
        --train_samples 500 --epochs 1

**FSDP2 (Fully Sharded Data Parallel v2) with multiple GPUs:**

    accelerate launch \\
        --use_fsdp \\
        --fsdp_version 2 \\
        --fsdp_cpu_ram_efficient_loading false \\
        --num_processes 2 \\
        huggingface_trainer_lm.py \\
        --train_samples 500 --epochs 1

**Context Parallelism (CP) - requires FSDP2:**

    accelerate launch \\
        --use_fsdp \\
        --fsdp_version 2 \\
        --fsdp_cpu_ram_efficient_loading false \\
        --use_parallelism_config \\
        --parallelism_config_cp_size 2 \\
        --parallelism_config_dp_shard_size 1 \\
        --num_processes 2 \\
        huggingface_trainer_lm.py \\
        --train_samples 500 --epochs 1

**Tensor Parallelism (TP) + FSDP2 - requires 4+ GPUs:**

    accelerate launch \\
        --use_fsdp \\
        --fsdp_version 2 \\
        --fsdp_cpu_ram_efficient_loading false \\
        --use_parallelism_config \\
        --parallelism_config_tp_size 2 \\
        --parallelism_config_dp_shard_size 2 \\
        --num_processes 4 \\
        huggingface_trainer_lm.py \\
        --train_samples 500 --epochs 1

**Adaptive Clipping (auto-adjusting gradient clipping threshold):**

    CUDA_VISIBLE_DEVICES=0 python huggingface_trainer_lm.py \\
        --train_samples 500 --epochs 1 --clipping adaptive

**Adaptive Clipping with FSDP2:**

    accelerate launch \\
        --use_fsdp \\
        --fsdp_version 2 \\
        --fsdp_cpu_ram_efficient_loading false \\
        --num_processes 2 \\
        huggingface_trainer_lm.py \\
        --train_samples 500 --epochs 1 --clipping adaptive

Supported Configurations
------------------------

- Single GPU: Fully supported
- DDP (via accelerate): Supported
- FSDP2 (via accelerate): Supported with LoRA-style fine-tuning
- Context Parallelism (CP): Supported (requires causal LM task)
- Tensor Parallelism (TP): Beta support
- FSDP1: **Not supported**

Clipping Modes
--------------

- **flat** (default): Fixed gradient clipping threshold (per_sample_max_grad_norm=0.5)
- **adaptive**: Auto-adjusting threshold based on gradient statistics. Uses the
  algorithm from "Differentially Private Learning with Adaptive Clipping"
  (https://arxiv.org/abs/1905.03871). The clipping bound adjusts to maintain
  a target fraction of unclipped samples (default: 50%).
- **per_layer**: Per-parameter gradient clipping. Each parameter has its own
  clipping threshold (defaults to per_sample_max_grad_norm for all parameters).
  This can provide better utility in some cases by allowing different layers
  to have different clipping bounds.

Notes
-----

- LoRA is enabled by default for memory efficiency.
- Uses GPT-2 by default for quick testing. Use --model_name for other models.
- For LLaMA models, you may need to authenticate with HuggingFace.
- CP splits the sequence dimension across devices using ring attention.
- All clipping modes (flat, adaptive, per_layer) work with all distributed modes.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import datasets
import torch
from huggingface_utils import DPTrainer, PrivacyArguments
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    logging,
)


logger = logging.get_logger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Opacus HuggingFace Trainer for Causal Language Modeling"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name from HuggingFace Hub (default: gpt2). For LLaMA: meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name from HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1000,
        help="Number of training samples to use. Set to -1 for full dataset.",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=200,
        help="Number of evaluation samples to use. Set to -1 for full dataset.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=8.0,
        help="Target epsilon for differential privacy",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        default=False,
        help="Disable LoRA (not recommended for large models)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--grad_sample_mode",
        type=str,
        default="hooks",
        choices=["hooks", "functorch"],
        help="Opacus grad_sample_mode (auto-adjusted for distributed)",
    )
    parser.add_argument(
        "--clipping",
        type=str,
        default="flat",
        choices=["flat", "adaptive", "per_layer"],
        help="Clipping mode: 'flat' (fixed threshold), 'adaptive' (auto-adjusting), or 'per_layer' (per-parameter clipping)",
    )
    parser.add_argument(
        "--target_unclipped_quantile",
        type=float,
        default=0.5,
        help="Target fraction of unclipped samples for adaptive clipping (default: 0.5)",
    )
    parser.add_argument(
        "--clipbound_learning_rate",
        type=float,
        default=0.2,
        help="Learning rate for adaptive clipping bound updates (default: 0.2)",
    )
    args = parser.parse_args()

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config}")
    raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config)

    # Tokenize function for causal LM
    def tokenize_function(examples):
        # Filter out empty texts
        texts = [t for t in examples["text"] if t and len(t.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors=None,
        )

    # Process datasets
    train_split = raw_datasets["train"]
    eval_split = raw_datasets.get("validation", raw_datasets.get("test", train_split))

    if args.train_samples > 0:
        train_split = train_split.select(range(min(args.train_samples, len(train_split))))
    if args.eval_samples > 0:
        eval_split = eval_split.select(range(min(args.eval_samples, len(eval_split))))

    # Tokenize
    train_dataset = train_split.map(
        tokenize_function,
        batched=True,
        remove_columns=train_split.column_names,
    )
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    eval_dataset = eval_split.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_split.column_names,
    )
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # 3. Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use float32 for DP training stability
        attn_implementation="sdpa",  # Required for CP
    )

    # Ensure model uses the same pad token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Apply LoRA
    if not args.no_lora:
        # Determine target modules based on model architecture
        if "gpt2" in args.model_name.lower():
            target_modules = ["c_attn", "c_proj"]
        elif "llama" in args.model_name.lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "opt" in args.model_name.lower():
            target_modules = ["q_proj", "v_proj"]
        else:
            # Generic fallback
            target_modules = ["q_proj", "v_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("LoRA applied to the model.")

    # 5. Data collator for causal LM (sets labels = input_ids)
    # For CP compatibility, we use a custom collator that removes attention_mask
    # to let the model use its default causal masking
    class CPCompatibleDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features, return_tensors=None):
            batch = super().__call__(features, return_tensors=return_tensors)
            # Remove attention_mask for CP compatibility - model will use default causal mask
            # This is needed because CP splits the sequence and the attention mask
            # shape becomes incompatible with the split sequence
            if "attention_mask" in batch:
                del batch["attention_mask"]
            return batch
    
    data_collator = CPCompatibleDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="./output_lm",
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",  # Disable checkpointing for simplicity
        report_to=None,
        remove_unused_columns=False,  # Keep all columns for CP
        dataloader_pin_memory=False,  # Avoid issues with distributed
    )

    # 7. Privacy arguments
    privacy_args = PrivacyArguments(
        target_epsilon=args.target_epsilon,
        target_delta=1e-5,
        grad_sample_mode=args.grad_sample_mode,
        clipping=args.clipping,
        target_unclipped_quantile=args.target_unclipped_quantile,
        clipbound_learning_rate=args.clipbound_learning_rate,
    )

    # 8. Initialize DPTrainer
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        privacy_args=privacy_args,
    )

    # 9. Train
    print("Starting training with DPTrainer...")
    trainer.train()

    # 10. Evaluate
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")

    # Calculate perplexity from loss
    if "eval_loss" in metrics:
        import math
        perplexity = math.exp(metrics["eval_loss"])
        print(f"Perplexity: {perplexity:.2f}")

    # 11. Cleanup
    trainer.detach_model()
    print("Training complete!")


if __name__ == "__main__":
    main()
