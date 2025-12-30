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
HuggingFace Trainer with Differential Privacy support.

This example demonstrates a robust integration of Opacus with HuggingFace Trainer,
using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning with
differential privacy guarantees.

Example Launch Commands
-----------------------

**Single GPU:**

    CUDA_VISIBLE_DEVICES=0 python huggingface_trainer.py \\
        --train_samples 1000 --eval_samples 200 --epochs 3

**DDP (Distributed Data Parallel) with multiple GPUs:**

    accelerate launch --multi_gpu --num_processes 2 huggingface_trainer.py \\
        --train_samples 1000 --eval_samples 200 --epochs 3

**FSDP2 (Fully Sharded Data Parallel v2) with multiple GPUs:**

    accelerate launch \\
        --use_fsdp \\
        --fsdp_version 2 \\
        --fsdp_cpu_ram_efficient_loading false \\
        --num_processes 2 \\
        huggingface_trainer.py \\
        --train_samples 1000 --eval_samples 200 --epochs 3

**FP8 Mixed Precision (requires compatible GPU and TransformerEngine/MS-AMP):**

    accelerate launch --mixed_precision fp8 --fp8_backend te huggingface_trainer.py \\
        --train_samples 1000 --eval_samples 200 --epochs 3

Supported Configurations
------------------------

- Single GPU: Fully supported
- DDP (via accelerate): Supported
- FSDP2 (via accelerate): Supported with LoRA-style fine-tuning
- FP8 Mixed Precision: Supported (uses functorch fallback for TE/MS-AMP layers)
- FSDP1: **Not supported** - FSDP1 flattens multiple parameters into a single
  FlatParameter tensor, making per-sample gradient computation impossible.
  Use FSDP2 (--fsdp_version 2) instead.
- DataParallel (nn.DataParallel): Not supported (incompatible with Opacus)

Notes
-----

- LoRA is enabled by default. Use --no_lora to disable (not recommended for FSDP).
- The grad_sample_mode is automatically adjusted based on the distributed configuration.
- For FSDP2, ignored_modules are auto-detected based on trainable parameters.
- cpu_ram_efficient_loading is auto-disabled for DP training compatibility.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import datasets
import torch
from huggingface_utils import DPTrainer, PrivacyArguments
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    logging,
)


logger = logging.get_logger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Opacus HuggingFace Trainer Example")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Model name from HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Dataset name from HuggingFace Hub",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=2000,
        help=(
            "Number of training samples to use. Set to -1 to use the full train split. "
            "(Note: full splits can take a long time under DP + FSDP.)"
        ),
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=500,
        help=(
            "Number of evaluation samples to use. Set to -1 to use the full eval/test split."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Accumulation steps"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--target_epsilon", type=float, default=9.0, help="Target epsilon"
    )
    parser.add_argument(
        "--no_lora", action="store_true", help="Disable LoRA", default=False
    )
    parser.add_argument(
        "--grad_sample_mode",
        type=str,
        default="hooks",
        choices=["hooks", "functorch"],
        help="Opacus grad_sample_mode (hooks or functorch). For FSDP/multi-GPU, automatically adjusted to hooks_fsdp.",
    )
    parser.add_argument(
        "--hf_fsdp",
        action="store_true",
        default=False,
        help=(
            "Use HuggingFace Trainer's built-in FSDP (`TrainingArguments.fsdp=...`). "
            "If not set, rely on `accelerate launch --use_fsdp` for FSDP."
        ),
    )
    args = parser.parse_args()

    # 1. Load dataset
    raw_datasets = datasets.load_dataset(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # If a dedicated eval/test split exists, prefer it. Otherwise, carve eval from train.
    train_split = raw_datasets["train"].shuffle(seed=42).rename_column("label", "labels")
    eval_split = (
        raw_datasets["test"].rename_column("label", "labels")
        if "test" in raw_datasets
        else train_split
    )

    if args.train_samples is not None and args.train_samples > 0:
        train_split = train_split.select(range(min(args.train_samples, len(train_split))))
    if args.eval_samples is not None and args.eval_samples > 0:
        eval_split = eval_split.select(range(min(args.eval_samples, len(eval_split))))

    train_dataset = train_split.map(tokenize_function, batched=True)
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    eval_dataset = eval_split.map(tokenize_function, batched=True)
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # 2. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    if not args.no_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
        )
        model = get_peft_model(model, peft_config)
        print("LoRA applied to the model.")

    # 3. Define training arguments
    training_args = TrainingArguments(
        output_dir="test_trainer",
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=1,
        restore_callback_states_from_checkpoint=True,
        report_to=None,
        fsdp="full_shard auto_wrap" if args.hf_fsdp else "",
        fsdp_config={"transformer_layer_cls_to_wrap": "TransformerBlock"}
        if args.hf_fsdp
        else None,
    )

    # 4. Initialize PrivacyArguments
    privacy_args = PrivacyArguments(
        target_epsilon=args.target_epsilon,
        target_delta=1e-5,
        grad_sample_mode=args.grad_sample_mode,
    )

    # 5. Initialize DPTrainer
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        privacy_args=privacy_args,
    )

    # 6. Train
    print("Starting training with DPTrainer...")
    trainer.train()

    # 7. Final epsilon
    # Epsilon is also available in metrics due to compute_privacy_metrics
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")

    # 8. Cleanup
    trainer.detach_model()


if __name__ == "__main__":
    main()
