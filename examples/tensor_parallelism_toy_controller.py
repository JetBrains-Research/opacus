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
Simple toy example demonstrating GradSampleControllerTP with tensor parallelism.

This example uses a 2-layer linear network with ColwiseParallel and RowwiseParallel
to validate that the TP controller properly aggregates per-sample gradient norms.
"""

import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.data import TensorDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleModule(nn.Module):
    """Simple 2-layer linear network for testing."""

    def __init__(self):
        super(SampleModule, self).__init__()
        self.fc1 = nn.Linear(4, 32, bias=False)
        self.fc2 = nn.Linear(32, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).flatten(start_dim=1)
        x = F.softmax(x, dim=-1)
        return x


def model_parallel(rank, world_size, m):
    """Run model with tensor parallelism using GradSampleControllerTP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"[Rank {rank}] Initializing...")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create input data
    input_data = torch.tensor(
        [
            [[-1.0, -1.0, -1.0, -1.0], [-2.0, -2.0, -2.0, -2.0]],
            [[-3.0, -3.0, -3.0, -3.0], [-4.0, -4.0, -4.0, -4.0]],
            [[-5.0, -5.0, -5.0, -5.0], [-6.0, -6.0, -6.0, -6.0]],
        ]
    ).to(device)
    labels = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).to(device)
    input_dataset = TensorDataset(input_data, labels)
    input_dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=2)
    print(f"[Rank {rank}] Input shape: {input_data.shape}")

    # Initialize device mesh for tensor parallelism
    tp_mesh = init_device_mesh("cuda", (world_size,))

    # Apply tensor parallelism to model
    # fc1: ColwiseParallel - split columns across devices
    # fc2: RowwiseParallel - split rows across devices
    sharded_model = parallelize_module(
        m.to(device), tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    )

    print(f"[Rank {rank}] Model parallelized")

    # Create optimizer
    optimizer = torch.optim.SGD(sharded_model.parameters(), lr=1.0)

    # Setup privacy engine with TP-aware controller
    privacy_engine = PrivacyEngineGradSampleController()

    # Make model private using TP grad_sample_mode
    controller, optimizer, _ = privacy_engine.make_private(
        module=sharded_model,
        optimizer=optimizer,
        data_loader=input_dataloader,
        noise_multiplier=0.0,  # No noise for testing
        max_grad_norm=0.1,
        grad_sample_mode="tp",  # Use TP-aware controller
        poisson_sampling=False,
        return_controller=True,  # Get controller for inspection
    )

    print(f"[Rank {rank}] Privacy engine configured")

    # Create loss criterion
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    # Forward pass
    optimizer.zero_grad()

    input_data, label = next(iter(input_dataloader))

    output = sharded_model(input_data)
    print(f"[Rank {rank}] Output shape: {output.shape}")

    # Create label and compute loss
    loss = criterion(output, label)

    # Backward pass
    loss.backward()

    print(f"[Rank {rank}] Backward pass complete")

    # Get per-sample norms from TP controller
    per_sample_norms = controller.get_per_sample_norms()
    print(f"[Rank {rank}] Per-sample norms: {per_sample_norms}")

    # Optimizer step (with clipping and noise)
    optimizer.step()

    # Print final gradients
    print(f"[Rank {rank}] Final parameter gradients:")
    for name, param in sharded_model.named_parameters():
        if param.grad is not None:
            print(f"[Rank {rank}]   {name}: grad_norm={param.grad.norm().item():.6f}")

    print(f"[Rank {rank}] Training step complete!")

    dist.barrier()
    if rank == 0:
        print("\n=== All ranks completed successfully ===\n")

    dist.destroy_process_group()


def main():
    """Main entry point."""
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"ERROR: Requires at least 2 GPUs to run, but got {n_gpus}")
        print("This example demonstrates tensor parallelism across multiple devices.")
        return

    print(f"Running with {n_gpus} GPUs")
    world_size = 2  # Use 2 GPUs for simplicity

    # Create model (will be copied to each process)
    m = SampleModule()

    # Spawn processes for distributed training
    mp.spawn(
        model_parallel,
        args=(world_size, copy.deepcopy(m)),
        nprocs=world_size,
        join=True,
    )

    print("=== Example completed successfully ===")


if __name__ == "__main__":
    main()
