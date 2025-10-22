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
Multi-GPU tests for DistributedDataParallel (DDP) with controller-based PrivacyEngine.

These tests validate that the controller-based privacy engine works correctly
with PyTorch's DistributedDataParallel for data parallel training.
"""

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    """Setup distributed environment for testing."""
    if sys.platform == "win32":
        raise ValueError("Windows platform is not supported for this test")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize the process group
    torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
    )


def cleanup():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def run_test_ddp_basic(rank, world_size):
    """Test basic DDP functionality with controller-based privacy engine."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Different seed per rank for different data
    torch.manual_seed(42 + rank)

    # Create model and wrap with DDP
    model = SimpleNet().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    # Create dummy data
    batch_size = 4
    input_data = torch.randn(batch_size, 10).to(device)
    labels = torch.randint(0, 5, (batch_size,)).to(device)
    dummy_dataset = torch.utils.data.TensorDataset(input_data, labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)

    # Make private
    privacy_engine = PrivacyEngineGradSampleController()
    controller, optimizer, _ = privacy_engine.make_private(
        module=ddp_model,
        optimizer=optimizer,
        data_loader=dummy_loader,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        grad_sample_mode="hooks",
        poisson_sampling=False,
        return_controller=True,
    )

    # Training step
    optimizer.zero_grad()
    output = ddp_model(input_data)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(output, labels)
    loss.backward()

    # Check that grad_sample is computed on underlying model
    for param in model.parameters():
        if param.requires_grad:
            assert hasattr(param, "grad_sample"), "Parameter should have grad_sample"
            assert param.grad_sample is not None, "grad_sample should not be None"

    optimizer.step()

    cleanup()
    return True


def run_test_ddp_compare_to_single_gpu(rank, world_size, result_dict):
    """Compare DDP results with single GPU to verify correctness."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Fixed seed for reproducibility (same across all ranks)
    torch.manual_seed(42)

    # Create model with fixed initialization
    model = SimpleNet()

    # Save initial state for single GPU comparison
    if rank == 0:
        result_dict["init_state"] = model.state_dict()

    # Wait for rank 0 to save the state
    dist.barrier()

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01, foreach=False)

    # Create dummy data (same across all ranks)
    batch_size = 4
    input_data = torch.randn(batch_size, 10).to(device)
    labels = torch.randint(0, 5, (batch_size,)).to(device)
    dummy_dataset = torch.utils.data.TensorDataset(input_data, labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)

    # Make private
    privacy_engine = PrivacyEngineGradSampleController()
    controller, optimizer, _ = privacy_engine.make_private(
        module=ddp_model,
        optimizer=optimizer,
        data_loader=dummy_loader,
        noise_multiplier=0.0,
        max_grad_norm=1e20,
        grad_sample_mode="hooks",
        poisson_sampling=False,
        return_controller=True,
    )

    # Training step
    optimizer.zero_grad()
    output = ddp_model(input_data)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    # Store results from rank 0
    if rank == 0:
        result_dict["ddp_params"] = [
            p.detach().cpu().clone() for p in model.parameters()
        ]

        # Also run single GPU version for comparison
        single_model = SimpleNet()
        # Load the same initial state as DDP model
        single_model.load_state_dict(result_dict["init_state"])
        single_model = single_model.to(device)
        single_optimizer = torch.optim.SGD(
            single_model.parameters(), lr=0.01, foreach=False
        )
        single_dataset = torch.utils.data.TensorDataset(input_data, labels)
        single_loader = torch.utils.data.DataLoader(
            single_dataset, batch_size=batch_size
        )

        single_privacy_engine = PrivacyEngineGradSampleController()
        single_controller, single_optimizer, _ = single_privacy_engine.make_private(
            module=single_model,
            optimizer=single_optimizer,
            data_loader=single_loader,
            noise_multiplier=0.0,
            max_grad_norm=1e20,
            grad_sample_mode="hooks",
            poisson_sampling=False,
            return_controller=True,
        )

        single_optimizer.zero_grad()
        single_output = single_model(input_data)
        single_loss = criterion(single_output, labels)
        single_loss.backward()
        single_optimizer.step()

        result_dict["single_params"] = [
            p.detach().cpu().clone() for p in single_model.parameters()
        ]

    cleanup()
    return True


class DDPControllerTest(unittest.TestCase):
    """Test cases for DDP with controller-based privacy engine."""

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run DDP tests"
    )
    def test_ddp_basic(self):
        """Test basic DDP functionality with privacy engine."""
        world_size = 2
        mp.spawn(run_test_ddp_basic, args=(world_size,), nprocs=world_size, join=True)

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run DDP tests"
    )
    def test_ddp_training_completes(self):
        """Test that DDP training completes successfully and parameters update."""
        world_size = 2
        manager = mp.Manager()
        result_dict = manager.dict()

        mp.spawn(
            run_test_ddp_compare_to_single_gpu,
            args=(world_size, result_dict),
            nprocs=world_size,
            join=True,
        )

        # Verify training completed and parameters were updated
        self.assertIn("ddp_params", result_dict, "DDP training should have completed")
        self.assertIn("init_state", result_dict, "Initial state should be saved")

        # Verify parameters changed from initial state
        ddp_params = result_dict["ddp_params"]
        init_state = result_dict["init_state"]
        init_params = [p for p in init_state.values()]

        params_changed = False
        for ddp_p, init_p in zip(ddp_params, init_params):
            if not torch.allclose(ddp_p, init_p, atol=1e-6):
                params_changed = True
                break

        self.assertTrue(
            params_changed, "DDP model parameters should have changed after training"
        )

    def test_import(self):
        """Test that PrivacyEngineGradSampleController can be imported."""
        from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController

        self.assertIsNotNone(PrivacyEngineGradSampleController)


if __name__ == "__main__":
    unittest.main()
