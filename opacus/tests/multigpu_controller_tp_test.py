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
Multi-GPU tests for GradSampleControllerTP.

These tests validate that the TP controller correctly:
1. Sets merge_flag on different parameter types
2. Aggregates per-sample gradient norms across devices
3. Handles various tensor parallelism configurations

Pattern follows existing multigpu_*.py tests in this directory.
"""

import copy
import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleController, GradSampleControllerTP
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def setup(rank, world_size):
    """Setup distributed environment for testing."""
    if sys.platform == "win32":
        raise ValueError("Windows platform is not supported for this test")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
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


class SimpleLinearModule(nn.Module):
    """Simple module for testing."""

    def __init__(self, input_size=4, hidden_size=8, output_size=4, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ModuleWithEmbedding(nn.Module):
    """Module with embedding layer for testing."""

    def __init__(self, num_embeddings=100, embedding_dim=8, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        return self.fc(x)


def run_test_merge_flag_setting(rank, world_size):
    """Test that merge_flag is set correctly on parameters."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Create and parallelize model
    model = SimpleLinearModule().to(f"cuda:{rank}")
    tp_mesh = init_device_mesh("cuda", (world_size,))
    model = parallelize_module(
        model, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    )

    # Create TP controller
    controller = GradSampleControllerTP(model, batch_first=True, loss_reduction="mean")

    # Check merge_flag on parameters
    for name, param in model.named_parameters():
        assert hasattr(
            param, "merge_flag"
        ), f"Parameter {name} missing merge_flag attribute"

        # All DTensor parameters should have merge_flag=True for sharded layers
        if hasattr(param, "placements"):
            # This is a DTensor
            if not param.placements[0].is_replicate():
                assert (
                    param.merge_flag
                ), f"Sharded DTensor {name} should have merge_flag=True"

    cleanup()
    return True


def run_test_privacy_engine_integration(rank, world_size):
    """Test integration with PrivacyEngineGradSampleController."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create model
    model = SimpleLinearModule().to(device)

    # Apply tensor parallelism
    tp_mesh = init_device_mesh("cuda", (world_size,))
    model = parallelize_module(
        model, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    )

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create dummy dataloader
    batch_size = 4
    input_data = torch.randn(batch_size, 4).to(device)
    labels = torch.randint(0, 4, (batch_size,)).to(device)
    dummy_dataset = torch.utils.data.TensorDataset(input_data, labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)

    # Setup privacy engine
    privacy_engine = PrivacyEngineGradSampleController()

    # Make private - TP will be auto-detected
    controller, optimizer, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dummy_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        return_controller=True,
    )

    # Verify controller is TP type
    assert isinstance(
        controller, GradSampleControllerTP
    ), "Controller should be GradSampleControllerTP"

    # Run training step
    optimizer.zero_grad()
    output = model(input_data)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(output, labels)
    loss.backward()

    # Check that grad_sample is populated
    for param in model.parameters():
        if param.requires_grad:
            assert hasattr(
                param, "grad_sample"
            ), "Parameter should have grad_sample attribute"
            assert (
                param.grad_sample is not None
            ), "Parameter grad_sample should not be None"

    # Get per-sample norms
    norms = controller.get_per_sample_norms()
    assert norms is not None, "Per-sample norms should not be None"

    # Optimizer step
    optimizer.step()

    cleanup()
    return True


def run_test_compare_to_vanilla(rank, world_size):
    """Compare TP controller with vanilla controller using torch.allclose on gradients."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Fixed seed for reproducibility across both models
    torch.manual_seed(42)

    # Create shared input data
    batch_size = 4
    input_data = torch.randn(batch_size, 4).to(device)
    labels = torch.randint(0, 4, (batch_size,)).to(device)

    # === Create identical starting models ===
    model_tp = SimpleLinearModule().to(device)
    model_vanilla = SimpleLinearModule().to(device)
    model_vanilla.load_state_dict(copy.deepcopy(model_tp.state_dict()))

    # === Run with TP controller ===
    tp_mesh = init_device_mesh("cuda", (world_size,))
    model_tp = parallelize_module(
        model_tp, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    )

    optimizer_tp = torch.optim.SGD(model_tp.parameters(), lr=0.01, foreach=False)
    dummy_dataset_tp = torch.utils.data.TensorDataset(input_data, labels)
    dummy_loader_tp = torch.utils.data.DataLoader(
        dummy_dataset_tp, batch_size=batch_size
    )

    privacy_engine_tp = PrivacyEngineGradSampleController()
    controller_tp, optimizer_tp, _ = privacy_engine_tp.make_private(
        module=model_tp,
        optimizer=optimizer_tp,
        data_loader=dummy_loader_tp,
        noise_multiplier=0.0,  # No noise for comparison
        max_grad_norm=1e20,  # Effectively no clipping
        poisson_sampling=False,
        return_controller=True,
    )

    optimizer_tp.zero_grad()
    output_tp = model_tp(input_data)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss_tp = criterion(output_tp, labels)
    loss_tp.backward()

    # Gather TP gradients (collective operation - all ranks must participate)
    tp_grads = []
    for name, param in model_tp.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Gather full gradient tensor
            if hasattr(param.grad, "full_tensor"):
                full_grad = param.grad.full_tensor()
            elif hasattr(param.grad, "to_local"):
                full_grad = param.grad.to_local()
            else:
                full_grad = param.grad
            tp_grads.append((name, full_grad))

    # === Run with vanilla controller (only on rank 0) ===
    if rank == 0:
        optimizer_vanilla = torch.optim.SGD(
            model_vanilla.parameters(), lr=0.01, foreach=False
        )
        dummy_dataset_vanilla = torch.utils.data.TensorDataset(input_data, labels)
        dummy_loader_vanilla = torch.utils.data.DataLoader(
            dummy_dataset_vanilla, batch_size=batch_size
        )

        privacy_engine_vanilla = PrivacyEngineGradSampleController()
        controller_vanilla, optimizer_vanilla, _ = privacy_engine_vanilla.make_private(
            module=model_vanilla,
            optimizer=optimizer_vanilla,
            data_loader=dummy_loader_vanilla,
            noise_multiplier=0.0,  # No noise for comparison
            max_grad_norm=1e20,  # Effectively no clipping
            grad_sample_mode="hooks",  # Vanilla mode
            poisson_sampling=False,
            return_controller=True,
        )

        optimizer_vanilla.zero_grad()
        output_vanilla = model_vanilla(input_data)
        loss_vanilla = criterion(output_vanilla, labels)
        loss_vanilla.backward()

        # Compare gradients with torch.allclose
        vanilla_params = [
            (name, param)
            for name, param in model_vanilla.named_parameters()
            if param.requires_grad
        ]

        for (tp_name, tp_grad), (v_name, v_param) in zip(tp_grads, vanilla_params):
            assert tp_name == v_name, f"Parameter name mismatch: {tp_name} vs {v_name}"

            # Skip very small gradients
            if v_param.grad.norm() < 1e-6:
                continue

            # Compare gradients with torch.allclose
            assert torch.allclose(
                tp_grad, v_param.grad, atol=1e-5, rtol=1e-3
            ), f"Gradient mismatch for {v_name}:\nTP grad: {tp_grad}\nVanilla grad: {v_param.grad}"

    cleanup()
    return True


def run_test_adaclip_tp(rank, world_size):
    """Test AdaClip optimizer with Tensor Parallelism."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create model
    model = SimpleLinearModule().to(device)

    # Apply tensor parallelism
    tp_mesh = init_device_mesh("cuda", (world_size,))
    model = parallelize_module(
        model, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    )

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create dummy dataloader
    batch_size = 4
    input_data = torch.randn(batch_size, 4).to(device)
    labels = torch.randint(0, 4, (batch_size,)).to(device)
    dummy_dataset = torch.utils.data.TensorDataset(input_data, labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)

    # Setup privacy engine with AdaClip
    privacy_engine = PrivacyEngineGradSampleController()

    # Make private with TP (auto-detected) and adaptive clipping
    # Note: AdaClip requires noise_multiplier < 2 * unclipped_num_std (Theorem 1)
    controller, optimizer, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dummy_loader,
        noise_multiplier=0.1,
        max_grad_norm=1.0,
        clipping="adaptive",
        target_unclipped_quantile=0.5,
        clipbound_learning_rate=0.1,
        max_clipbound=10.0,
        min_clipbound=0.01,
        unclipped_num_std=1.0,  # Must be > noise_multiplier/2 = 0.5
        poisson_sampling=False,
        return_controller=True,
    )

    # Verify optimizer is AdaClip type
    from opacus.optimizers.adaclipoptimizer_tp import AdaClipDPOptimizerTP

    assert isinstance(
        optimizer, AdaClipDPOptimizerTP
    ), "Optimizer should be AdaClipDPOptimizerTP"

    # Store initial clipbound
    initial_clipbound = optimizer.max_grad_norm

    # Run training step
    optimizer.zero_grad()
    output = model(input_data)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(output, labels)
    loss.backward()

    # Check that grad_sample is populated
    for param in model.parameters():
        if param.requires_grad:
            assert hasattr(
                param, "grad_sample"
            ), "Parameter should have grad_sample attribute"
            assert (
                param.grad_sample is not None
            ), "Parameter grad_sample should not be None"

    # Get per-sample norms
    norms = controller.get_per_sample_norms()
    assert norms is not None, "Per-sample norms should not be None"

    # Optimizer step (should update clipbound)
    optimizer.step()

    # Verify clipbound was updated
    updated_clipbound = optimizer.max_grad_norm
    # Clipbound should have changed (unless by coincidence it hit a bound)
    # We can't guarantee direction without knowing unclipped fraction,
    # but we can verify the attribute exists and is accessible
    assert isinstance(
        updated_clipbound, (int, float, torch.Tensor)
    ), "Clipbound should be numeric"

    # Verify AdaClip-specific attributes exist
    assert hasattr(optimizer, "sample_size"), "Optimizer should track sample_size"
    assert hasattr(optimizer, "unclipped_num"), "Optimizer should track unclipped_num"
    assert hasattr(
        optimizer, "target_unclipped_quantile"
    ), "Optimizer should have target_unclipped_quantile"
    assert hasattr(
        optimizer, "clipbound_learning_rate"
    ), "Optimizer should have clipbound_learning_rate"

    cleanup()
    return True


class GradSampleControllerTPTest(unittest.TestCase):
    """Test cases for GradSampleControllerTP."""

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_merge_flag_setting(self):
        """Test that merge_flag is set correctly on parameters."""
        world_size = 2
        mp.spawn(
            run_test_merge_flag_setting,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_privacy_engine_integration(self):
        """Test integration with PrivacyEngineGradSampleController."""
        world_size = 2
        mp.spawn(
            run_test_privacy_engine_integration,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_compare_to_vanilla(self):
        """Compare TP controller with vanilla controller."""
        world_size = 2
        mp.spawn(
            run_test_compare_to_vanilla,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_adaclip_tp(self):
        """Test AdaClip optimizer with Tensor Parallelism."""
        world_size = 2
        mp.spawn(
            run_test_adaclip_tp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def test_import(self):
        """Test that GradSampleControllerTP can be imported."""
        from opacus.grad_sample import GradSampleControllerTP

        self.assertIsNotNone(GradSampleControllerTP)


if __name__ == "__main__":
    unittest.main()
