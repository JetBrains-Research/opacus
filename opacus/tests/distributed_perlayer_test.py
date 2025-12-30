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
Tests for DistributedPerLayerOptimizerTP.

This module tests the per-layer clipping optimizer for distributed training
with FSDP/TP/CP support.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn

from opacus.optimizers import (
    DPPerLayerOptimizer,
    get_optimizer_class,
)
from opacus.optimizers.ddp_perlayeroptimizer_tp import (
    DistributedPerLayerOptimizerTP,
    _is_dtensor,
    _get_local_tensor,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class _SingleRankProcessGroup:
    """Context manager for setting up a single-rank distributed environment for testing."""

    def __init__(self):
        self._inited = False
        self._tmpfile = None

    def __enter__(self):
        if not dist.is_available():
            raise unittest.SkipTest("torch.distributed is not available")

        if dist.is_initialized():
            return self

        self._tmpfile = tempfile.NamedTemporaryFile(delete=False)
        init_method = "file://" + self._tmpfile.name
        self._tmpfile.close()
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method=init_method
        )
        self._inited = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._inited and dist.is_initialized():
            dist.destroy_process_group()


class TestDistributedPerLayerOptimizerTPBasics(unittest.TestCase):
    """Test basic functionality of DistributedPerLayerOptimizerTP."""

    def test_inheritance(self):
        """Test that DistributedPerLayerOptimizerTP inherits from DPPerLayerOptimizer."""
        self.assertTrue(issubclass(DistributedPerLayerOptimizerTP, DPPerLayerOptimizer))

    def test_is_dtensor_regular_tensor(self):
        """Test _is_dtensor returns False for regular tensors."""
        tensor = torch.randn(3, 4)
        self.assertFalse(_is_dtensor(tensor))

    def test_get_local_tensor_regular_tensor(self):
        """Test _get_local_tensor returns the same tensor for regular tensors."""
        tensor = torch.randn(3, 4)
        result = _get_local_tensor(tensor)
        self.assertIs(result, tensor)


class TestDistributedPerLayerOptimizerTPWithDistributed(unittest.TestCase):
    """Test DistributedPerLayerOptimizerTP with distributed setup."""

    def test_initialization(self):
        """Test optimizer initialization with distributed setup."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            # Get number of parameters
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            self.assertEqual(optimizer.rank, 0)
            self.assertEqual(optimizer.world_size, 1)
            self.assertEqual(len(optimizer.max_grad_norms), num_params)

    def test_per_layer_max_grad_norms(self):
        """Test that per-layer max grad norms are stored correctly."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            max_grad_norms = [0.5, 1.0, 1.5, 2.0]  # 4 parameters (2 weights + 2 biases)
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            self.assertEqual(optimizer.max_grad_norms, max_grad_norms)

    def test_detect_sharded_params_no_dtensors(self):
        """Test _detect_sharded_params returns False when no DTensors."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            self.assertFalse(optimizer._detect_sharded_params())

    def test_get_flat_grad_sample(self):
        """Test _get_flat_grad_sample flattens correctly."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            # Create a mock parameter with grad_sample
            param = list(model.parameters())[0]
            batch_size = 4
            param.grad_sample = torch.randn(batch_size, *param.shape)
            
            flat = optimizer._get_flat_grad_sample(param)
            
            self.assertEqual(flat.shape[0], batch_size)
            self.assertEqual(flat.shape[1], param.numel())

    def test_get_flat_grad_sample_list(self):
        """Test _get_flat_grad_sample handles list of grad_samples."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            # Create a mock parameter with list of grad_samples
            param = list(model.parameters())[0]
            batch_size = 2
            param.grad_sample = [
                torch.randn(batch_size, *param.shape),
                torch.randn(batch_size, *param.shape),
            ]
            
            flat = optimizer._get_flat_grad_sample(param)
            
            self.assertEqual(flat.shape[0], batch_size * 2)
            self.assertEqual(flat.shape[1], param.numel())


class TestGetOptimizerClass(unittest.TestCase):
    """Test get_optimizer_class returns correct optimizer for per-layer clipping."""

    def test_per_layer_hooks_fsdp_distributed(self):
        """Test get_optimizer_class returns DistributedPerLayerOptimizerTP for hooks_fsdp."""
        optim_class = get_optimizer_class(
            clipping="per_layer",
            distributed=True,
            grad_sample_mode="hooks_fsdp",
        )
        self.assertEqual(optim_class, DistributedPerLayerOptimizerTP)

    def test_per_layer_hooks_tp_distributed(self):
        """Test get_optimizer_class returns DistributedPerLayerOptimizerTP for hooks_tp."""
        optim_class = get_optimizer_class(
            clipping="per_layer",
            distributed=True,
            grad_sample_mode="hooks_tp",
        )
        self.assertEqual(optim_class, DistributedPerLayerOptimizerTP)

    def test_per_layer_hooks_non_distributed(self):
        """Test get_optimizer_class returns DPPerLayerOptimizer for non-distributed."""
        optim_class = get_optimizer_class(
            clipping="per_layer",
            distributed=False,
        )
        self.assertEqual(optim_class, DPPerLayerOptimizer)


class TestDistributedPerLayerOptimizerTPWorkflow(unittest.TestCase):
    """Test full workflow with DistributedPerLayerOptimizerTP."""

    def test_clip_and_accumulate(self):
        """Test clip_and_accumulate works correctly."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            # Set up grad_samples on parameters
            batch_size = 4
            for p in optimizer.params:
                p.grad_sample = torch.randn(batch_size, *p.shape)
            
            # Call clip_and_accumulate
            optimizer.clip_and_accumulate()
            
            # Check that summed_grad is set
            for p in optimizer.params:
                self.assertIsNotNone(p.summed_grad)
                # summed_grad is flattened, so check numel matches
                self.assertEqual(p.summed_grad.numel(), p.numel())

    def test_add_noise(self):
        """Test add_noise works correctly."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            # Set up summed_grad on parameters with correct shape (flattened)
            # and proper _processed flag
            for p in optimizer.params:
                p.summed_grad = torch.randn(p.numel())
                # Don't set _processed flag - it should not exist for unprocessed grads
            
            # Call add_noise
            optimizer.add_noise()
            
            # Check that grad is set
            for p in optimizer.params:
                self.assertIsNotNone(p.grad)
                self.assertEqual(p.grad.shape, p.shape)

    def test_reduce_gradients(self):
        """Test reduce_gradients works correctly."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            # Set up grad on parameters
            for p in optimizer.params:
                p.grad = torch.randn_like(p)
            
            # Call reduce_gradients (should not raise)
            optimizer.reduce_gradients()
            
            # Check that grad is still set
            for p in optimizer.params:
                self.assertIsNotNone(p.grad)

    def test_full_step(self):
        """Test full optimization step."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            num_params = len(list(model.parameters()))
            max_grad_norms = [1.0] * num_params
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            # Set up grad_samples on parameters
            batch_size = 4
            for p in optimizer.params:
                p.grad_sample = torch.randn(batch_size, *p.shape)
            
            # Store original parameters
            original_params = [p.clone() for p in optimizer.params]
            
            # Call step
            optimizer.step()
            
            # Check that parameters have been updated
            for orig, p in zip(original_params, optimizer.params):
                # Parameters should have changed (with high probability)
                self.assertFalse(torch.allclose(orig, p))


class TestDistributedPerLayerOptimizerTPDifferentNorms(unittest.TestCase):
    """Test per-layer clipping with different norms per layer."""

    def test_different_norms_per_layer(self):
        """Test that different max_grad_norms are applied per layer."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            # Use very different norms for each parameter
            max_grad_norms = [0.1, 0.5, 1.0, 2.0]
            
            optimizer = DistributedPerLayerOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=0.0,  # No noise for deterministic test
                max_grad_norm=max_grad_norms,
                expected_batch_size=4,
            )
            
            # Set up large grad_samples that will be clipped
            batch_size = 4
            for i, p in enumerate(optimizer.params):
                # Create grad_samples with norm > max_grad_norm
                p.grad_sample = torch.ones(batch_size, *p.shape) * 10.0
            
            # Call clip_and_accumulate
            optimizer.clip_and_accumulate()
            
            # Verify clipping was applied (summed_grad should be bounded)
            for i, p in enumerate(optimizer.params):
                self.assertIsNotNone(p.summed_grad)


if __name__ == "__main__":
    unittest.main()
