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
Tests for DistributedAdaClipDPOptimizer.

This module tests the unified distributed adaptive clipping optimizer that
handles all distributed modes: DDP, FSDP, TP, CP.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn

from opacus.optimizers import (
    AdaClipDPOptimizer,
    DistributedAdaClipDPOptimizer,
    get_optimizer_class,
)


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

        if self._tmpfile is not None:
            try:
                os.remove(self._tmpfile.name)
            except OSError:
                pass


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class TestGetOptimizerClass(unittest.TestCase):
    """Tests for get_optimizer_class with adaptive clipping."""

    def test_adaptive_non_distributed_returns_adaclip(self):
        """Test that adaptive + non-distributed returns AdaClipDPOptimizer."""
        cls = get_optimizer_class(clipping="adaptive", distributed=False)
        self.assertEqual(cls, AdaClipDPOptimizer)

    def test_adaptive_distributed_returns_distributed_adaclip(self):
        """Test that adaptive + distributed returns DistributedAdaClipDPOptimizer."""
        cls = get_optimizer_class(clipping="adaptive", distributed=True)
        self.assertEqual(cls, DistributedAdaClipDPOptimizer)


class TestDistributedAdaClipDPOptimizerInit(unittest.TestCase):
    """Tests for DistributedAdaClipDPOptimizer initialization."""

    def test_inheritance(self):
        """Test that DistributedAdaClipDPOptimizer inherits from AdaClipDPOptimizer."""
        self.assertTrue(issubclass(DistributedAdaClipDPOptimizer, AdaClipDPOptimizer))

    def test_initialization_with_distributed(self):
        """Test initialization in a distributed environment."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            # Note: noise_multiplier must be < 2 * unclipped_num_std per Theorem 1
            # in https://arxiv.org/pdf/1905.03871.pdf
            dp_optimizer = DistributedAdaClipDPOptimizer(
                optimizer,
                noise_multiplier=0.5,
                target_unclipped_quantile=0.5,
                clipbound_learning_rate=0.1,
                max_clipbound=10.0,
                min_clipbound=0.1,
                unclipped_num_std=0.5,
                max_grad_norm=1.0,
                expected_batch_size=32,
            )

            self.assertEqual(dp_optimizer.rank, 0)
            self.assertEqual(dp_optimizer.world_size, 1)
            self.assertIsNone(dp_optimizer._has_sharded_params)
            self.assertEqual(dp_optimizer.target_unclipped_quantile, 0.5)
            self.assertEqual(dp_optimizer.clipbound_learning_rate, 0.1)
            self.assertEqual(dp_optimizer.max_clipbound, 10.0)
            self.assertEqual(dp_optimizer.min_clipbound, 0.1)


class TestDistributedAdaClipDPOptimizerMethods(unittest.TestCase):
    """Tests for DistributedAdaClipDPOptimizer methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.ctx = _SingleRankProcessGroup()
        self.ctx.__enter__()

        self.model = SimpleModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        # Note: noise_multiplier must be < 2 * unclipped_num_std per Theorem 1
        self.dp_optimizer = DistributedAdaClipDPOptimizer(
            self.optimizer,
            noise_multiplier=0.5,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.1,
            max_clipbound=10.0,
            min_clipbound=0.1,
            unclipped_num_std=0.5,
            max_grad_norm=1.0,
            expected_batch_size=32,
        )

    def tearDown(self):
        """Clean up."""
        self.ctx.__exit__(None, None, None)

    def test_detect_sharded_params_regular_tensors(self):
        """Test _detect_sharded_params returns False for regular tensors."""
        result = self.dp_optimizer._detect_sharded_params()
        self.assertFalse(result)
        # Should be cached
        self.assertFalse(self.dp_optimizer._has_sharded_params)

    def test_detect_sharded_params_caching(self):
        """Test that _detect_sharded_params caches the result."""
        # First call
        result1 = self.dp_optimizer._detect_sharded_params()
        # Manually set to True to verify caching
        self.dp_optimizer._has_sharded_params = True
        # Second call should return cached value
        result2 = self.dp_optimizer._detect_sharded_params()
        self.assertTrue(result2)

    def test_get_flat_grad_sample_tensor(self):
        """Test _get_flat_grad_sample with a tensor."""
        param = list(self.model.parameters())[0]
        batch_size = 4
        param.grad_sample = torch.randn(batch_size, *param.shape)

        result = self.dp_optimizer._get_flat_grad_sample(param)
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.dim(), 2)

    def test_get_flat_grad_sample_list(self):
        """Test _get_flat_grad_sample with a list of tensors."""
        param = list(self.model.parameters())[0]
        batch_size = 2
        param.grad_sample = [
            torch.randn(batch_size, *param.shape),
            torch.randn(batch_size, *param.shape),
        ]

        result = self.dp_optimizer._get_flat_grad_sample(param)
        self.assertEqual(result.shape[0], batch_size * 2)
        self.assertEqual(result.dim(), 2)

    def test_aggregate_clipping_stats(self):
        """Test _aggregate_clipping_stats aggregates across ranks."""
        self.dp_optimizer.sample_size = 10
        self.dp_optimizer.unclipped_num = 5

        self.dp_optimizer._aggregate_clipping_stats()

        # With world_size=1, values should remain the same
        self.assertEqual(self.dp_optimizer.sample_size, 10.0)
        self.assertEqual(self.dp_optimizer.unclipped_num, 5.0)

    def test_update_max_grad_norm_decreases_when_many_unclipped(self):
        """Test that max_grad_norm decreases when unclipped fraction > target.
        
        When too many samples are unclipped (above target), we should decrease
        the clipping bound to clip more aggressively.
        """
        initial_norm = self.dp_optimizer.max_grad_norm
        self.dp_optimizer.sample_size = 100
        # 80% unclipped, target is 50%
        self.dp_optimizer.unclipped_num = 80

        self.dp_optimizer.update_max_grad_norm()

        # Should decrease because unclipped_frac > target (need to clip more)
        self.assertLess(self.dp_optimizer.max_grad_norm, initial_norm)

    def test_update_max_grad_norm_increases_when_few_unclipped(self):
        """Test that max_grad_norm increases when unclipped fraction < target.
        
        When too few samples are unclipped (below target), we should increase
        the clipping bound to clip less aggressively.
        """
        initial_norm = self.dp_optimizer.max_grad_norm
        self.dp_optimizer.sample_size = 100
        # 20% unclipped, target is 50%
        self.dp_optimizer.unclipped_num = 20

        self.dp_optimizer.update_max_grad_norm()

        # Should increase because unclipped_frac < target (need to clip less)
        self.assertGreater(self.dp_optimizer.max_grad_norm, initial_norm)

    def test_update_max_grad_norm_respects_max_bound(self):
        """Test that max_grad_norm doesn't exceed max_clipbound."""
        self.dp_optimizer.max_grad_norm = 9.9
        self.dp_optimizer.sample_size = 100
        self.dp_optimizer.unclipped_num = 99  # Very high unclipped fraction

        self.dp_optimizer.update_max_grad_norm()

        self.assertLessEqual(self.dp_optimizer.max_grad_norm, self.dp_optimizer.max_clipbound)

    def test_update_max_grad_norm_respects_min_bound(self):
        """Test that max_grad_norm doesn't go below min_clipbound."""
        self.dp_optimizer.max_grad_norm = 0.2
        self.dp_optimizer.sample_size = 100
        self.dp_optimizer.unclipped_num = 1  # Very low unclipped fraction

        self.dp_optimizer.update_max_grad_norm()

        self.assertGreaterEqual(self.dp_optimizer.max_grad_norm, self.dp_optimizer.min_clipbound)


class TestDistributedAdaClipDPOptimizerIntegration(unittest.TestCase):
    """Integration tests for DistributedAdaClipDPOptimizer."""

    def test_full_training_step(self):
        """Test a full training step with the optimizer."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            dp_optimizer = DistributedAdaClipDPOptimizer(
                optimizer,
                noise_multiplier=0.1,
                target_unclipped_quantile=0.5,
                clipbound_learning_rate=0.1,
                max_clipbound=10.0,
                min_clipbound=0.1,
                unclipped_num_std=0.1,
                max_grad_norm=1.0,
                expected_batch_size=4,
            )

            # Simulate per-sample gradients
            batch_size = 4
            for param in model.parameters():
                param.grad_sample = torch.randn(batch_size, *param.shape)

            initial_norm = dp_optimizer.max_grad_norm

            # Perform optimization step
            dp_optimizer.step()

            # max_grad_norm should have been updated
            # (may increase or decrease depending on random grad_samples)
            self.assertIsNotNone(dp_optimizer.max_grad_norm)

            # Gradients should be set
            for param in model.parameters():
                self.assertIsNotNone(param.grad)

    def test_zero_grad_resets_stats(self):
        """Test that zero_grad resets sample_size and unclipped_num."""
        with _SingleRankProcessGroup():
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            # Note: noise_multiplier must be < 2 * unclipped_num_std per Theorem 1
            dp_optimizer = DistributedAdaClipDPOptimizer(
                optimizer,
                noise_multiplier=0.5,
                target_unclipped_quantile=0.5,
                clipbound_learning_rate=0.1,
                max_clipbound=10.0,
                min_clipbound=0.1,
                unclipped_num_std=0.5,
                max_grad_norm=1.0,
                expected_batch_size=32,
            )

            dp_optimizer.sample_size = 100
            dp_optimizer.unclipped_num = 50

            dp_optimizer.zero_grad()

            self.assertEqual(dp_optimizer.sample_size, 0)
            self.assertEqual(dp_optimizer.unclipped_num, 0)


class TestDistributedAdaClipDPOptimizerDTensorHandling(unittest.TestCase):
    """Tests for DTensor handling in DistributedAdaClipDPOptimizer."""

    def test_is_dtensor_regular_tensor(self):
        """Test that regular tensors are not detected as DTensors."""
        from opacus.optimizers.distributed_adaclipoptimizer import _is_dtensor

        tensor = torch.randn(4, 4)
        self.assertFalse(_is_dtensor(tensor))

    def test_get_local_tensor_regular_tensor(self):
        """Test that _get_local_tensor returns the same tensor for regular tensors."""
        from opacus.optimizers.distributed_adaclipoptimizer import _get_local_tensor

        tensor = torch.randn(4, 4)
        result = _get_local_tensor(tensor)
        self.assertIs(result, tensor)


if __name__ == "__main__":
    unittest.main()
