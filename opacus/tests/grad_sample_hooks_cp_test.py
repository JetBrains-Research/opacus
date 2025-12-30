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
Tests for GradSampleHooksCP and GradSampleModuleCP classes.

These tests verify the Context Parallelism (CP) support for per-sample gradient computation.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn

from opacus.grad_sample.grad_sample_module import GradSampleHooks, GradSampleModule
from opacus.grad_sample.grad_sample_module_cp import (
    GradSampleHooksCP,
    GradSampleModuleCP,
)
from opacus.grad_sample.utils import get_gsm_class, get_hooks_class, wrap_model


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TestGetGsmClassCP(unittest.TestCase):
    """Tests for get_gsm_class with CP modes."""

    def test_hooks_cp_mode(self):
        """Test that hooks_cp mode returns GradSampleModuleCP."""
        cls = get_gsm_class("hooks_cp")
        self.assertEqual(cls, GradSampleModuleCP)


class TestGetHooksClassCP(unittest.TestCase):
    """Tests for get_hooks_class with CP modes."""

    def test_hooks_cp_mode(self):
        """Test that hooks_cp mode returns GradSampleHooksCP."""
        cls = get_hooks_class("hooks_cp")
        self.assertEqual(cls, GradSampleHooksCP)


class TestWrapModelCP(unittest.TestCase):
    """Tests for wrap_model function with CP modes."""

    def test_wrap_model_hooks_cp_mode_wrapping(self):
        """Test wrap_model with hooks_cp mode and wrapping enabled."""
        model = SimpleModel()
        wrapped = wrap_model(model, grad_sample_mode="hooks_cp", wrap_model=True)
        self.assertIsInstance(wrapped, GradSampleModuleCP)

    def test_wrap_model_hooks_cp_mode_no_wrapping(self):
        """Test wrap_model with hooks_cp mode and wrapping disabled."""
        model = SimpleModel()
        hooks = wrap_model(model, grad_sample_mode="hooks_cp", wrap_model=False)
        self.assertIsInstance(hooks, GradSampleHooksCP)


class TestGradSampleHooksCP(unittest.TestCase):
    """Tests for GradSampleHooksCP class."""

    def test_initialization(self):
        """Test that GradSampleHooksCP initializes correctly."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        self.assertIsNotNone(hooks)
        self.assertEqual(hooks._module, model)

    def test_cp_group_none_by_default(self):
        """Test that cp_group is None by default."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        self.assertIsNone(hooks.cp_group)
        self.assertIsNone(hooks._cp_world_size)
        self.assertIsNone(hooks._cp_rank)

    def test_should_aggregate_returns_false_without_distributed(self):
        """Test that _should_aggregate_across_cp returns False when not distributed."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        # Without distributed initialization, should return False
        self.assertFalse(hooks._should_aggregate_across_cp())

    def test_hooks_attachment(self):
        """Test that hooks are properly attached to the model."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        self.assertTrue(hasattr(model, "autograd_grad_sample_hooks"))

    def test_cleanup(self):
        """Test that cleanup removes hooks."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        self.assertTrue(hasattr(model, "autograd_grad_sample_hooks"))

        hooks.cleanup()

        self.assertFalse(hasattr(model, "autograd_grad_sample_hooks"))

    def test_enable_disable_hooks(self):
        """Test enable/disable hooks functionality."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        self.assertTrue(hooks.hooks_enabled)

        hooks.disable_hooks()
        self.assertFalse(hooks.hooks_enabled)

        hooks.enable_hooks()
        self.assertTrue(hooks.hooks_enabled)

    def test_grad_sample_shape(self):
        """Test that grad_sample has correct shape [batch_size, *param_shape]."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        batch_size = 4
        x = torch.randn(batch_size, 4)
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check grad_sample shapes
        for name, param in model.named_parameters():
            if param.requires_grad and hasattr(param, "grad_sample"):
                gs = param.grad_sample
                if gs is not None:
                    # grad_sample should be [batch_size, *param_shape]
                    expected_shape = (batch_size,) + param.shape
                    self.assertEqual(gs.shape, expected_shape)


class TestGradSampleModuleCP(unittest.TestCase):
    """Tests for GradSampleModuleCP class."""

    def test_initialization(self):
        """Test that GradSampleModuleCP initializes correctly."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapped = GradSampleModuleCP(model)

        self.assertIsNotNone(wrapped)
        self.assertEqual(wrapped._module, model)

    def test_forward_pass(self):
        """Test forward pass through wrapped model."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapped = GradSampleModuleCP(model)

        x = torch.randn(4, 4)
        wrapped.train()
        output = wrapped(x)

        self.assertEqual(output.shape, (4, 2))

    def test_grad_sample_computation(self):
        """Test that grad_sample is computed correctly."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapped = GradSampleModuleCP(model)

        batch_size = 4
        x = torch.randn(batch_size, 4)
        wrapped.train()
        output = wrapped(x)
        loss = output.sum()
        loss.backward()

        # Check that grad_sample exists on parameters
        has_grad_sample = False
        for param in model.parameters():
            if param.requires_grad and hasattr(param, "grad_sample"):
                if param.grad_sample is not None:
                    has_grad_sample = True
                    break

        self.assertTrue(has_grad_sample)


class TestCPInheritance(unittest.TestCase):
    """Tests for CP class inheritance from FSDP."""

    def test_hooks_cp_inherits_from_fsdp(self):
        """Test that GradSampleHooksCP inherits from GradSampleHooksFSDP."""
        from opacus.grad_sample.grad_sample_module_fsdp import GradSampleHooksFSDP

        self.assertTrue(issubclass(GradSampleHooksCP, GradSampleHooksFSDP))

    def test_module_cp_inherits_from_hooks_cp(self):
        """Test that GradSampleModuleCP inherits from GradSampleHooksCP."""
        self.assertTrue(issubclass(GradSampleModuleCP, GradSampleHooksCP))


class TestCPGroupManagement(unittest.TestCase):
    """Tests for CP process group management."""

    def test_set_cp_group(self):
        """Test setting CP group after initialization."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        # Create a mock process group
        mock_group = MagicMock()

        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.get_rank", return_value=0):
                hooks.set_cp_group(mock_group)

        self.assertEqual(hooks.cp_group, mock_group)
        self.assertEqual(hooks._cp_world_size, 2)
        self.assertEqual(hooks._cp_rank, 0)


class TestCPNormAggregation(unittest.TestCase):
    """Tests for per-sample norm aggregation in CP."""

    def test_get_per_sample_norms_local(self):
        """Test get_per_sample_norms without distributed (local computation)."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        batch_size = 4
        x = torch.randn(batch_size, 4)
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Should compute local norms without error
        norms = hooks.get_per_sample_norms()
        self.assertEqual(norms.shape, (batch_size,))
        self.assertTrue(torch.all(norms >= 0))


class TestCPWithMockedDistributed(unittest.TestCase):
    """Tests for CP with mocked distributed environment."""

    def test_should_aggregate_with_cp_group(self):
        """Test _should_aggregate_across_cp with CP group set."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        # Set up mock CP group
        mock_group = MagicMock()
        hooks.cp_group = mock_group
        hooks._cp_world_size = 2
        hooks._cp_rank = 0

        with patch("torch.distributed.is_initialized", return_value=True):
            self.assertTrue(hooks._should_aggregate_across_cp())

    def test_should_not_aggregate_with_world_size_1(self):
        """Test _should_aggregate_across_cp returns False with world_size=1."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksCP(model)

        # Set up mock CP group with world_size=1
        mock_group = MagicMock()
        hooks.cp_group = mock_group
        hooks._cp_world_size = 1
        hooks._cp_rank = 0

        with patch("torch.distributed.is_initialized", return_value=True):
            self.assertFalse(hooks._should_aggregate_across_cp())


if __name__ == "__main__":
    unittest.main()
