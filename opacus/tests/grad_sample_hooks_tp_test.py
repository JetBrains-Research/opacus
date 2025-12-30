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
Tests for Tensor Parallelism (TP) support in Opacus.

Tests cover:
- GradSampleHooksTP and GradSampleModuleTP classes
- DistributedDPOptimizerTP class
- Mode registration in utils.py
- Integration with wrap_model function
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
from opacus.grad_sample.grad_sample_module_tp import (
    GradSampleHooksTP,
    GradSampleModuleTP,
)
from opacus.grad_sample.utils import get_gsm_class, get_hooks_class, wrap_model


# Check for TP availability
def _tp_available() -> bool:
    """Check if PyTorch Tensor Parallelism APIs are available."""
    try:
        from torch.distributed.tensor import DTensor  # noqa: F401
        from torch.distributed.tensor.parallel import (  # noqa: F401
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )

        return True
    except ImportError:
        return False


# Context manager for single-rank process group setup
class _SingleRankProcessGroup:
    """Context manager for setting up a single-rank distributed environment for testing."""

    def __init__(self):
        self._inited = False
        self._tmpfile = None
        self._prev_default_device = None

    def __enter__(self):
        if not dist.is_available():
            raise unittest.SkipTest("torch.distributed is not available")

        if dist.is_initialized():
            return self

        # Force default device to CPU to avoid accidental CPU/CUDA tensor mismatches.
        if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
            self._prev_default_device = torch.get_default_device()
            torch.set_default_device("cpu")

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

        if (
            self._prev_default_device is not None
            and hasattr(torch, "set_default_device")
        ):
            torch.set_default_device(self._prev_default_device)

        if self._tmpfile is not None:
            try:
                os.remove(self._tmpfile.name)
            except OSError:
                pass


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)  # No bias for TP compatibility
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class SimpleModelWithBias(nn.Module):
    """Simple model with bias for testing replicated parameter handling."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2, bias=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TestGetGsmClassTP(unittest.TestCase):
    """Tests for get_gsm_class with TP modes."""

    def test_hooks_tp_mode(self):
        """Test that hooks_tp mode returns GradSampleModuleTP."""
        cls = get_gsm_class("hooks_tp")
        self.assertEqual(cls, GradSampleModuleTP)

    def test_ghost_tp_mode(self):
        """Test that ghost_tp mode returns the correct class."""
        from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_tp import (
            GradSampleModuleFastGradientClippingTP,
        )

        cls = get_gsm_class("ghost_tp")
        self.assertEqual(cls, GradSampleModuleFastGradientClippingTP)


class TestGetHooksClassTP(unittest.TestCase):
    """Tests for get_hooks_class with TP modes."""

    def test_hooks_tp_mode(self):
        """Test that hooks_tp mode returns GradSampleHooksTP."""
        cls = get_hooks_class("hooks_tp")
        self.assertEqual(cls, GradSampleHooksTP)

    def test_ghost_tp_mode(self):
        """Test that ghost_tp mode returns the correct class."""
        from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_tp import (
            GradSampleHooksFastGradientClippingTP,
        )

        cls = get_hooks_class("ghost_tp")
        self.assertEqual(cls, GradSampleHooksFastGradientClippingTP)


class TestWrapModelTP(unittest.TestCase):
    """Tests for wrap_model function with TP modes."""

    def test_wrap_model_hooks_tp_mode_wrapping(self):
        """Test wrap_model with hooks_tp mode and wrapping enabled."""
        model = SimpleModel()
        wrapped = wrap_model(model, grad_sample_mode="hooks_tp", wrap_model=True)
        self.assertIsInstance(wrapped, GradSampleModuleTP)

    def test_wrap_model_hooks_tp_mode_no_wrapping(self):
        """Test wrap_model with hooks_tp mode and wrapping disabled."""
        model = SimpleModel()
        hooks = wrap_model(model, grad_sample_mode="hooks_tp", wrap_model=False)
        self.assertIsInstance(hooks, GradSampleHooksTP)


class TestGradSampleHooksTP(unittest.TestCase):
    """Tests for GradSampleHooksTP class."""

    def test_initialization(self):
        """Test that GradSampleHooksTP initializes correctly."""
        model = SimpleModel()
        # Suppress the beta warning during tests
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        self.assertIsNotNone(hooks)
        self.assertEqual(hooks._module, model)

    def test_is_dtensor_with_regular_tensor(self):
        """Test _is_dtensor returns False for regular tensors."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        regular_tensor = torch.randn(3, 4)
        self.assertFalse(hooks._is_dtensor(regular_tensor))

    def test_to_local_tensor_with_regular_tensor(self):
        """Test _to_local_tensor returns tensor as-is for regular tensors."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        regular_tensor = torch.randn(3, 4)
        result = hooks._to_local_tensor(regular_tensor)
        self.assertTrue(torch.equal(result, regular_tensor))

    def test_analyze_tp_placements_non_dtensor(self):
        """Test _analyze_tp_placements sets merge_flag=False for non-DTensor params."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        # All parameters should have _tp_merge_flag=False since they're not DTensors
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertTrue(hasattr(param, "_tp_merge_flag"))
                self.assertFalse(param._tp_merge_flag)

    def test_attribute_initialization_simple_model(self):
        """Test that hooks properly initialize attributes on parameters."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        # Run a forward pass to trigger hooks
        x = torch.randn(2, 4)
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that grad_sample attributes are set
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertTrue(hasattr(param, "grad_sample"))

    def test_cleanup_removes_hooks(self):
        """Test that cleanup properly removes hooks."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        self.assertTrue(hasattr(model, "autograd_grad_sample_hooks"))

        hooks.cleanup()

        self.assertFalse(hasattr(model, "autograd_grad_sample_hooks"))

    def test_enable_disable_hooks(self):
        """Test enable/disable hooks functionality."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

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
            hooks = GradSampleHooksTP(model)

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


class TestReplicatedParameterAutoFreeze(unittest.TestCase):
    """Tests for automatic freezing of replicated parameters."""

    def test_replicated_params_auto_frozen_with_warning(self):
        """Test that replicated parameters are auto-frozen with a warning."""
        import warnings

        # Create a mock DTensor parameter with Replicate placement
        model = SimpleModel()

        # Mock a parameter to appear as a replicated DTensor
        mock_placement = MagicMock()
        mock_placement.is_shard.return_value = False
        mock_placement.is_replicate.return_value = True

        # Patch the parameter to look like a DTensor
        original_fc1_weight = model.fc1.weight
        mock_dtensor = MagicMock(spec=torch.Tensor)
        mock_dtensor.requires_grad = True
        mock_dtensor.placements = [mock_placement]
        mock_dtensor.shape = original_fc1_weight.shape

        # We need to test the actual behavior, so let's use SimpleModelWithBias
        # and mock the DTensor check
        model_with_bias = SimpleModelWithBias()

        # Store original requires_grad state
        original_bias_requires_grad = {
            "fc1.bias": model_with_bias.fc1.bias.requires_grad,
            "fc2.bias": model_with_bias.fc2.bias.requires_grad,
        }
        self.assertTrue(all(original_bias_requires_grad.values()))

        # Mock _is_dtensor to return True for bias parameters
        # and mock placements to return Replicate for bias
        with patch.object(GradSampleHooksTP, "_is_dtensor") as mock_is_dtensor:

            def is_dtensor_side_effect(param):
                # Return True for bias parameters (small 1D tensors)
                return len(param.shape) == 1

            mock_is_dtensor.side_effect = is_dtensor_side_effect

            # Also need to add placements attribute to bias params
            mock_replicate = MagicMock()
            mock_replicate.is_shard.return_value = False
            mock_replicate.is_replicate.return_value = True

            mock_shard = MagicMock()
            mock_shard.is_shard.return_value = True
            mock_shard.is_replicate.return_value = False

            # Add placements to parameters
            model_with_bias.fc1.bias.placements = [mock_replicate]
            model_with_bias.fc2.bias.placements = [mock_replicate]
            model_with_bias.fc1.weight.placements = [mock_shard]
            model_with_bias.fc2.weight.placements = [mock_shard]

            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                hooks = GradSampleHooksTP(model_with_bias)

                # Check that a warning was issued about frozen parameters
                replicated_warnings = [
                    warning
                    for warning in w
                    if "Replicated parameters" in str(warning.message)
                ]
                self.assertEqual(len(replicated_warnings), 1)
                warning_msg = str(replicated_warnings[0].message)
                self.assertIn("automatically frozen", warning_msg)
                self.assertIn("fc1.bias", warning_msg)
                self.assertIn("fc2.bias", warning_msg)

            # Verify bias parameters are now frozen
            self.assertFalse(model_with_bias.fc1.bias.requires_grad)
            self.assertFalse(model_with_bias.fc2.bias.requires_grad)

            # Verify weight parameters are still trainable
            self.assertTrue(model_with_bias.fc1.weight.requires_grad)
            self.assertTrue(model_with_bias.fc2.weight.requires_grad)

            hooks.cleanup()

    def test_no_warning_when_no_replicated_params(self):
        """Test that no warning is issued when there are no replicated parameters."""
        import warnings

        model = SimpleModel()  # No bias, so no replicated params

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hooks = GradSampleHooksTP(model)

            # Filter for replicated parameter warnings
            replicated_warnings = [
                warning
                for warning in w
                if "Replicated parameters" in str(warning.message)
            ]
            self.assertEqual(len(replicated_warnings), 0)

            hooks.cleanup()


class TestGradSampleModuleTP(unittest.TestCase):
    """Tests for GradSampleModuleTP class."""

    def test_initialization(self):
        """Test that GradSampleModuleTP initializes correctly."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gsm = GradSampleModuleTP(model)

        self.assertIsNotNone(gsm)
        self.assertIsInstance(gsm, nn.Module)
        self.assertIsInstance(gsm, GradSampleHooksTP)

    def test_forward_pass(self):
        """Test forward pass through GradSampleModuleTP."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gsm = GradSampleModuleTP(model)

        x = torch.randn(2, 4)
        gsm.train()
        output = gsm(x)

        self.assertEqual(output.shape, (2, 2))

    def test_grad_sample_computation(self):
        """Test that grad_sample is computed correctly."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gsm = GradSampleModuleTP(model)

        batch_size = 3
        x = torch.randn(batch_size, 4)
        gsm.train()
        output = gsm(x)
        loss = output.sum()
        loss.backward()

        # Verify grad_sample exists and has correct batch dimension
        for name, param in gsm.named_parameters():
            if param.requires_grad:
                self.assertTrue(hasattr(param, "grad_sample"))
                if param.grad_sample is not None:
                    self.assertEqual(param.grad_sample.shape[0], batch_size)


class TestDistributedDPOptimizerTP(unittest.TestCase):
    """Tests for DistributedDPOptimizerTP class."""

    def test_import(self):
        """Test that DistributedDPOptimizerTP can be imported."""
        from opacus.optimizers.ddpoptimizer_tp import DistributedDPOptimizerTP

        self.assertIsNotNone(DistributedDPOptimizerTP)

    def test_optimizer_class_selection(self):
        """Test that get_optimizer_class returns DistributedDPOptimizerTP for hooks_tp mode."""
        from opacus.optimizers import get_optimizer_class
        from opacus.optimizers.ddpoptimizer_tp import DistributedDPOptimizerTP

        cls = get_optimizer_class(
            clipping="flat", distributed=True, grad_sample_mode="hooks_tp"
        )
        self.assertEqual(cls, DistributedDPOptimizerTP)

    def test_optimizer_class_selection_invalid(self):
        """Test that get_optimizer_class raises for invalid hooks_tp combinations."""
        from opacus.optimizers import get_optimizer_class

        # hooks_tp requires distributed=True
        with self.assertRaises(ValueError):
            get_optimizer_class(
                clipping="flat", distributed=False, grad_sample_mode="hooks_tp"
            )

    @unittest.skipUnless(dist.is_available(), "torch.distributed not available")
    def test_optimizer_initialization(self):
        """Test DistributedDPOptimizerTP initialization with distributed environment."""
        from opacus.optimizers.ddpoptimizer_tp import DistributedDPOptimizerTP

        with _SingleRankProcessGroup():
            model = SimpleModel()
            base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            dp_optimizer = DistributedDPOptimizerTP(
                optimizer=base_optimizer,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                expected_batch_size=32,
            )

            self.assertIsNotNone(dp_optimizer)
            self.assertEqual(dp_optimizer.noise_multiplier, 1.0)
            self.assertEqual(dp_optimizer.max_grad_norm, 1.0)


class TestTPHelperFunctions(unittest.TestCase):
    """Tests for TP helper functions in ddpoptimizer_tp.py."""

    def test_is_dtensor_function(self):
        """Test _is_dtensor helper function."""
        from opacus.optimizers.ddpoptimizer_tp import _is_dtensor

        regular_tensor = torch.randn(3, 4)
        self.assertFalse(_is_dtensor(regular_tensor))

    def test_get_local_tensor_function(self):
        """Test _get_local_tensor helper function."""
        from opacus.optimizers.ddpoptimizer_tp import _get_local_tensor

        regular_tensor = torch.randn(3, 4)
        result = _get_local_tensor(regular_tensor)
        self.assertTrue(torch.equal(result, regular_tensor))


@unittest.skipUnless(_tp_available(), "PyTorch TP APIs not available")
class TestTPWithActualDTensor(unittest.TestCase):
    """Tests that require actual DTensor support."""

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Requires 2+ GPUs",
    )
    def test_dtensor_detection(self):
        """Test that DTensor parameters are correctly detected."""
        # This test requires actual multi-GPU setup
        # Skip for now as it needs special environment
        pass


class TestTPIntegrationNonDistributed(unittest.TestCase):
    """Integration tests for TP that don't require distributed environment."""

    def test_full_workflow_non_dtensor(self):
        """Test full workflow with non-DTensor model (simulates single-rank TP)."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gsm = GradSampleModuleTP(model)

        batch_size = 4
        x = torch.randn(batch_size, 4)

        gsm.train()
        output = gsm(x)
        loss = output.sum()
        loss.backward()

        # Verify grad_samples are computed
        grad_samples_found = False
        for name, param in gsm.named_parameters():
            if param.requires_grad and hasattr(param, "grad_sample"):
                if param.grad_sample is not None:
                    grad_samples_found = True
                    # Verify shape
                    self.assertEqual(param.grad_sample.shape[0], batch_size)

        self.assertTrue(grad_samples_found, "No grad_samples were computed")

    def test_multiple_forward_backward_passes(self):
        """Test multiple forward/backward passes."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gsm = GradSampleModuleTP(model)

        gsm.train()

        for i in range(3):
            # Clear previous grad_samples
            for param in gsm.parameters():
                if hasattr(param, "grad_sample"):
                    param.grad_sample = None

            batch_size = 2 + i
            x = torch.randn(batch_size, 4)
            output = gsm(x)
            loss = output.sum()
            loss.backward()

            # Verify grad_samples have correct batch size
            for name, param in gsm.named_parameters():
                if param.requires_grad and hasattr(param, "grad_sample"):
                    if param.grad_sample is not None:
                        self.assertEqual(param.grad_sample.shape[0], batch_size)


@unittest.skipUnless(dist.is_available(), "torch.distributed not available")
class TestTPWithDistributed(unittest.TestCase):
    """Tests that require distributed environment but not actual TP."""

    def test_compute_local_norms(self):
        """Test _compute_local_norms fallback method."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        batch_size = 4
        x = torch.randn(batch_size, 4)
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Test local norm computation (non-distributed fallback)
        norms = hooks._compute_local_norms()
        self.assertEqual(norms.shape, (batch_size,))
        self.assertTrue(torch.all(norms >= 0))

    def test_get_per_sample_norms_non_distributed(self):
        """Test get_per_sample_norms when distributed is not initialized."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        batch_size = 4
        x = torch.randn(batch_size, 4)
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()

        # When distributed is not initialized, should fall back to local computation
        norms = hooks.get_per_sample_norms()
        self.assertEqual(norms.shape, (batch_size,))


class TestTPInheritsFromFSDP(unittest.TestCase):
    """Tests verifying that GradSampleHooksTP properly inherits from GradSampleHooksFSDP."""

    def test_inheritance(self):
        """Test that GradSampleHooksTP inherits from GradSampleHooksFSDP."""
        from opacus.grad_sample.grad_sample_module_fsdp import GradSampleHooksFSDP

        self.assertTrue(issubclass(GradSampleHooksTP, GradSampleHooksFSDP))

    def test_module_tp_inheritance(self):
        """Test that GradSampleModuleTP inherits from GradSampleHooksTP."""
        self.assertTrue(issubclass(GradSampleModuleTP, GradSampleHooksTP))

    def test_has_fsdp_get_module_type(self):
        """Test that GradSampleHooksTP has the FSDP _get_module_type method."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        # Should have _get_module_type from FSDP parent
        self.assertTrue(hasattr(hooks, "_get_module_type"))
        
        # Test that it works on a regular module
        module_type = hooks._get_module_type(model.fc1)
        self.assertEqual(module_type, nn.Linear)

    def test_fsdp_attribute_initialization(self):
        """Test that FSDP attribute initialization works in TP hooks."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks = GradSampleHooksTP(model)

        # Run forward/backward to trigger hooks
        batch_size = 2
        x = torch.randn(batch_size, 4)
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that parameters have both FSDP and TP attributes
        for name, param in model.named_parameters():
            if param.requires_grad:
                # FSDP attributes
                self.assertTrue(hasattr(param, "grad_sample"))
                self.assertTrue(hasattr(param, "_forward_counter"))
                # TP attributes
                self.assertTrue(hasattr(param, "_tp_merge_flag"))


class TestCombinedFSDPTPMode(unittest.TestCase):
    """Tests for combined FSDP + TP (2D parallelism) support."""

    def test_hooks_tp_supports_fsdp_features(self):
        """Test that hooks_tp mode supports FSDP-specific features."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Create TP hooks (which now inherit from FSDP hooks)
            hooks = GradSampleHooksTP(model)

        # Verify it has FSDP capabilities
        from opacus.grad_sample.grad_sample_module_fsdp import GradSampleHooksFSDP
        self.assertIsInstance(hooks, GradSampleHooksFSDP)

        # Verify it has TP capabilities
        self.assertTrue(hasattr(hooks, "_analyze_tp_placements"))
        self.assertTrue(hasattr(hooks, "_is_dtensor"))
        self.assertTrue(hasattr(hooks, "_to_local_tensor"))
        self.assertTrue(hasattr(hooks, "get_per_sample_norms"))

    def test_grad_sample_mode_registration(self):
        """Test that hooks_tp is properly registered and returns correct class."""
        from opacus.grad_sample.utils import get_gsm_class, get_hooks_class
        from opacus.grad_sample.grad_sample_module_fsdp import GradSampleHooksFSDP

        # Get the TP classes
        gsm_class = get_gsm_class("hooks_tp")
        hooks_class = get_hooks_class("hooks_tp")

        # Verify they inherit from FSDP classes
        self.assertTrue(issubclass(hooks_class, GradSampleHooksFSDP))

    def test_workflow_with_non_dtensor_model(self):
        """Test full workflow with non-DTensor model (simulates FSDP without TP sharding)."""
        model = SimpleModel()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gsm = GradSampleModuleTP(model)

        batch_size = 4
        x = torch.randn(batch_size, 4)

        gsm.train()
        output = gsm(x)
        loss = output.sum()
        loss.backward()

        # Verify grad_samples are computed correctly
        for name, param in gsm.named_parameters():
            if param.requires_grad and hasattr(param, "grad_sample"):
                if param.grad_sample is not None:
                    # Shape should be [batch_size, *param_shape]
                    expected_shape = (batch_size,) + param.shape
                    self.assertEqual(param.grad_sample.shape, expected_shape)
                    
                    # For non-DTensor params, _tp_merge_flag should be False
                    self.assertFalse(param._tp_merge_flag)


if __name__ == "__main__":
    unittest.main()
