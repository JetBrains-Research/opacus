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
Comprehensive tests for FSDP support in Opacus.

This module consolidates all FSDP-related tests including:
1. GradSampleHooksFSDP class methods and attribute initialization
2. GradSampleModuleFSDP wrapping and per-sample gradient computation
3. promote_current_grad_sample defensive hasattr check
4. wrap_model with hooks_fsdp mode
5. get_gsm_class and get_hooks_class functions
6. FSDP2Wrapper utility class
7. Integration tests with actual FSDP2 wrapping
8. Standard Opacus usage without HuggingFace Trainer
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn

from opacus.grad_sample.grad_sample_module import (
    GradSampleHooks,
    GradSampleModule,
    promote_current_grad_sample,
)
from opacus.grad_sample.grad_sample_module_fsdp import (
    GradSampleHooksFSDP,
    GradSampleModuleFSDP,
)
from opacus.grad_sample.utils import get_gsm_class, get_hooks_class, wrap_model


# Check for FSDP availability
try:
    from torch.distributed.fsdp import MixedPrecisionPolicy

    _HAS_FSDP = True
except Exception:
    MixedPrecisionPolicy = None
    _HAS_FSDP = False


# Check for composable FSDP2 availability
def _composable_fsdp_available() -> bool:
    try:
        from torch.distributed._composable.fsdp import fully_shard  # noqa: F401

        return True
    except Exception:
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

        # Composable FSDP2 may create a DeviceMesh and heuristically choose a CUDA
        # device even for single-rank CPU tests; force default device to CPU to
        # avoid accidental CPU/CUDA tensor mismatches.
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
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NestedModel(nn.Module):
    """Nested model for testing deeper hierarchies."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TestPromoteCurrentGradSample(unittest.TestCase):
    """Tests for promote_current_grad_sample defensive hasattr check."""

    def test_promote_with_current_grad_sample(self):
        """Test normal case where _current_grad_sample exists."""
        param = nn.Parameter(torch.randn(3, 4))
        param.requires_grad = True
        param.grad_sample = None
        param._current_grad_sample = torch.randn(2, 3, 4)

        promote_current_grad_sample(param)

        self.assertIsNotNone(param.grad_sample)
        self.assertFalse(hasattr(param, "_current_grad_sample"))

    def test_promote_without_current_grad_sample(self):
        """Test defensive case where _current_grad_sample doesn't exist."""
        param = nn.Parameter(torch.randn(3, 4))
        param.requires_grad = True
        param.grad_sample = None
        # Intentionally NOT setting _current_grad_sample

        # Should not raise an error
        promote_current_grad_sample(param)

        # grad_sample should remain None
        self.assertIsNone(param.grad_sample)

    def test_promote_with_existing_grad_sample_list(self):
        """Test accumulation case where grad_sample is already a list."""
        param = nn.Parameter(torch.randn(3, 4))
        param.requires_grad = True
        existing_gs = torch.randn(2, 3, 4)
        param.grad_sample = [existing_gs]
        param._current_grad_sample = torch.randn(2, 3, 4)

        promote_current_grad_sample(param)

        self.assertIsInstance(param.grad_sample, list)
        self.assertEqual(len(param.grad_sample), 2)

    def test_promote_with_existing_grad_sample_tensor(self):
        """Test case where grad_sample is a tensor (converts to list)."""
        param = nn.Parameter(torch.randn(3, 4))
        param.requires_grad = True
        existing_gs = torch.randn(2, 3, 4)
        param.grad_sample = existing_gs
        param._current_grad_sample = torch.randn(2, 3, 4)

        promote_current_grad_sample(param)

        self.assertIsInstance(param.grad_sample, list)
        self.assertEqual(len(param.grad_sample), 2)

    def test_promote_non_trainable_param(self):
        """Test that non-trainable params are skipped."""
        param = nn.Parameter(torch.randn(3, 4))
        param.requires_grad = False
        param._current_grad_sample = torch.randn(2, 3, 4)

        promote_current_grad_sample(param)

        # Should still have _current_grad_sample since requires_grad=False
        self.assertTrue(hasattr(param, "_current_grad_sample"))


class TestGetGsmClass(unittest.TestCase):
    """Tests for get_gsm_class function."""

    def test_hooks_mode(self):
        """Test that 'hooks' mode returns GradSampleModule."""
        cls = get_gsm_class("hooks")
        self.assertIs(cls, GradSampleModule)

    def test_functorch_mode(self):
        """Test that 'functorch' mode returns GradSampleModule."""
        cls = get_gsm_class("functorch")
        self.assertIs(cls, GradSampleModule)

    def test_hooks_fsdp_mode(self):
        """Test that 'hooks_fsdp' mode returns GradSampleModuleFSDP."""
        cls = get_gsm_class("hooks_fsdp")
        self.assertIs(cls, GradSampleModuleFSDP)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_gsm_class("invalid_mode")
        self.assertIn("Unexpected grad_sample_mode", str(ctx.exception))


class TestGetHooksClass(unittest.TestCase):
    """Tests for get_hooks_class function."""

    def test_hooks_mode(self):
        """Test that 'hooks' mode returns GradSampleHooks."""
        cls = get_hooks_class("hooks")
        self.assertIs(cls, GradSampleHooks)

    def test_functorch_mode(self):
        """Test that 'functorch' mode returns GradSampleHooks."""
        cls = get_hooks_class("functorch")
        self.assertIs(cls, GradSampleHooks)

    def test_hooks_fsdp_mode(self):
        """Test that 'hooks_fsdp' mode returns GradSampleHooksFSDP."""
        cls = get_hooks_class("hooks_fsdp")
        self.assertIs(cls, GradSampleHooksFSDP)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_hooks_class("invalid_mode")
        self.assertIn("Unexpected grad_sample_mode", str(ctx.exception))


class TestWrapModel(unittest.TestCase):
    """Tests for wrap_model function."""

    def test_wrap_model_hooks_mode_wrapping(self):
        """Test wrap_model with hooks mode and wrapping=True."""
        model = SimpleModel()
        wrapped = wrap_model(model, "hooks", wrap_model=True)
        self.assertIsInstance(wrapped, GradSampleModule)

    def test_wrap_model_hooks_mode_no_wrapping(self):
        """Test wrap_model with hooks mode and wrapping=False."""
        model = SimpleModel()
        hooks = wrap_model(model, "hooks", wrap_model=False)
        self.assertIsInstance(hooks, GradSampleHooks)

    def test_wrap_model_hooks_fsdp_mode_wrapping(self):
        """Test wrap_model with hooks_fsdp mode and wrapping=True."""
        model = SimpleModel()
        wrapped = wrap_model(model, "hooks_fsdp", wrap_model=True, strict=False)
        self.assertIsInstance(wrapped, GradSampleModuleFSDP)

    def test_wrap_model_hooks_fsdp_mode_no_wrapping(self):
        """Test wrap_model with hooks_fsdp mode and wrapping=False."""
        model = SimpleModel()
        hooks = wrap_model(model, "hooks_fsdp", wrap_model=False, strict=False)
        self.assertIsInstance(hooks, GradSampleHooksFSDP)

    def test_wrap_model_functorch_sets_flag(self):
        """Test that functorch mode sets force_functorch flag."""
        model = SimpleModel()
        wrapped = wrap_model(model, "functorch", wrap_model=True)
        self.assertTrue(wrapped.force_functorch)


class TestGradSampleHooksFSDP(unittest.TestCase):
    """Tests for GradSampleHooksFSDP class."""

    def test_get_module_type_regular_module(self):
        """Test _get_module_type with regular nn.Module."""
        model = nn.Linear(4, 3)
        hooks = GradSampleHooksFSDP.__new__(GradSampleHooksFSDP)
        self.assertIs(hooks._get_module_type(model), nn.Linear)

    def test_get_module_type_sequential(self):
        """Test _get_module_type with nn.Sequential."""
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
        hooks = GradSampleHooksFSDP.__new__(GradSampleHooksFSDP)
        self.assertIs(hooks._get_module_type(model), nn.Sequential)

    def test_attribute_initialization_simple_model(self):
        """Test that attributes are initialized on parameters."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        try:
            hooks.enable_hooks()
            batch_size = 4
            x = torch.randn(batch_size, 4)
            model.train()
            out = model(x)
            loss = out.sum()
            loss.backward()

            # Check that trainable parameters have grad_sample
            for p in model.parameters():
                if p.requires_grad:
                    self.assertTrue(hasattr(p, "grad_sample"))
        finally:
            hooks.cleanup()

    def test_attribute_initialization_nested_model(self):
        """Test attribute initialization with nested model."""
        model = NestedModel()
        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        try:
            hooks.enable_hooks()
            batch_size = 4
            x = torch.randn(batch_size, 4)
            model.train()
            out = model(x)
            loss = out.sum()
            loss.backward()

            # Check that trainable parameters have grad_sample
            trainable_count = 0
            for p in model.parameters():
                if p.requires_grad:
                    trainable_count += 1
                    self.assertTrue(hasattr(p, "grad_sample"))

            self.assertGreater(trainable_count, 0)
        finally:
            hooks.cleanup()

    def test_multiple_forward_passes(self):
        """Test that multiple forward passes work correctly."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        try:
            hooks.enable_hooks()
            model.train()

            # First forward/backward
            x1 = torch.randn(4, 4)
            out1 = model(x1)
            loss1 = out1.sum()
            loss1.backward()

            # Clear grad_sample for next iteration
            for p in model.parameters():
                if hasattr(p, "grad_sample"):
                    p.grad_sample = None

            # Second forward/backward
            x2 = torch.randn(4, 4)
            out2 = model(x2)
            loss2 = out2.sum()
            loss2.backward()

            # Check that grad_sample is populated after second pass
            for p in model.parameters():
                if p.requires_grad:
                    self.assertTrue(hasattr(p, "grad_sample"))
        finally:
            hooks.cleanup()

    def test_cleanup_removes_hooks(self):
        """Test that cleanup properly removes hooks."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        # Verify hooks are attached (stored in autograd_grad_sample_hooks)
        self.assertTrue(hasattr(hooks, "autograd_grad_sample_hooks"))
        self.assertTrue(len(hooks.autograd_grad_sample_hooks) > 0)

        # Cleanup
        hooks.cleanup()

        # Verify hooks are removed (attribute is deleted)
        self.assertFalse(hasattr(hooks, "autograd_grad_sample_hooks"))

    def test_enable_disable_hooks(self):
        """Test enable/disable hooks functionality."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        try:
            # Hooks are enabled by default after initialization (add_hooks calls enable_hooks)
            self.assertTrue(hooks.hooks_enabled)

            # Disable
            hooks.disable_hooks()
            self.assertFalse(hooks.hooks_enabled)

            # Re-enable
            hooks.enable_hooks()
            self.assertTrue(hooks.hooks_enabled)
        finally:
            hooks.cleanup()

    def test_grad_sample_shape(self):
        """Test that grad_sample has correct shape."""
        model = nn.Linear(4, 3)
        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        try:
            hooks.enable_hooks()
            batch_size = 8
            x = torch.randn(batch_size, 4)
            model.train()
            out = model(x)
            loss = out.sum()
            loss.backward()

            # Check weight grad_sample shape
            gs = model.weight.grad_sample
            if isinstance(gs, list):
                gs = gs[0]
            self.assertEqual(gs.shape[0], batch_size)
            self.assertEqual(gs.shape[1:], model.weight.shape)

            # Check bias grad_sample shape
            gs = model.bias.grad_sample
            if isinstance(gs, list):
                gs = gs[0]
            self.assertEqual(gs.shape[0], batch_size)
            self.assertEqual(gs.shape[1:], model.bias.shape)
        finally:
            hooks.cleanup()


class TestGradSampleModuleFSDP(unittest.TestCase):
    """Tests for GradSampleModuleFSDP class."""

    def test_initialization(self):
        """Test GradSampleModuleFSDP initialization."""
        model = SimpleModel()
        gsm = GradSampleModuleFSDP(model, strict=False)

        self.assertIsInstance(gsm, GradSampleModuleFSDP)
        self.assertIsInstance(gsm, GradSampleHooksFSDP)

    def test_forward_pass(self):
        """Test forward pass through GradSampleModuleFSDP."""
        model = SimpleModel()
        gsm = GradSampleModuleFSDP(model, strict=False)

        gsm.enable_hooks()
        batch_size = 4
        x = torch.randn(batch_size, 4)
        gsm.train()
        out = gsm(x)

        self.assertEqual(out.shape, (batch_size, 2))

    def test_grad_sample_computation(self):
        """Test grad_sample computation through GradSampleModuleFSDP."""
        model = SimpleModel()
        gsm = GradSampleModuleFSDP(model, strict=False)

        gsm.enable_hooks()
        batch_size = 4
        x = torch.randn(batch_size, 4)
        gsm.train()
        out = gsm(x)
        loss = out.sum()
        loss.backward()

        # Check that trainable parameters have grad_sample
        for p in gsm.parameters():
            if p.requires_grad:
                self.assertTrue(hasattr(p, "grad_sample"))
                self.assertIsNotNone(p.grad_sample)


class TestGradSampleHooksFSDPWithPartialTrainable(unittest.TestCase):
    """Tests for GradSampleHooksFSDP with partially trainable models (like LoRA)."""

    def test_partial_trainable_model(self):
        """Test with model where only some parameters are trainable."""
        model = SimpleModel()
        # Freeze fc1, keep fc2 trainable (simulating LoRA-like scenario)
        model.fc1.weight.requires_grad = False
        model.fc1.bias.requires_grad = False

        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        try:
            hooks.enable_hooks()
            batch_size = 4
            x = torch.randn(batch_size, 4)
            model.train()
            out = model(x)
            loss = out.sum()
            loss.backward()

            # Check that only trainable parameters have grad_sample
            self.assertFalse(
                hasattr(model.fc1.weight, "grad_sample")
                and model.fc1.weight.grad_sample is not None
            )
            self.assertTrue(hasattr(model.fc2.weight, "grad_sample"))
            self.assertIsNotNone(model.fc2.weight.grad_sample)
        finally:
            hooks.cleanup()

    def test_bias_only_trainable(self):
        """Test with model where only biases are trainable."""
        model = nn.Linear(4, 3)
        model.weight.requires_grad = False
        model.bias.requires_grad = True

        hooks = GradSampleHooksFSDP(model, batch_first=True, loss_reduction="mean")

        try:
            hooks.enable_hooks()
            batch_size = 4
            x = torch.randn(batch_size, 4)
            model.train()
            out = model(x)
            loss = out.sum()
            loss.backward()

            # Check that only bias has grad_sample
            self.assertTrue(hasattr(model.bias, "grad_sample"))
            self.assertIsNotNone(model.bias.grad_sample)
        finally:
            hooks.cleanup()


# =============================================================================
# FSDP2Wrapper Tests (from fsdp_utils_test.py)
# =============================================================================


@unittest.skipUnless(_HAS_FSDP, "torch.distributed.fsdp is not available")
class TestFSDP2Wrapper(unittest.TestCase):
    """Tests for FSDP2Wrapper utility class."""

    def test_fsdp2wrapper_sets_root_ignored_params_when_requested(self):
        """Test that FSDP2Wrapper sets ignored_params for trainable params when requested."""
        from opacus.utils.fsdp_utils import FSDP2Wrapper

        model = SimpleModel()
        calls = []

        def _fake_fully_shard(module, **kwargs):
            calls.append((module, kwargs))
            return module

        with patch("opacus.utils.fsdp_utils.fully_shard", new=_fake_fully_shard):
            FSDP2Wrapper(model, ignore_trainable_params=True)

        self.assertGreaterEqual(
            len(calls),
            1,
            "Expected FSDP2Wrapper to call fully_shard at least for the root model",
        )

        root_module, root_kwargs = calls[-1]
        self.assertIs(root_module, model)
        self.assertIn("ignored_params", root_kwargs)
        self.assertEqual(
            set(root_kwargs["ignored_params"]),
            {p for p in model.parameters() if p.requires_grad},
        )

    def test_fsdp2wrapper_does_not_set_ignored_params_by_default(self):
        """Test that FSDP2Wrapper does not set ignored_params by default."""
        from opacus.utils.fsdp_utils import FSDP2Wrapper

        model = SimpleModel()
        calls = []

        def _fake_fully_shard(module, **kwargs):
            calls.append((module, kwargs))
            return module

        with patch("opacus.utils.fsdp_utils.fully_shard", new=_fake_fully_shard):
            FSDP2Wrapper(model)

        self.assertGreaterEqual(len(calls), 1)
        for _module, kwargs in calls:
            self.assertIn("ignored_params", kwargs)
            self.assertIsNone(kwargs["ignored_params"])

    def test_fsdp2wrapper_overrides_mp_policy_for_high_precision_layers(self):
        """Test that FSDP2Wrapper overrides mp_policy for high precision layers."""
        from opacus.utils.fsdp_utils import FSDP2Wrapper

        model = SimpleModel()
        calls = []
        sentinel_mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

        def _fake_fully_shard(module, **kwargs):
            calls.append((module, kwargs))
            return module

        with patch("opacus.utils.fsdp_utils.fully_shard", new=_fake_fully_shard):
            FSDP2Wrapper(
                model,
                mp_policy=sentinel_mp,
                opacus_high_precision_layers=(nn.Linear,),
            )

        found_override = False
        for module, kwargs in calls[:-1]:
            if isinstance(module, nn.Linear):
                mp_policy = kwargs.get("mp_policy")
                self.assertIsInstance(mp_policy, MixedPrecisionPolicy)
                if mp_policy is not sentinel_mp:
                    self.assertEqual(mp_policy.param_dtype, torch.get_default_dtype())
                    found_override = True

        self.assertTrue(found_override)


# =============================================================================
# FSDP2 Integration Tests (from fsdp2_grad_sample_hooks_test.py)
# =============================================================================


def _cuda_causes_device_mismatch() -> bool:
    """Check if CUDA availability will cause device mismatch in FSDP2Wrapper tests.
    
    When CUDA is available, FSDP2's DeviceMesh heuristically selects CUDA devices,
    but the test's _SingleRankProcessGroup sets default device to CPU. This causes
    a device mismatch where some tensors end up on CUDA and others on CPU.
    """
    return torch.cuda.is_available()


@unittest.skipUnless(_composable_fsdp_available(), "Composable FSDP2 is not available")
class TestFSDP2Integration(unittest.TestCase):
    """Integration tests for GradSampleHooksFSDP with actual FSDP2 wrapping."""

    def test_get_module_type_returns_original_class(self) -> None:
        """Test that _get_module_type returns the original class for FSDP-wrapped modules."""
        from torch.distributed._composable.fsdp import fully_shard

        with _SingleRankProcessGroup():
            m = nn.Linear(4, 3)
            fully_shard(m)

            hooks = GradSampleHooksFSDP.__new__(GradSampleHooksFSDP)
            self.assertIs(hooks._get_module_type(m), nn.Linear)

    @unittest.skipIf(_cuda_causes_device_mismatch(), "FSDP2Wrapper causes device mismatch when CUDA is available")
    def test_per_sample_grads_work_with_fsdp2_non_wrapping(self) -> None:
        """Test per-sample gradients with FSDP2Wrapper and sparse fine-tuning."""
        from opacus.utils.fsdp_utils import FSDP2Wrapper

        batch_size = 5

        with _SingleRankProcessGroup():
            model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
            # Simulate sparse fine-tuning (e.g. LoRA-only): keep only biases trainable.
            model[0].weight.requires_grad = False
            model[2].weight.requires_grad = False
            model[0].bias.requires_grad = True
            model[2].bias.requires_grad = True

            # Force model to CPU for testing
            model = model.to("cpu")
            model = FSDP2Wrapper(model, ignore_trainable_params=True)

            gsm = GradSampleModuleFSDP(model, strict=False)
            gsm.enable_hooks()

            # Ensure input is on CPU
            x = torch.randn(batch_size, 4, device="cpu")
            y = gsm(x)
            loss = y.sum()
            loss.backward()

            trainable_params = [p for p in gsm.parameters() if p.requires_grad]
            self.assertTrue(trainable_params, "Expected at least one trainable parameter")

            for p in trainable_params:
                self.assertTrue(hasattr(p, "grad_sample"))
                self.assertIsNotNone(p.grad_sample)

                gs = p.grad_sample[0] if isinstance(p.grad_sample, list) else p.grad_sample
                self.assertEqual(gs.shape[0], batch_size)

    @unittest.skipIf(_cuda_causes_device_mismatch(), "FSDP2Wrapper causes device mismatch when CUDA is available")
    def test_fsdp2_wrapper_ignore_trainable_params(self) -> None:
        """Test FSDP2Wrapper with ignore_trainable_params for bias-only training."""
        from opacus.utils.fsdp_utils import FSDP2Wrapper

        batch_size = 4

        with _SingleRankProcessGroup():
            model = nn.Linear(4, 3)
            # Simulate sparse fine-tuning: freeze weights, train only bias.
            model.weight.requires_grad = False
            model.bias.requires_grad = True

            # Force model to CPU for testing
            model = model.to("cpu")
            model = FSDP2Wrapper(model, ignore_trainable_params=True)

            gsm = GradSampleModuleFSDP(model, strict=False)
            gsm.enable_hooks()

            # Ensure input is on CPU
            x = torch.randn(batch_size, 4, device="cpu")
            y = gsm(x)
            y.sum().backward()

            self.assertTrue(hasattr(model.bias, "grad_sample"))
            self.assertIsNotNone(model.bias.grad_sample)
            gs = (
                model.bias.grad_sample[0]
                if isinstance(model.bias.grad_sample, list)
                else model.bias.grad_sample
            )
            self.assertEqual(gs.shape[0], batch_size)


# =============================================================================
# Actual FSDP2 Tests with fully_shard (from grad_sample_hooks_fsdp_test.py)
# =============================================================================


@unittest.skipUnless(_HAS_FSDP, "torch.distributed.fsdp is not available")
class TestActualFSDP2(unittest.TestCase):
    """Tests using actual FSDP2 fully_shard."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        if not dist.is_available():
            raise unittest.SkipTest("torch.distributed is not available")

        cls._destroy_pg_on_exit = False
        if not dist.is_initialized():
            init_file = tempfile.NamedTemporaryFile(delete=False)
            init_file.close()
            dist.init_process_group(
                backend="gloo",
                rank=0,
                world_size=1,
                init_method=f"file://{init_file.name}",
            )
            cls._destroy_pg_on_exit = True

    @classmethod
    def tearDownClass(cls):
        if cls._destroy_pg_on_exit and dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    @unittest.skipUnless(_composable_fsdp_available(), "Composable FSDP2 is not available")
    def test_linear_fsdp2_grad_sample_is_populated(self):
        """Test that grad_sample is populated for FSDP2-wrapped linear layer."""
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.tensor import DeviceMesh

        # Wrap a leaf module so the backward hook runs on an `FSDPModule` subclass.
        # To mirror the current supported workaround (LoRA-only), keep the small
        # trainable subset unsharded/stable via `ignored_params`.
        layer = nn.Linear(4, 3)
        layer.weight.requires_grad = False
        self.assertIsNotNone(layer.bias)
        layer.bias.requires_grad = True

        mesh = DeviceMesh("cpu", [0])
        layer = fully_shard(layer, mesh=mesh, ignored_params={layer.bias})

        hooks = GradSampleHooksFSDP(layer, batch_first=True, loss_reduction="mean")
        try:
            batch_size = 8
            x = torch.randn(batch_size, 4, device=torch.device("cpu"))
            layer.train()
            out = layer(x)
            loss = out.sum()
            loss.backward()

            # Assert that trainable parameters (kept unsharded via `ignored_params`)
            # have populated per-sample gradients.
            trainable_params = [p for p in layer.parameters() if p.requires_grad]
            self.assertGreater(len(trainable_params), 0)

            found = False
            for p in trainable_params:
                if not hasattr(p, "grad_sample") or p.grad_sample is None:
                    continue

                found = True
                if isinstance(p.grad_sample, torch.Tensor):
                    self.assertEqual(p.grad_sample.shape[0], batch_size)
                else:
                    self.assertGreaterEqual(len(p.grad_sample), 1)
                    self.assertEqual(p.grad_sample[0].shape[0], batch_size)

            self.assertTrue(found, "Expected at least one parameter with grad_sample")
        finally:
            hooks.cleanup()


# =============================================================================
# Standard Opacus Usage Tests (without HuggingFace Trainer)
# =============================================================================


class TestStandardOpacusUsage(unittest.TestCase):
    """
    Tests for standard Opacus usage without HuggingFace Trainer.
    
    These tests verify that GradSampleModuleFSDP works correctly in the
    standard Opacus workflow where users wrap their model directly.
    """

    def test_standard_workflow_with_grad_sample_module_fsdp(self):
        """Test standard Opacus workflow using GradSampleModuleFSDP directly."""
        # Create model
        model = SimpleModel()
        
        # Wrap with GradSampleModuleFSDP (standard Opacus pattern)
        gsm = GradSampleModuleFSDP(model, strict=False)
        gsm.enable_hooks()
        gsm.train()
        
        # Forward pass
        batch_size = 8
        x = torch.randn(batch_size, 4)
        output = gsm(x)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Verify per-sample gradients are computed
        for name, param in gsm.named_parameters():
            if param.requires_grad:
                self.assertTrue(
                    hasattr(param, "grad_sample"),
                    f"Parameter {name} missing grad_sample"
                )
                self.assertIsNotNone(
                    param.grad_sample,
                    f"Parameter {name} has None grad_sample"
                )
                
                gs = param.grad_sample[0] if isinstance(param.grad_sample, list) else param.grad_sample
                self.assertEqual(
                    gs.shape[0], batch_size,
                    f"Parameter {name} grad_sample has wrong batch dimension"
                )

    def test_standard_workflow_with_wrap_model_function(self):
        """Test standard Opacus workflow using wrap_model function."""
        model = SimpleModel()
        
        # Use wrap_model with hooks_fsdp mode (wrapping approach)
        gsm = wrap_model(model, "hooks_fsdp", wrap_model=True, strict=False)
        gsm.enable_hooks()
        gsm.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 4)
        output = gsm(x)
        loss = output.sum()
        loss.backward()
        
        # Verify per-sample gradients
        trainable_params = [p for p in gsm.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)
        
        for p in trainable_params:
            self.assertTrue(hasattr(p, "grad_sample"))
            self.assertIsNotNone(p.grad_sample)

    def test_standard_workflow_hooks_only_no_wrapping(self):
        """Test standard Opacus workflow using hooks without wrapping."""
        model = SimpleModel()
        
        # Use wrap_model with hooks_fsdp mode (hooks-only approach)
        hooks = wrap_model(model, "hooks_fsdp", wrap_model=False, strict=False)
        hooks.enable_hooks()
        model.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 4)
        output = model(x)  # Use original model, not wrapped
        loss = output.sum()
        loss.backward()
        
        # Verify per-sample gradients on original model's parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)
        
        for p in trainable_params:
            self.assertTrue(hasattr(p, "grad_sample"))
            self.assertIsNotNone(p.grad_sample)
        
        # Cleanup
        hooks.cleanup()

    def test_standard_workflow_with_lora_like_freezing(self):
        """Test standard Opacus workflow with LoRA-like parameter freezing."""
        model = NestedModel()
        
        # Freeze encoder, keep decoder trainable (LoRA-like scenario)
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        gsm = GradSampleModuleFSDP(model, strict=False)
        gsm.enable_hooks()
        gsm.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 4)
        output = gsm(x)
        loss = output.sum()
        loss.backward()
        
        # Verify only decoder parameters have grad_sample
        for name, param in model.encoder.named_parameters():
            # Frozen params should not have grad_sample populated
            if hasattr(param, "grad_sample"):
                self.assertIsNone(param.grad_sample)
        
        for name, param in model.decoder.named_parameters():
            if param.requires_grad:
                self.assertTrue(hasattr(param, "grad_sample"))
                self.assertIsNotNone(param.grad_sample)

    def test_multiple_batches_accumulation(self):
        """Test that multiple batches can be processed with gradient accumulation."""
        model = SimpleModel()
        gsm = GradSampleModuleFSDP(model, strict=False)
        gsm.enable_hooks()
        gsm.train()
        
        batch_size = 4
        
        # First batch
        x1 = torch.randn(batch_size, 4)
        out1 = gsm(x1)
        loss1 = out1.sum()
        loss1.backward()
        
        # Verify first batch grad_sample
        for p in gsm.parameters():
            if p.requires_grad:
                self.assertTrue(hasattr(p, "grad_sample"))
                self.assertIsNotNone(p.grad_sample)
        
        # Clear grad_sample for next batch (simulating optimizer step)
        for p in gsm.parameters():
            if hasattr(p, "grad_sample"):
                p.grad_sample = None
        
        # Second batch
        x2 = torch.randn(batch_size, 4)
        out2 = gsm(x2)
        loss2 = out2.sum()
        loss2.backward()
        
        # Verify second batch grad_sample
        for p in gsm.parameters():
            if p.requires_grad:
                self.assertTrue(hasattr(p, "grad_sample"))
                self.assertIsNotNone(p.grad_sample)

    def test_eval_mode_no_grad_sample(self):
        """Test that eval mode doesn't compute grad_sample."""
        model = SimpleModel()
        gsm = GradSampleModuleFSDP(model, strict=False)
        gsm.enable_hooks()
        gsm.eval()  # Set to eval mode
        
        batch_size = 4
        x = torch.randn(batch_size, 4)
        
        with torch.no_grad():
            output = gsm(x)
        
        # In eval mode with no_grad, grad_sample should not be computed
        for p in gsm.parameters():
            if hasattr(p, "grad_sample"):
                self.assertIsNone(p.grad_sample)


# =============================================================================
# Tests for Tensor Parallelism (TP) Support in Unified GradSampleHooksFSDP
# =============================================================================

class TestTPSupportInUnifiedFSDP(unittest.TestCase):
    """Tests for TP (Tensor Parallelism) support in the unified GradSampleHooksFSDP class.
    
    TP is auto-detected via DTensor parameters. These tests verify the TP-related
    methods work correctly with both regular tensors and DTensors.
    """

    def test_is_dtensor_with_regular_tensor(self):
        """Test _is_dtensor returns False for regular tensors."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        regular_tensor = torch.randn(4, 4)
        self.assertFalse(hooks._is_dtensor(regular_tensor))

    def test_to_local_tensor_with_regular_tensor(self):
        """Test _to_local_tensor returns the same tensor for regular tensors."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        regular_tensor = torch.randn(4, 4)
        result = hooks._to_local_tensor(regular_tensor)
        self.assertIs(result, regular_tensor)

    def test_analyze_tp_placements_non_dtensor(self):
        """Test _analyze_tp_placements with non-DTensor model."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        # Initially not analyzed
        self.assertFalse(hooks._tp_analyzed)
        self.assertFalse(hooks._has_dtensor_params)
        
        # Trigger analysis
        hooks._analyze_tp_placements()
        
        # After analysis
        self.assertTrue(hooks._tp_analyzed)
        self.assertFalse(hooks._has_dtensor_params)  # No DTensors in regular model
        
        # All params should have _tp_merge_flag = False
        for p in model.parameters():
            if p.requires_grad:
                self.assertTrue(hasattr(p, "_tp_merge_flag"))
                self.assertFalse(p._tp_merge_flag)

    def test_tp_analysis_only_runs_once(self):
        """Test that TP analysis only runs once (lazy initialization)."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        # First analysis
        hooks._analyze_tp_placements()
        self.assertTrue(hooks._tp_analyzed)
        
        # Modify flag to verify second call doesn't re-analyze
        hooks._has_dtensor_params = True
        hooks._analyze_tp_placements()
        
        # Should still be True (not reset)
        self.assertTrue(hooks._has_dtensor_params)

    def test_tp_merge_flag_set_on_forward(self):
        """Test that _tp_merge_flag is set during forward pass."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        hooks.enable_hooks()
        model.train()
        
        # Before forward, params don't have _tp_merge_flag
        for p in model.parameters():
            self.assertFalse(hasattr(p, "_tp_merge_flag"))
        
        # Forward pass triggers lazy TP analysis
        x = torch.randn(4, 4)
        output = model(x)
        
        # After forward, params should have _tp_merge_flag
        for p in model.parameters():
            if p.requires_grad:
                self.assertTrue(hasattr(p, "_tp_merge_flag"))

    def test_grad_sample_with_non_dtensor_model(self):
        """Test full workflow with non-DTensor model (TP auto-detection finds no DTensors)."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        hooks.enable_hooks()
        model.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 4)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Verify grad_sample is computed
        for p in model.parameters():
            if p.requires_grad:
                self.assertTrue(hasattr(p, "grad_sample"))
                self.assertIsNotNone(p.grad_sample)
                self.assertEqual(p.grad_sample.shape[0], batch_size)
        
        hooks.cleanup()


# =============================================================================
# Tests for Context Parallelism (CP) Support in Unified GradSampleHooksFSDP
# =============================================================================

class TestCPSupportInUnifiedFSDP(unittest.TestCase):
    """Tests for CP (Context Parallelism) support in the unified GradSampleHooksFSDP class.
    
    CP is enabled via the cp_group parameter. These tests verify CP-related
    methods work correctly.
    """

    def test_cp_group_none_by_default(self):
        """Test that cp_group is None by default."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        self.assertIsNone(hooks.cp_group)
        self.assertIsNone(hooks._cp_world_size)
        self.assertIsNone(hooks._cp_rank)

    def test_should_aggregate_returns_false_without_distributed(self):
        """Test _should_aggregate_across_cp returns False when not distributed."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        # Without distributed initialized, should return False
        if not dist.is_initialized():
            self.assertFalse(hooks._should_aggregate_across_cp())

    def test_should_aggregate_returns_false_without_cp_group(self):
        """Test _should_aggregate_across_cp returns False when cp_group is None."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        # Even with distributed, no cp_group means no aggregation
        self.assertFalse(hooks._should_aggregate_across_cp())

    def test_set_cp_group_updates_attributes(self):
        """Test set_cp_group updates cp_group and related attributes."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        # Create a mock process group
        with _SingleRankProcessGroup():
            # Use default process group as cp_group
            default_pg = dist.distributed_c10d._get_default_group()
            hooks.set_cp_group(default_pg)
            
            self.assertIsNotNone(hooks.cp_group)
            self.assertEqual(hooks._cp_world_size, 1)
            self.assertEqual(hooks._cp_rank, 0)

    def test_cp_group_in_constructor(self):
        """Test passing cp_group in constructor."""
        model = SimpleModel()
        
        with _SingleRankProcessGroup():
            default_pg = dist.distributed_c10d._get_default_group()
            hooks = GradSampleHooksFSDP(model, strict=False, cp_group=default_pg)
            
            self.assertIsNotNone(hooks.cp_group)
            self.assertEqual(hooks._cp_world_size, 1)
            self.assertEqual(hooks._cp_rank, 0)

    def test_grad_sample_with_cp_enabled(self):
        """Test full workflow with CP enabled."""
        model = SimpleModel()
        
        with _SingleRankProcessGroup():
            default_pg = dist.distributed_c10d._get_default_group()
            hooks = GradSampleHooksFSDP(model, strict=False, cp_group=default_pg)
            hooks.enable_hooks()
            model.train()
            
            batch_size = 4
            x = torch.randn(batch_size, 4)
            output = model(x)
            loss = output.sum()
            loss.backward()
            
            # Verify grad_sample is computed
            for p in model.parameters():
                if p.requires_grad:
                    self.assertTrue(hasattr(p, "grad_sample"))
                    self.assertIsNotNone(p.grad_sample)
            
            hooks.cleanup()


# =============================================================================
# Tests for Combined TP+CP Support in Unified GradSampleHooksFSDP
# =============================================================================

class TestCombinedTPCPSupport(unittest.TestCase):
    """Tests for combined TP+CP support in the unified GradSampleHooksFSDP class."""

    def test_unified_class_supports_both_tp_and_cp(self):
        """Test that the unified class can be configured for both TP and CP."""
        model = SimpleModel()
        
        with _SingleRankProcessGroup():
            default_pg = dist.distributed_c10d._get_default_group()
            hooks = GradSampleHooksFSDP(model, strict=False, cp_group=default_pg)
            
            # CP is configured
            self.assertIsNotNone(hooks.cp_group)
            
            # TP analysis is available (will be triggered on forward)
            self.assertFalse(hooks._tp_analyzed)
            
            # Trigger TP analysis
            hooks._analyze_tp_placements()
            self.assertTrue(hooks._tp_analyzed)

    def test_get_per_sample_norms_non_distributed(self):
        """Test get_per_sample_norms works in non-distributed mode."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        hooks.enable_hooks()
        model.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 4)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Get per-sample norms
        norms = hooks.get_per_sample_norms()
        
        self.assertEqual(norms.shape, (batch_size,))
        self.assertTrue(torch.all(norms >= 0))
        
        hooks.cleanup()

    def test_compute_local_norms(self):
        """Test _compute_local_norms method."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        hooks.enable_hooks()
        model.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 4)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Compute local norms
        norms = hooks._compute_local_norms()
        
        self.assertEqual(norms.shape, (batch_size,))
        self.assertTrue(torch.all(norms >= 0))
        
        hooks.cleanup()


# =============================================================================
# Tests for GradSampleModuleFSDP with TP/CP Support
# =============================================================================

class TestGradSampleModuleFSDPWithTPCP(unittest.TestCase):
    """Tests for GradSampleModuleFSDP wrapper with TP/CP support."""

    def test_module_accepts_cp_group(self):
        """Test GradSampleModuleFSDP accepts cp_group parameter."""
        model = SimpleModel()
        
        with _SingleRankProcessGroup():
            default_pg = dist.distributed_c10d._get_default_group()
            gsm = GradSampleModuleFSDP(model, strict=False, cp_group=default_pg)
            
            self.assertIsNotNone(gsm.cp_group)
            self.assertEqual(gsm._cp_world_size, 1)

    def test_module_workflow_with_cp(self):
        """Test full workflow with GradSampleModuleFSDP and CP enabled."""
        model = SimpleModel()
        
        with _SingleRankProcessGroup():
            default_pg = dist.distributed_c10d._get_default_group()
            gsm = GradSampleModuleFSDP(model, strict=False, cp_group=default_pg)
            gsm.enable_hooks()
            gsm.train()
            
            batch_size = 4
            x = torch.randn(batch_size, 4)
            output = gsm(x)
            loss = output.sum()
            loss.backward()
            
            # Verify grad_sample is computed
            for p in gsm.parameters():
                if p.requires_grad:
                    self.assertTrue(hasattr(p, "grad_sample"))
                    self.assertIsNotNone(p.grad_sample)


# =============================================================================
# Tests for Inheritance and Class Structure
# =============================================================================

class TestUnifiedClassInheritance(unittest.TestCase):
    """Tests verifying the unified class inheritance structure."""

    def test_hooks_fsdp_inherits_from_grad_sample_hooks(self):
        """Test GradSampleHooksFSDP inherits from GradSampleHooks."""
        self.assertTrue(issubclass(GradSampleHooksFSDP, GradSampleHooks))

    def test_module_fsdp_inherits_from_hooks_fsdp(self):
        """Test GradSampleModuleFSDP inherits from GradSampleHooksFSDP."""
        self.assertTrue(issubclass(GradSampleModuleFSDP, GradSampleHooksFSDP))

    def test_unified_class_has_tp_methods(self):
        """Test unified class has TP-related methods."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        self.assertTrue(hasattr(hooks, "_is_dtensor"))
        self.assertTrue(hasattr(hooks, "_to_local_tensor"))
        self.assertTrue(hasattr(hooks, "_analyze_tp_placements"))
        self.assertTrue(callable(hooks._is_dtensor))
        self.assertTrue(callable(hooks._to_local_tensor))
        self.assertTrue(callable(hooks._analyze_tp_placements))

    def test_unified_class_has_cp_methods(self):
        """Test unified class has CP-related methods."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        self.assertTrue(hasattr(hooks, "set_cp_group"))
        self.assertTrue(hasattr(hooks, "_should_aggregate_across_cp"))
        self.assertTrue(hasattr(hooks, "_initialize_cp_info"))
        self.assertTrue(callable(hooks.set_cp_group))
        self.assertTrue(callable(hooks._should_aggregate_across_cp))
        self.assertTrue(callable(hooks._initialize_cp_info))

    def test_unified_class_has_fsdp_methods(self):
        """Test unified class has FSDP-related methods."""
        model = SimpleModel()
        hooks = GradSampleHooksFSDP(model, strict=False)
        
        self.assertTrue(hasattr(hooks, "_get_module_type"))
        self.assertTrue(callable(hooks._get_module_type))


if __name__ == "__main__":
    unittest.main()
