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

import unittest

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_controller_fast_gradient_clipping import (
    GradSampleControllerFastGradientClipping,
)
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GradSampleControllerFastGradientClippingTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5
        self.max_grad_norm = 1.0
        self.loss_reduction = "mean"

    def test_controller_creation(self):
        """Test that controller can be created without wrapping model"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)
        original_type = type(model)

        controller = GradSampleControllerFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=True,
        )

        # Model type should be preserved
        self.assertEqual(type(model), original_type)
        self.assertIsInstance(model, SimpleModel)

        # Controller should have hooks
        self.assertTrue(len(controller.autograd_grad_sample_hooks) > 0)
        self.assertTrue(controller.hooks_enabled)

        # Clean up
        controller.cleanup()

    def test_norm_sample_computation(self):
        """Test that norm samples are computed correctly"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        controller = GradSampleControllerFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,  # Use fast gradient clipping for testing
        )

        # Create dummy input and target
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass
        model.train()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target, reduction=self.loss_reduction)
        loss.backward()

        # Check that norm samples are computed
        for param in controller.trainable_parameters:
            if param.requires_grad:
                self.assertIsNotNone(param._norm_sample)
                self.assertEqual(param._norm_sample.shape[0], self.batch_size)

        # Clean up
        controller.cleanup()

    def test_clipping_coefficient(self):
        """Test that clipping coefficients are computed correctly"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        controller = GradSampleControllerFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,
        )

        # Create dummy input and target
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass
        model.train()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target, reduction=self.loss_reduction)
        loss.backward()

        # Get clipping coefficient
        coeff = controller.get_clipping_coef()

        # Coefficients should be between 0 and 1
        self.assertTrue(torch.all(coeff >= 0))
        self.assertTrue(torch.all(coeff <= 1))
        self.assertEqual(coeff.shape[0], self.batch_size)

        # Clean up
        controller.cleanup()

    def test_hooks_enable_disable(self):
        """Test that hooks can be enabled and disabled"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        controller = GradSampleControllerFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
        )

        # Hooks should be enabled by default
        self.assertTrue(controller.hooks_enabled)

        # Disable hooks
        controller.disable_hooks()
        self.assertFalse(controller.hooks_enabled)

        # Enable hooks
        controller.enable_hooks()
        self.assertTrue(controller.hooks_enabled)

        # Clean up
        controller.cleanup()

    def test_cleanup(self):
        """Test that cleanup removes all hooks and attributes"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        controller = GradSampleControllerFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
        )

        # Verify hooks and attributes exist
        self.assertTrue(len(controller.autograd_grad_sample_hooks) > 0)
        for param in model.parameters():
            self.assertTrue(hasattr(param, "_forward_counter"))

        # Cleanup
        controller.cleanup()

        # Verify hooks are removed
        self.assertEqual(len(controller.autograd_grad_sample_hooks), 0)

        # Verify attributes are removed
        for param in model.parameters():
            self.assertFalse(hasattr(param, "grad_sample"))
            self.assertFalse(hasattr(param, "_forward_counter"))
            self.assertFalse(hasattr(param, "_norm_sample"))

    def test_controller_vs_wrapped_equivalence(self):
        """Test that controller produces same norms as wrapped module"""
        # Create two identical models
        model_controller = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)
        model_wrapped = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Copy weights
        model_wrapped.load_state_dict(model_controller.state_dict())

        # Create controller and wrapped module
        controller = GradSampleControllerFastGradientClipping(
            model_controller,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,  # Use FGC for comparison
        )

        wrapped = GradSampleModuleFastGradientClipping(
            model_wrapped,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,
        )

        # Create dummy input and target
        torch.manual_seed(42)
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass for controller
        model_controller.train()
        output_controller = model_controller(x.clone())
        loss_controller = nn.functional.cross_entropy(
            output_controller, target, reduction=self.loss_reduction
        )
        loss_controller.backward()

        # Forward and backward pass for wrapped
        wrapped.train()
        output_wrapped = wrapped(x.clone())
        loss_wrapped = nn.functional.cross_entropy(
            output_wrapped, target, reduction=self.loss_reduction
        )
        loss_wrapped.backward()

        # Get norms
        norm_controller = controller.get_norm_sample()
        norm_wrapped = wrapped.get_norm_sample()

        # Norms should be very close
        self.assertTrue(
            torch.allclose(norm_controller, norm_wrapped, rtol=1e-4, atol=1e-4)
        )

        # Clean up
        controller.cleanup()

    def test_isinstance_preserved(self):
        """Test that isinstance checks work after controller attachment"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Before controller
        self.assertIsInstance(model, SimpleModel)
        self.assertIsInstance(model, nn.Module)

        # Create controller
        controller = GradSampleControllerFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
        )

        # After controller - isinstance should still work
        self.assertIsInstance(model, SimpleModel)
        self.assertIsInstance(model, nn.Module)

        # Clean up
        controller.cleanup()

    def test_dp_tensor_arithmetic_operations(self):
        """Test that DPTensorFastGradientClipping supports arithmetic operations"""
        from opacus.utils.fast_gradient_clipping_utils import DPTensorFastGradientClipping
        from opacus.optimizers import DPOptimizerFastGradientClipping

        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        controller = GradSampleControllerFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
        )

        dp_optimizer = DPOptimizerFastGradientClipping(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=self.batch_size,
            loss_reduction=self.loss_reduction,
        )

        loss_per_sample = torch.randn(self.batch_size)
        dp_loss = DPTensorFastGradientClipping(
            controller, dp_optimizer, loss_per_sample, self.loss_reduction
        )

        # Test division
        divided_loss = dp_loss / 2.0
        self.assertIsInstance(divided_loss, DPTensorFastGradientClipping)
        self.assertTrue(
            torch.allclose(divided_loss.loss_per_sample, loss_per_sample / 2.0)
        )

        # Test multiplication
        multiplied_loss = dp_loss * 3.0
        self.assertIsInstance(multiplied_loss, DPTensorFastGradientClipping)
        self.assertTrue(
            torch.allclose(multiplied_loss.loss_per_sample, loss_per_sample * 3.0)
        )

        # Test right multiplication
        rmultiplied_loss = 3.0 * dp_loss
        self.assertIsInstance(rmultiplied_loss, DPTensorFastGradientClipping)
        self.assertTrue(
            torch.allclose(rmultiplied_loss.loss_per_sample, 3.0 * loss_per_sample)
        )

        # Test addition with scalar
        added_loss = dp_loss + 1.0
        self.assertIsInstance(added_loss, DPTensorFastGradientClipping)
        self.assertTrue(
            torch.allclose(added_loss.loss_per_sample, loss_per_sample + 1.0)
        )

        # Test addition with another DPTensor
        loss_per_sample2 = torch.randn(self.batch_size)
        dp_loss2 = DPTensorFastGradientClipping(
            controller, dp_optimizer, loss_per_sample2, self.loss_reduction
        )
        summed_loss = dp_loss + dp_loss2
        self.assertIsInstance(summed_loss, DPTensorFastGradientClipping)
        self.assertTrue(
            torch.allclose(
                summed_loss.loss_per_sample, loss_per_sample + loss_per_sample2
            )
        )

        # Test subtraction
        subtracted_loss = dp_loss - 0.5
        self.assertIsInstance(subtracted_loss, DPTensorFastGradientClipping)
        self.assertTrue(
            torch.allclose(subtracted_loss.loss_per_sample, loss_per_sample - 0.5)
        )

        # Test right subtraction
        rsubtracted_loss = 1.0 - dp_loss
        self.assertIsInstance(rsubtracted_loss, DPTensorFastGradientClipping)
        self.assertTrue(
            torch.allclose(rsubtracted_loss.loss_per_sample, 1.0 - loss_per_sample)
        )

        # Test negation
        negated_loss = -dp_loss
        self.assertIsInstance(negated_loss, DPTensorFastGradientClipping)
        self.assertTrue(torch.allclose(negated_loss.loss_per_sample, -loss_per_sample))

        # Test item()
        item_value = dp_loss.item()
        self.assertIsInstance(item_value, float)

        # Test string representations
        repr_str = repr(dp_loss)
        self.assertIn("DPTensorFastGradientClipping", repr_str)
        str_str = str(dp_loss)
        self.assertIn("DPTensorFastGradientClipping", str_str)

        # Clean up
        controller.cleanup()


if __name__ == "__main__":
    unittest.main()
