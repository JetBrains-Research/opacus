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
from opacus.privacy_engine_hook_based import PrivacyEngineHookBased
from torch.utils.data import DataLoader, TensorDataset


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PrivacyEngineHookBasedTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        # Create a simple dataset
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=10)

    def test_initialization(self):
        """Test that PrivacyEngineHookBased can be initialized."""
        privacy_engine = PrivacyEngineHookBased()
        self.assertIsNotNone(privacy_engine)
        self.assertIsNotNone(privacy_engine.accountant)

    def test_make_private_returns_unwrapped_model(self):
        """Test that make_private returns the original model, not a wrapper."""
        privacy_engine = PrivacyEngineHookBased()

        original_model = self.model
        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Model should be the same object, not wrapped
        self.assertIs(model, original_model)
        self.assertIsInstance(model, SimpleNet)
        self.assertNotIsInstance(model, nn.Module.__class__)  # Not a wrapper

    def test_hooks_are_attached(self):
        """Test that hooks are properly attached to the model."""
        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Check that hook controller was created
        self.assertIsNotNone(privacy_engine.hook_controller)
        self.assertTrue(len(privacy_engine.hook_controller.hook_handles) > 0)

    def test_grad_sample_computation(self):
        """Test that per-sample gradients are computed correctly."""
        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,  # Disable for simpler test
        )

        # Get a batch
        data, target = next(iter(dataloader))

        # Forward pass
        model.train()
        output = model(data)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Check that grad_sample was computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad_sample)
                # grad_sample should have batch dimension
                self.assertEqual(param.grad_sample.shape[0], data.shape[0])

    def test_optimizer_step(self):
        """Test that optimizer step works with hook-based approach."""
        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

        # Get a batch
        data, target = next(iter(dataloader))

        # Training step
        model.train()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Store original parameters
        original_params = [p.clone() for p in model.parameters()]

        # Optimizer step should work
        optimizer.step()
        optimizer.zero_grad()

        # Parameters should have changed
        for original_param, current_param in zip(original_params, model.parameters()):
            self.assertFalse(torch.allclose(original_param, current_param))

    def test_cleanup(self):
        """Test that cleanup removes hooks and attributes."""
        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Cleanup
        privacy_engine.cleanup()

        # Hook controller should be None
        self.assertIsNone(privacy_engine.hook_controller)

        # Parameters should not have grad_sample attribute
        for param in model.parameters():
            self.assertFalse(hasattr(param, "grad_sample"))

    def test_state_dict_unchanged(self):
        """Test that state_dict remains unchanged (no wrapper prefix)."""
        privacy_engine = PrivacyEngineHookBased()

        # Get state dict before making private
        state_dict_before = self.model.state_dict()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Get state dict after making private
        state_dict_after = model.state_dict()

        # Keys should be identical (no _module prefix)
        self.assertEqual(set(state_dict_before.keys()), set(state_dict_after.keys()))

        # Values should be identical
        for key in state_dict_before.keys():
            self.assertTrue(
                torch.allclose(state_dict_before[key], state_dict_after[key])
            )

    def test_model_attribute_access(self):
        """Test that model attributes can be accessed directly."""
        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Should be able to access model attributes directly
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.fc2, nn.Linear)

        # Should be able to check module type
        self.assertIsInstance(model, SimpleNet)

    def test_checkpoint_save_load(self):
        """Test that checkpoints can be saved and loaded."""
        import tempfile

        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Train for one step
        data, target = next(iter(dataloader))
        model.train()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False) as f:
            checkpoint_path = f.name
            privacy_engine.save_checkpoint(
                path=checkpoint_path,
                module=model,
                optimizer=optimizer,
            )

        # Create new engine and model
        new_privacy_engine = PrivacyEngineHookBased()
        new_model = SimpleNet()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)

        new_model, new_optimizer, _ = new_privacy_engine.make_private(
            module=new_model,
            optimizer=new_optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Load checkpoint
        new_privacy_engine.load_checkpoint(
            path=checkpoint_path,
            module=new_model,
            optimizer=new_optimizer,
        )

        # Parameters should match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # Clean up
        import os
        os.unlink(checkpoint_path)

    def test_context_manager(self):
        """Test that context manager properly cleans up."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with PrivacyEngineHookBased() as privacy_engine:
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.dataloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )

            # Hook controller should exist
            self.assertIsNotNone(privacy_engine.hook_controller)

            # Train one step
            data, target = next(iter(dataloader))
            model.train()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()

        # After exiting context, cleanup should have happened
        self.assertIsNone(privacy_engine.hook_controller)

        # Parameters should not have grad_sample
        for param in model.parameters():
            self.assertFalse(hasattr(param, "grad_sample"))

    def test_context_manager_with_exception(self):
        """Test that context manager cleans up even with exceptions."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        try:
            with PrivacyEngineHookBased() as privacy_engine:
                model, optimizer, dataloader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=self.dataloader,
                    noise_multiplier=1.0,
                    max_grad_norm=1.0,
                )

                # Hook controller should exist
                self.assertIsNotNone(privacy_engine.hook_controller)

                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Cleanup should have happened despite exception
        self.assertIsNone(privacy_engine.hook_controller)

    def test_ddp_model_handling(self):
        """Test that DDP-wrapped models are properly handled."""
        model = SimpleNet()

        # Note: We use a mock DDP that doesn't inherit from actual DDP
        # In this case, the model is treated as a regular module (not unwrapped)
        # Real DDP instances would be unwrapped by HookController
        class MockDDP(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)

            def parameters(self, recurse=True):
                return self.module.parameters(recurse=recurse)

        ddp_model = MockDDP(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        privacy_engine = PrivacyEngineHookBased()

        # Make private with DDP model
        ddp_model, optimizer, dataloader = privacy_engine.make_private(
            module=ddp_model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

        # Hook controller should exist
        # Since MockDDP is not a real DDP, target_module will be the MockDDP itself
        self.assertIsNotNone(privacy_engine.hook_controller)
        # For real DDP instances, this would be unwrapped to model
        # But MockDDP is treated as a regular module
        self.assertIs(privacy_engine.hook_controller.target_module, ddp_model)

        # Should be able to train
        data, target = next(iter(dataloader))
        ddp_model.train()
        output = ddp_model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Grad samples should be on underlying module's parameters
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad_sample)

        privacy_engine.cleanup()


if __name__ == "__main__":
    unittest.main()
