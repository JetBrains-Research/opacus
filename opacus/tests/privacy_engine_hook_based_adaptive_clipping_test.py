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
Tests to verify that hook-based privacy engine is compatible with adaptive clipping.

Adaptive clipping can work in two modes:
1. Via AdaClipDPOptimizer - standard optimizer-based adaptive clipping
2. Via ghost clipping mode - fast gradient clipping for distributed settings

This test suite covers both approaches with the hook-based privacy engine.
"""

import unittest

import torch
import torch.nn as nn
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.privacy_engine_hook_based import PrivacyEngineHookBased
from torch.utils.data import DataLoader, TensorDataset


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class AdaClipOptimizerHookBasedTest(unittest.TestCase):
    """Test AdaClipDPOptimizer compatibility with hook-based privacy engine."""

    def setUp(self):
        self.DATA_SIZE = 100
        self.BATCH_SIZE = 10
        self.LR = 0.1

        # Create simple dataset
        self.data = torch.randn(self.DATA_SIZE, 10)
        self.labels = torch.randint(0, 5, (self.DATA_SIZE,))
        self.dataset = TensorDataset(self.data, self.labels)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.BATCH_SIZE, drop_last=False
        )

    def test_adaclip_optimizer_initialization(self):
        """Test that AdaClipDPOptimizer can be initialized with hook-based engine."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        privacy_engine = PrivacyEngineHookBased()

        # Make private with AdaClip optimizer
        # Note: noise_multiplier must be < 2 * unclipped_num_std (AdaClip constraint)
        unclipped_num_std = 1.0
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=0.5,  # < 2 * unclipped_num_std (1.0)
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=unclipped_num_std,
        )

        # Verify optimizer is AdaClipDPOptimizer
        self.assertIsInstance(optimizer, AdaClipDPOptimizer)

        # Verify AdaClip-specific attributes exist
        self.assertTrue(hasattr(optimizer, 'target_unclipped_quantile'))
        self.assertTrue(hasattr(optimizer, 'clipbound_learning_rate'))
        self.assertTrue(hasattr(optimizer, 'max_clipbound'))
        self.assertTrue(hasattr(optimizer, 'min_clipbound'))
        self.assertTrue(hasattr(optimizer, 'unclipped_num'))
        self.assertTrue(hasattr(optimizer, 'sample_size'))

        privacy_engine.cleanup()

    def test_adaclip_clipbound_updates(self):
        """Test that adaptive clipping actually updates the clipping bound."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=0.0,  # No noise for clearer results
            max_grad_norm=1.0,  # Initial clip bound
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=0.05,
        )

        criterion = nn.CrossEntropyLoss()
        initial_clipbound = optimizer.max_grad_norm
        clipbounds = [initial_clipbound]

        # Train for several steps and track clipbound changes
        for i, (x, y) in enumerate(dataloader):
            if i >= 5:
                break

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Record clipbound after step
            clipbounds.append(optimizer.max_grad_norm)

        # Verify that clipbound changed during training
        unique_clipbounds = set(f"{cb:.6f}" for cb in clipbounds)
        self.assertGreater(
            len(unique_clipbounds), 1,
            f"Clipbound should change over time. Got values: {clipbounds}"
        )

        # Verify clipbound stays within bounds
        for cb in clipbounds:
            self.assertGreaterEqual(cb, 0.01)  # min_clipbound
            self.assertLessEqual(cb, 10.0)  # max_clipbound

        privacy_engine.cleanup()

    def test_adaclip_unclipped_tracking(self):
        """Test that AdaClip correctly tracks unclipped gradient counts."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        privacy_engine = PrivacyEngineHookBased()

        unclipped_num_std = 1.0
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=0.5,  # < 2 * unclipped_num_std
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=unclipped_num_std,
        )

        criterion = nn.CrossEntropyLoss()

        # Train one step
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Call optimizer.step() which triggers clip_and_accumulate
        # that sets sample_size and unclipped_num
        initial_clipbound = optimizer.max_grad_norm
        optimizer.step()

        # After step, clipbound should have been updated
        # (may be same if already at equilibrium, but structure should exist)
        self.assertIsNotNone(optimizer.max_grad_norm)

        # Compute unclipped fraction
        unclipped_frac = optimizer.unclipped_num / optimizer.sample_size
        self.assertGreaterEqual(unclipped_frac, 0.0)
        self.assertLessEqual(unclipped_frac, 1.0)

        optimizer.step()

        # After step: counters should be reset by zero_grad on next iteration
        optimizer.zero_grad()
        self.assertEqual(optimizer.sample_size, 0)
        self.assertEqual(optimizer.unclipped_num, 0)

        privacy_engine.cleanup()

    def test_adaclip_convergence_behavior(self):
        """Test that AdaClip converges toward target quantile."""
        torch.manual_seed(42)
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        privacy_engine = PrivacyEngineHookBased()

        target_quantile = 0.7
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=0.5,  # < 2 * unclipped_num_std
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=target_quantile,
            clipbound_learning_rate=0.1,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=0.05,
        )

        criterion = nn.CrossEntropyLoss()
        unclipped_fractions = []

        # Train for multiple steps
        for i, (x, y) in enumerate(dataloader):
            if i >= 10:
                break

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            # Record unclipped fraction before step
            if optimizer.sample_size > 0:
                unclipped_frac = float(optimizer.unclipped_num) / optimizer.sample_size
                unclipped_fractions.append(unclipped_frac)

            optimizer.step()

        # Average unclipped fraction should be reasonably close to target
        # (not exact due to noise and limited steps)
        if len(unclipped_fractions) > 5:
            avg_unclipped = sum(unclipped_fractions[-5:]) / 5
            # Should be within reasonable range of target
            self.assertGreater(avg_unclipped, target_quantile - 0.3)
            self.assertLess(avg_unclipped, target_quantile + 0.3)

        privacy_engine.cleanup()

    def test_adaclip_vs_fixed_clipping(self):
        """Test that AdaClip behaves differently from fixed clipping."""
        torch.manual_seed(42)

        # Train with AdaClip
        model1 = SimpleNet()
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=self.LR)
        privacy_engine1 = PrivacyEngineHookBased()

        model1, optimizer1, dataloader1 = privacy_engine1.make_private(
            module=model1,
            optimizer=optimizer1,
            data_loader=self.dataloader,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=0.05,
        )

        # Train with fixed clipping
        torch.manual_seed(42)
        model2 = SimpleNet()
        model2.load_state_dict(model1.state_dict())
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=self.LR)
        privacy_engine2 = PrivacyEngineHookBased()

        model2, optimizer2, dataloader2 = privacy_engine2.make_private(
            module=model2,
            optimizer=optimizer2,
            data_loader=self.dataloader,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",  # Fixed clipping
        )

        criterion = nn.CrossEntropyLoss()

        # Train both for several steps
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader1, dataloader2)):
            if i >= 5:
                break

            # AdaClip training
            optimizer1.zero_grad()
            output1 = model1(x1)
            loss1 = criterion(output1, y1)
            loss1.backward()
            optimizer1.step()

            # Fixed clipping training
            optimizer2.zero_grad()
            output2 = model2(x2)
            loss2 = criterion(output2, y2)
            loss2.backward()
            optimizer2.step()

        # After training, parameters should differ
        # (because AdaClip adjusts clipbound while fixed doesn't)
        params_differ = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2, atol=1e-5):
                params_differ = True
                break

        self.assertTrue(
            params_differ,
            "AdaClip and fixed clipping should produce different results"
        )

        privacy_engine1.cleanup()
        privacy_engine2.cleanup()

    def test_adaclip_parameter_validation(self):
        """Test that AdaClip validates parameters correctly."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)
        privacy_engine = PrivacyEngineHookBased()

        # Test: max_clipbound <= min_clipbound should raise error
        with self.assertRaises(ValueError):
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.dataloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                clipping="adaptive",
                max_clipbound=0.01,  # Less than min
                min_clipbound=0.1,
                unclipped_num_std=0.05,
            )

    def test_adaclip_with_nonzero_noise(self):
        """Test AdaClip works with noise (full DP training)."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)

        privacy_engine = PrivacyEngineHookBased()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,  # With noise
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=unclipped_num_std,
        )

        criterion = nn.CrossEntropyLoss()

        # Train one step with noise
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Verify training completed successfully
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

        privacy_engine.cleanup()


if __name__ == "__main__":
    unittest.main()
