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

import pytest
import torch
from opacus.data_loader import DPDataLoader
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader, TensorDataset


class DPDataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_size = 10
        self.dimension = 7
        self.num_classes = 11

    def test_empty_batches_skipped(self) -> None:
        """Test that samplers skip empty batches but count all sampling rounds."""
        # Use very low sample rate to ensure we get some empty batches
        num_samples = 100
        sample_rate = 0.01  # Very low probability of selecting samples
        steps = 100

        # Create a sampler with fixed seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(42)

        sampler = UniformWithReplacementSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            generator=generator,
            steps=steps,
        )

        # Collect all batches
        batches = list(sampler)

        # Verify that:
        # 1. All yielded batches are non-empty
        for batch in batches:
            self.assertGreater(len(batch), 0, "Sampler should not yield empty batches")

        # 2. The number of yielded batches is less than or equal to steps
        # (because empty batches are skipped)
        self.assertLessEqual(len(batches), steps)

        # 3. With very low sample_rate, we expect fewer batches than steps
        # (some batches will be empty and skipped)
        self.assertLess(
            len(batches),
            steps,
            "With low sample_rate, some batches should be empty and skipped",
        )

    def test_no_empty_batches_with_dataloader(self) -> None:
        """Test that DPDataLoader never yields empty batches."""
        x = torch.randn(self.data_size, self.dimension)
        y = torch.randint(low=0, high=self.num_classes, size=(self.data_size,))

        dataset = TensorDataset(x, y)

        # Use very low sample rate to maximize chance of empty batches
        data_loader = DPDataLoader(dataset, sample_rate=0.01)

        # Iterate through all batches
        batch_count = 0
        for x_batch, y_batch in data_loader:
            # Verify batch is not empty
            self.assertGreater(
                x_batch.size(0), 0, "DPDataLoader should never yield empty batches"
            )
            self.assertGreater(
                y_batch.size(0), 0, "DPDataLoader should never yield empty batches"
            )
            batch_count += 1

        # We should have gotten at least some batches (though not all steps)
        self.assertGreater(batch_count, 0, "Should have yielded at least one batch")

    def test_sampler_length_unchanged(self) -> None:
        """Test that __len__ returns the expected number of steps, not actual batches."""
        num_samples = 100
        sample_rate = 0.01
        steps = 50

        sampler = UniformWithReplacementSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            steps=steps,
        )

        # __len__ should return steps (for privacy accounting)
        self.assertEqual(len(sampler), steps)

        # But actual number of non-empty batches will be less
        actual_batches = len(list(sampler))
        self.assertLess(actual_batches, steps)

    def test_drop_last_true(self) -> None:
        x = torch.randn(self.data_size, self.dimension)

        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, drop_last=True)
        _ = DPDataLoader.from_data_loader(data_loader)
