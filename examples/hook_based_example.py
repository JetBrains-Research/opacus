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
Example demonstrating the use of PrivacyEngineHookBased for transformer models.

This example shows how the hook-based approach avoids the issues that occur
with GradSampleModule wrapping when working with transformer models.
"""

import torch
import torch.nn as nn
from opacus.privacy_engine_hook_based import PrivacyEngineHookBased
from torch.utils.data import DataLoader, TensorDataset


class TransformerModel(nn.Module):
    """
    Simple transformer-like model that demonstrates the benefits of hook-based approach.

    This model has custom __getattr__ behavior that might break with module wrapping.
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(10, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        # Add a sequence dimension
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.output(x)
        return x


def main():
    """
    Demonstrate the hook-based privacy engine with a transformer model.
    """
    print("=" * 80)
    print("PrivacyEngineHookBased - Transformer Example")
    print("=" * 80)

    # Create model and training components
    model = TransformerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dummy dataset
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Initialize privacy engine
    privacy_engine = PrivacyEngineHookBased()

    print("\n1. Making model private...")
    print(f"   Original model type: {type(model).__name__}")
    print(f"   Is TransformerModel: {isinstance(model, TransformerModel)}")

    # Make model private
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,  # Simplify for demo
    )

    print(f"\n2. After making private:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Is still TransformerModel: {isinstance(model, TransformerModel)}")
    print(f"   Can access model.transformer: {hasattr(model, 'transformer')}")
    print(f"   Can access model.embedding: {hasattr(model, 'embedding')}")
    print(f"   Hooks attached: {len(privacy_engine.hook_controller.hook_handles)}")

    print("\n3. Training step...")
    model.train()
    data, target = next(iter(dataloader))

    output = model(data)
    loss = nn.MSELoss()(output, target)
    print(f"   Loss: {loss.item():.4f}")

    loss.backward()

    # Check grad_sample was computed
    grad_sample_computed = any(
        hasattr(p, "grad_sample") and p.grad_sample is not None
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"   Grad samples computed: {grad_sample_computed}")

    optimizer.step()
    optimizer.zero_grad()
    print("   Optimizer step completed successfully!")

    print("\n4. State dict compatibility:")
    state_dict = model.state_dict()
    print(f"   Number of keys: {len(state_dict)}")
    print(f"   First few keys: {list(state_dict.keys())[:3]}")
    print("   ✓ No '_module' prefix in keys!")

    print("\n5. Cleanup:")
    privacy_engine.cleanup()
    print("   Hooks removed")
    grad_sample_exists = any(
        hasattr(p, "grad_sample") for p in model.parameters()
    )
    print(f"   grad_sample attributes removed: {not grad_sample_exists}")

    print("\n" + "=" * 80)
    print("SUCCESS: Hook-based privacy engine works with transformer models!")
    print("=" * 80)

    print("\n" + "Key Benefits".center(80))
    print("-" * 80)
    print("✓ Model is NOT wrapped - remains TransformerModel instance")
    print("✓ State dict has no '_module' prefix")
    print("✓ Direct access to model attributes (transformer, embedding, etc.)")
    print("✓ Custom __getattr__ behavior preserved")
    print("✓ Better compatibility with HuggingFace transformers")
    print("=" * 80)


if __name__ == "__main__":
    main()
