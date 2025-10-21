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
Example demonstrating arithmetic operations on DPTensorFastGradientClipping.

This shows how to compose multiple losses using arithmetic operations
while maintaining ghost clipping functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


class MultiTaskModel(nn.Module):
    """Simple multi-task model with shared backbone"""

    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Task-specific heads
        self.task1_head = nn.Linear(hidden_dim, output_dim1)
        self.task2_head = nn.Linear(hidden_dim, output_dim2)

    def forward(self, x):
        # Shared backbone
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Task outputs
        task1_output = self.task1_head(x)
        task2_output = self.task2_head(x)

        return task1_output, task2_output

    def l2_regularization(self):
        """Compute L2 regularization"""
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param) ** 2
        return l2_reg


def main():
    # Configuration
    n_samples = 1000
    input_dim = 20
    hidden_dim = 64
    output_dim1 = 10  # Task 1: classification
    output_dim2 = 5  # Task 2: classification
    batch_size = 32

    # Create synthetic multi-task dataset
    X = torch.randn(n_samples, input_dim)
    y1 = torch.randint(0, output_dim1, (n_samples,))
    y2 = torch.randint(0, output_dim2, (n_samples,))

    dataset = TensorDataset(X, y1, y2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MultiTaskModel(input_dim, hidden_dim, output_dim1, output_dim2)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setup privacy engine with ghost clipping
    privacy_engine = PrivacyEngineGradSampleController()

    controller, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        grad_sample_mode="ghost",  # Use ghost clipping
        return_controller=True,
    )

    print("=" * 60)
    print("Ghost Clipping with Arithmetic Operations Example")
    print("=" * 60)
    print(f"Model type preserved: {isinstance(model, MultiTaskModel)}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print()

    # Setup loss functions for both tasks
    criterion1 = nn.CrossEntropyLoss(reduction="mean")
    criterion2 = nn.CrossEntropyLoss(reduction="mean")

    # Wrap loss functions for ghost clipping
    loss_fn1 = DPLossFastGradientClipping(controller, optimizer, criterion1, "mean")
    loss_fn2 = DPLossFastGradientClipping(controller, optimizer, criterion2, "mean")

    # Training parameters
    epochs = 3
    delta = 1e-5
    task1_weight = 1.0
    task2_weight = 0.5
    reg_weight = 0.01

    print("Training with composed losses:")
    print(f"  Task 1 weight: {task1_weight}")
    print(f"  Task 2 weight: {task2_weight}")
    print(f"  Regularization weight: {reg_weight}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_task1_loss = 0
        total_task2_loss = 0
        total_reg_loss = 0

        for batch_idx, (data, target1, target2) in enumerate(dataloader):
            data = data.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)

            optimizer.zero_grad()

            # Forward pass
            output1, output2 = model(data)

            # Compute individual losses (returns DPTensorFastGradientClipping objects)
            task1_loss = loss_fn1(output1, target1)
            task2_loss = loss_fn2(output2, target2)

            # Get regularization as scalar
            reg_loss_value = model.l2_regularization().item()

            # Compose losses using arithmetic operations!
            # This demonstrates multiplication, addition, and scalar operations
            total = task1_weight * task1_loss + task2_weight * task2_loss

            # For regularization, we need to add it as a weighted term
            # Since reg is a scalar, we can add it to the loss
            reg_loss_scalar = reg_weight * model.l2_regularization()

            # Backward pass (ghost clipping handles the composed loss)
            total.backward()

            # Add regularization gradient separately (not through ghost clipping)
            reg_loss_scalar.backward()

            optimizer.step()

            # Track losses
            total_loss += total.item()
            total_task1_loss += task1_loss.item()
            total_task2_loss += task2_loss.item()
            total_reg_loss += reg_loss_value

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1} [{batch_idx}/{len(dataloader)}] "
                    f"Total: {total.item():.4f} | "
                    f"Task1: {task1_loss.item():.4f} | "
                    f"Task2: {task2_loss.item():.4f} | "
                    f"Reg: {reg_loss_value:.4f}"
                )

        # Compute privacy budget
        epsilon = privacy_engine.get_epsilon(delta)

        avg_total = total_loss / len(dataloader)
        avg_task1 = total_task1_loss / len(dataloader)
        avg_task2 = total_task2_loss / len(dataloader)
        avg_reg = total_reg_loss / len(dataloader)

        print()
        print(f"Epoch {epoch + 1}/{epochs} Summary:")
        print(f"  Total Loss: {avg_total:.4f}")
        print(f"  Task 1 Loss: {avg_task1:.4f}")
        print(f"  Task 2 Loss: {avg_task2:.4f}")
        print(f"  Regularization: {avg_reg:.4f}")
        print(f"  Privacy: ε = {epsilon:.2f} (δ = {delta})")
        print()

    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    print()
    print("Arithmetic operations demonstrated:")
    print("  ✓ Scalar multiplication: task1_weight * task1_loss")
    print("  ✓ Addition: weighted_task1 + weighted_task2")
    print("  ✓ Combined: task1_weight * task1_loss + task2_weight * task2_loss")
    print()
    print("All operations preserved ghost clipping functionality!")

    # Cleanup
    controller.cleanup()
    print("Controller cleaned up successfully.")


if __name__ == "__main__":
    main()
