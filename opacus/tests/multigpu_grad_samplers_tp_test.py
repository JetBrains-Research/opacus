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
Tests for grad samplers with Tensor Parallelism support.

This test suite covers two scenarios:

1. **Real TP tests**: Linear layers with actual tensor parallelism sharding
   (ColwiseParallel/RowwiseParallel) to verify grad_sample computation works
   correctly with DTensor sharding.

2. **TP environment compatibility tests**: Layer types that cannot be effectively
   sharded but need to work in a TP environment. These tests verify that the
   `implicit_replication()` context manager handles mixed Tensor/DTensor operations
   correctly when gradients flow through unsharded layers in a TP model.

Note: Conv, Embedding, normalization layers, and complex layers (LSTM, Attention)
cannot be effectively sharded due to dimension constraints, but must still work
when other parts of the model use TP.
"""

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleControllerTP
from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from opacus.layers.dp_rnn import DPLSTM
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def setup(rank, world_size):
    """Setup distributed environment for testing."""
    if sys.platform == "win32":
        raise ValueError("Windows platform is not supported for this test")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
    )


def cleanup():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


# ============================================================================
# Test Modules for Different Layer Types
# ============================================================================


class LinearModule(nn.Module):
    """Module with Linear layers."""

    def __init__(self, input_size=8, hidden_size=16, output_size=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Conv2dModule(nn.Module):
    """Module with Conv2d layers and Linear layers for realistic TP testing.

    Linear layers can be sharded while Conv2d cannot, testing implicit_replication.
    """

    def __init__(self, in_channels=3, hidden_channels=16, out_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        # Add Linear layers that can be sharded to test implicit_replication
        self.fc = nn.Linear(out_channels * 16 * 16, 32, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # Flatten and pass through Linear (which will be sharded)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LayerNormModule(nn.Module):
    """Module with LayerNorm - normalization layers should not be sharded."""

    def __init__(self, input_size=16, hidden_size=32, output_size=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        # Note: LayerNorm expects non-sharded input, so we need careful TP planning
        x = self.norm(x)
        return self.fc2(x)


class GroupNormModule(nn.Module):
    """Module with GroupNorm and Linear layer for realistic TP testing.

    Linear layer can be sharded while GroupNorm cannot, testing implicit_replication.
    """

    def __init__(self, in_channels=16, hidden_channels=32, out_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=4, num_channels=hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        # Add Linear layer that can be sharded to test implicit_replication
        self.fc = nn.Linear(out_channels * 8 * 8, 32, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        # Flatten and pass through Linear (which will be sharded)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class InstanceNormModule(nn.Module):
    """Module with InstanceNorm and Linear layer for realistic TP testing.

    Linear layer can be sharded while InstanceNorm cannot, testing implicit_replication.
    """

    def __init__(self, in_channels=16, hidden_channels=32, out_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.InstanceNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        # Add Linear layer that can be sharded to test implicit_replication
        self.fc = nn.Linear(out_channels * 8 * 8, 32, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        # Flatten and pass through Linear (which will be sharded)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RMSNormModule(nn.Module):
    """Module with RMSNorm."""

    def __init__(self, input_size=16, hidden_size=32, output_size=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.norm = nn.RMSNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        return self.fc2(x)


class EmbeddingModule(nn.Module):
    """Module with Embedding and Linear layers for TP testing.

    Note: While PyTorch's TP API supports sharding Embeddings (ColwiseParallel/
    RowwiseParallel), the Opacus Embedding grad sampler uses scatter_add_ operations
    which don't have DTensor sharding strategies. This is a fundamental limitation:
    scatter operations are incompatible with DTensor sharding.

    This module keeps Embedding replicated and shards only the Linear layer to test
    implicit_replication when gradients flow through unsharded Embedding.
    """

    def __init__(self, vocab_size=100, embedding_dim=32, output_size=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        # Average over sequence dimension
        x = x.mean(dim=1)
        return self.fc(x)


class LSTMModule(nn.Module):
    """Module with DPLSTM layer."""

    def __init__(self, input_size=32, hidden_size=64, output_size=10, num_layers=2):
        super().__init__()
        self.lstm = DPLSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last time step output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class MultiheadAttentionModule(nn.Module):
    """Module with DPMultiheadAttention layer."""

    def __init__(self, embed_dim=64, num_heads=4, output_size=10):
        super().__init__()
        self.attention = DPMultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, output_size, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        # Take mean over sequence dimension
        pooled = attn_output.mean(dim=1)
        return self.fc(pooled)


# ============================================================================
# Test Helper Functions
# ============================================================================


def run_grad_sample_test(rank, world_size, module_class, input_generator, tp_plan):
    """
    Generic test runner for grad sampler with TP.

    Args:
        rank: Process rank
        world_size: Number of processes
        module_class: Module class to test
        input_generator: Function that takes device and returns (input, target) tensors
        tp_plan: Tensor parallelism plan dict for parallelize_module
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create model
    model = module_class().to(device)

    # Apply tensor parallelism
    tp_mesh = init_device_mesh("cuda", (world_size,))
    model = parallelize_module(model, tp_mesh, tp_plan)

    # Create controller
    controller = GradSampleControllerTP(model, batch_first=True, loss_reduction="mean")

    # Generate input and target
    input_data, target = input_generator(device)
    batch_size = input_data.shape[0]

    # Forward pass
    output = model(input_data)

    # Adjust target size to match output (which may be sharded)
    # For sharded outputs, the target should match the local shard size
    if target.shape != output.shape:
        if target.dim() == 1 and target.dtype in [torch.int32, torch.int64]:
            # For classification with sharded output, adjust target class indices
            # to be within the range of the sharded dimension
            num_classes = output.shape[-1]
            target = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            # For regression, reshape target to match output
            target = torch.randn_like(output).to(device)

    # Compute loss
    if target.dim() == 1 and target.dtype in [torch.int32, torch.int64]:
        # Classification loss
        criterion = nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
    else:
        # Regression loss
        criterion = nn.MSELoss(reduction="mean")
        loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Check grad_sample is populated
    grad_sample_found = False
    for param in model.parameters():
        if param.requires_grad:
            assert hasattr(
                param, "grad_sample"
            ), "Parameter should have grad_sample attribute"
            if param.grad_sample is not None:
                grad_sample_found = True
                # Check shape: should be [batch_size, *param.shape]
                assert (
                    param.grad_sample.shape[0] == batch_size
                ), f"grad_sample batch dimension should be {batch_size}, got {param.grad_sample.shape[0]}"

    assert grad_sample_found, "At least one parameter should have grad_sample"

    # Get per-sample norms
    norms = controller.get_per_sample_norms()
    assert norms is not None, "Per-sample norms should not be None"
    assert (
        norms.shape[0] == batch_size
    ), f"Norms should have batch_size {batch_size}, got {norms.shape[0]}"
    assert not torch.isnan(norms).any(), "Norms should not contain NaN"
    assert not torch.isinf(norms).any(), "Norms should not contain Inf"

    cleanup()
    return True


# ============================================================================
# Specific Test Functions for Each Layer Type
# ============================================================================


def run_test_linear_tp(rank, world_size):
    """Test Linear layers with TP."""

    def input_generator(device):
        batch_size = 4
        input_size = 8
        output_size = 8
        input_data = torch.randn(batch_size, input_size).to(device)
        target = torch.randint(0, output_size, (batch_size,)).to(device)
        return input_data, target

    tp_plan = {
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, LinearModule, input_generator, tp_plan
    )


def run_test_conv2d_tp(rank, world_size):
    """Test Conv2d with sharded Linear layer tests implicit_replication.

    Conv layers are not sharded, but the Linear layer IS sharded. This tests that
    gradients can flow correctly through unsharded Conv2d when mixed with sharded Linear.
    """

    def input_generator(device):
        batch_size = 4
        in_channels = 3
        height, width = 16, 16
        input_data = torch.randn(batch_size, in_channels, height, width).to(device)
        # Target size matches Linear output (32)
        target = torch.randn(batch_size, 32).to(device)
        return input_data, target

    # Shard the Linear layer while Conv2d remains replicated
    # This tests implicit_replication when gradients flow through unsharded Conv2d
    tp_plan = {
        "fc": ColwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, Conv2dModule, input_generator, tp_plan
    )


def run_test_layernorm_tp(rank, world_size):
    """Test LayerNorm with sharded Linear layer tests implicit_replication.

    LayerNorm cannot be sharded, but fc2 IS sharded. This tests that gradients
    can flow correctly through unsharded LayerNorm when mixed with sharded Linear.
    Note: fc1 cannot be sharded because LayerNorm expects full feature dimensions.
    """

    def input_generator(device):
        batch_size = 4
        input_size = 16
        output_size = 16
        input_data = torch.randn(batch_size, input_size).to(device)
        # Use classification target
        target = torch.randint(0, output_size, (batch_size,)).to(device)
        return input_data, target

    # Shard fc2 (after LayerNorm) while fc1 and LayerNorm remain replicated
    # This tests implicit_replication when gradients flow through unsharded LayerNorm
    tp_plan = {
        "fc2": ColwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, LayerNormModule, input_generator, tp_plan
    )


def run_test_groupnorm_tp(rank, world_size):
    """Test GroupNorm with sharded Linear layer tests implicit_replication.

    GroupNorm is not sharded, but the Linear layer IS sharded. This tests that
    gradients can flow correctly through unsharded GroupNorm when mixed with sharded Linear.
    """

    def input_generator(device):
        batch_size = 4
        in_channels = 16
        height, width = 8, 8
        input_data = torch.randn(batch_size, in_channels, height, width).to(device)
        # Target size matches Linear output (32)
        target = torch.randn(batch_size, 32).to(device)
        return input_data, target

    # Shard the Linear layer while GroupNorm remains replicated
    # This tests implicit_replication when gradients flow through unsharded GroupNorm
    tp_plan = {
        "fc": ColwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, GroupNormModule, input_generator, tp_plan
    )


def run_test_instancenorm_tp(rank, world_size):
    """Test InstanceNorm with sharded Linear layer tests implicit_replication.

    InstanceNorm is not sharded, but the Linear layer IS sharded. This tests that
    gradients can flow correctly through unsharded InstanceNorm when mixed with sharded Linear.
    """

    def input_generator(device):
        batch_size = 4
        in_channels = 16
        height, width = 8, 8
        input_data = torch.randn(batch_size, in_channels, height, width).to(device)
        # Target size matches Linear output (32)
        target = torch.randn(batch_size, 32).to(device)
        return input_data, target

    # Shard the Linear layer while InstanceNorm remains replicated
    # This tests implicit_replication when gradients flow through unsharded InstanceNorm
    tp_plan = {
        "fc": ColwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, InstanceNormModule, input_generator, tp_plan
    )


def run_test_rmsnorm_tp(rank, world_size):
    """Test RMSNorm with sharded Linear layer tests implicit_replication.

    RMSNorm cannot be sharded, but fc2 IS sharded. This tests that gradients
    can flow correctly through unsharded RMSNorm when mixed with sharded Linear.
    Note: fc1 cannot be sharded because RMSNorm expects full feature dimensions.
    """

    def input_generator(device):
        batch_size = 4
        input_size = 16
        output_size = 16
        input_data = torch.randn(batch_size, input_size).to(device)
        target = torch.randint(0, output_size, (batch_size,)).to(device)
        return input_data, target

    # Shard fc2 (after RMSNorm) while fc1 and RMSNorm remain replicated
    # This tests implicit_replication when gradients flow through unsharded RMSNorm
    tp_plan = {
        "fc2": ColwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, RMSNormModule, input_generator, tp_plan
    )


def run_test_embedding_tp(rank, world_size):
    """Test Embedding with sharded Linear layer tests implicit_replication.

    Note: Embedding layers cannot be sharded due to scatter_add_ operations lacking
    DTensor sharding strategies. This test keeps Embedding replicated and shards the
    Linear layer, testing that gradients flow correctly through unsharded Embedding
    when mixed with sharded Linear.
    """

    def input_generator(device):
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        output_size = 10
        # Generate token indices
        input_data = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        target = torch.randn(batch_size, output_size).to(device)
        return input_data, target

    # Shard the Linear layer while Embedding remains replicated
    # Embedding sharding is blocked by scatter_add_ not having DTensor sharding strategies
    tp_plan = {
        "fc": ColwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, EmbeddingModule, input_generator, tp_plan
    )


def run_test_lstm_tp(rank, world_size):
    """Test DPLSTM with sharded Linear layer tests implicit_replication.

    DPLSTM cannot be easily sharded (complex internal state), but the Linear
    layer IS sharded. This tests that gradients can flow correctly through
    unsharded DPLSTM when mixed with sharded Linear.
    """

    def input_generator(device):
        batch_size = 4
        seq_len = 8
        input_size = 32
        output_size = 10
        # Generate random sequential data
        input_data = torch.randn(batch_size, seq_len, input_size).to(device)
        target = torch.randn(batch_size, output_size).to(device)
        return input_data, target

    # Shard the Linear layer while LSTM remains replicated
    # This tests implicit_replication when gradients flow through unsharded DPLSTM
    tp_plan = {
        "fc": ColwiseParallel(),
    }

    return run_grad_sample_test(rank, world_size, LSTMModule, input_generator, tp_plan)


def run_test_multihead_attention_tp(rank, world_size):
    """Test DPMultiheadAttention with sharded Linear layer tests implicit_replication.

    DPMultiheadAttention cannot be easily sharded (complex Q/K/V projections), but
    the Linear layer IS sharded. This tests that gradients can flow correctly through
    unsharded DPMultiheadAttention when mixed with sharded Linear.
    """

    def input_generator(device):
        batch_size = 4
        seq_len = 8
        embed_dim = 64
        output_size = 10
        # Generate random sequential data
        input_data = torch.randn(batch_size, seq_len, embed_dim).to(device)
        target = torch.randn(batch_size, output_size).to(device)
        return input_data, target

    # Shard the Linear layer while DPMultiheadAttention remains replicated
    # This tests implicit_replication when gradients flow through unsharded attention
    tp_plan = {
        "fc": ColwiseParallel(),
    }

    return run_grad_sample_test(
        rank, world_size, MultiheadAttentionModule, input_generator, tp_plan
    )


# ============================================================================
# Test Cases
# ============================================================================


class GradSamplersTPTest(unittest.TestCase):
    """Test cases for grad samplers with Tensor Parallelism.

    This test suite is organized into two categories:

    1. Real TP tests (test_linear_tp only):
       - Tests actual tensor parallelism with sharded layers
       - Verifies grad_sample computation works with DTensor sharding
       - Linear: ColwiseParallel + RowwiseParallel

    2. TP environment compatibility tests (all other tests):
       - Tests layers that cannot be effectively sharded
       - Verifies implicit_replication() handles mixed Tensor/DTensor operations
       - Ensures these layers work when used alongside TP-sharded layers
       - Note: Embedding cannot be sharded due to scatter_add_ lacking DTensor strategies
    """

    # ========================================================================
    # Real TP Tests - Layers with actual tensor parallelism sharding
    # ========================================================================

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_linear_tp(self):
        """Test Linear layers with real tensor parallelism (ColwiseParallel/RowwiseParallel)."""
        world_size = 2
        mp.spawn(run_test_linear_tp, args=(world_size,), nprocs=world_size, join=True)

    # ========================================================================
    # TP Environment Compatibility Tests - Testing implicit_replication
    # ========================================================================

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_conv2d_tp(self):
        """Test Conv2d grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(run_test_conv2d_tp, args=(world_size,), nprocs=world_size, join=True)

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_embedding_tp(self):
        """Test Embedding grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(
            run_test_embedding_tp, args=(world_size,), nprocs=world_size, join=True
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_layernorm_tp(self):
        """Test LayerNorm grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(
            run_test_layernorm_tp, args=(world_size,), nprocs=world_size, join=True
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_groupnorm_tp(self):
        """Test GroupNorm grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(
            run_test_groupnorm_tp, args=(world_size,), nprocs=world_size, join=True
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_instancenorm_tp(self):
        """Test InstanceNorm grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(
            run_test_instancenorm_tp, args=(world_size,), nprocs=world_size, join=True
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_rmsnorm_tp(self):
        """Test RMSNorm grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(run_test_rmsnorm_tp, args=(world_size,), nprocs=world_size, join=True)

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_lstm_tp(self):
        """Test DPLSTM grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(run_test_lstm_tp, args=(world_size,), nprocs=world_size, join=True)

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 GPUs to run TP tests"
    )
    def test_multihead_attention_tp(self):
        """Test DPMultiheadAttention grad sampler works in TP environment via implicit_replication."""
        world_size = 2
        mp.spawn(
            run_test_multihead_attention_tp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
