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
Distributed Per-Layer Optimizer with FSDP/TP/CP support.

This module provides a per-layer clipping optimizer that handles DTensor
(sharded) parameters for FSDP, Tensor Parallelism, and Context Parallelism.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.optim import Optimizer

from .optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from .perlayeroptimizer import DPPerLayerOptimizer


def _is_dtensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a DTensor (sharded parameter)."""
    try:
        return type(tensor) is torch.distributed.tensor.DTensor
    except AttributeError:
        return False


def _get_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Get the local tensor from a DTensor, or return as-is."""
    if _is_dtensor(tensor):
        return tensor.to_local()
    return tensor


class DistributedPerLayerOptimizerTP(DPPerLayerOptimizer):
    """
    Per-layer clipping optimizer for distributed training with FSDP/TP/CP support.

    This optimizer implements per-layer gradient clipping where each parameter
    has its own clipping threshold. It handles DTensor (sharded) parameters
    correctly for FSDP, Tensor Parallelism, and Context Parallelism.

    For DTensor (sharded) parameters:
    - All ranks add noise to their local shards
    - Gradients are all-reduced across ranks

    For regular (replicated) parameters:
    - Only rank 0 adds noise
    - Gradients are all-reduced across ranks

    Args:
        optimizer: The wrapped optimizer
        noise_multiplier: The ratio of the standard deviation of the Gaussian noise
            to the L2-sensitivity of the function to which the noise is added
        max_grad_norm: List of maximum L2 norms for per-sample gradients, one per parameter
        expected_batch_size: Expected batch size for scaling
        loss_reduction: How to reduce the loss ("mean" or "sum")
        generator: Random number generator for reproducibility
        secure_mode: If True, uses secure random number generation
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        # Detect if we have sharded parameters (lazy, on first use)
        self._has_sharded_params = None

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )

    def _detect_sharded_params(self) -> bool:
        """Detect if any parameters are DTensors (sharded)."""
        if self._has_sharded_params is None:
            self._has_sharded_params = any(_is_dtensor(p) for p in self.params)
        return self._has_sharded_params

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Override to handle DTensor parameters.

        For DTensor parameters, the grad_sample is computed on the local shard,
        so we need to handle the shape correctly.
        """
        grad_sample = p.grad_sample
        if isinstance(grad_sample, list):
            grad_sample = torch.cat(grad_sample, dim=0)

        # Flatten to [batch_size, -1]
        return grad_sample.reshape(len(grad_sample), -1)

    def add_noise(self):
        """
        Adds noise to clipped gradients with per-layer noise scaling.

        For DTensor (sharded) parameters: All ranks add noise to local shards
        For regular (replicated) parameters: Only rank 0 adds noise
        """
        is_sharded = self._detect_sharded_params()

        for p, max_grad_norm in zip(self.params, self.max_grad_norms):
            _check_processed_flag(p.summed_grad)

            # Calculate per-layer noise std
            noise_std = self.noise_multiplier * max_grad_norm

            if _is_dtensor(p):
                # Sharded parameter - all ranks add noise to local shards
                local_p = _get_local_tensor(p)
                noise = _generate_noise(
                    std=noise_std,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                local_grad = (p.summed_grad + noise).view_as(local_p)

                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                _get_local_tensor(p.grad).copy_(local_grad)
            elif is_sharded or self.rank == 0:
                # Replicated parameter on rank 0, or in sharded mode
                noise = _generate_noise(
                    std=noise_std,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                p.grad = (p.summed_grad + noise).view_as(p)
            else:
                # Replicated parameter on non-zero rank - no noise
                p.grad = p.summed_grad.view_as(p)

            _mark_as_processed(p.summed_grad)

    def reduce_gradients(self):
        """
        All-reduce gradients across ranks.

        Handles both DTensor and regular tensor gradients.
        """
        for p in self.params:
            if not p.requires_grad:
                continue

            grad = p.grad
            if grad is None:
                continue

            if _is_dtensor(grad):
                local_grad = _get_local_tensor(grad)
                torch.distributed.all_reduce(
                    local_grad, op=torch.distributed.ReduceOp.SUM
                )
                if self.loss_reduction == "mean":
                    local_grad /= self.world_size
            else:
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
                if self.loss_reduction == "mean":
                    grad /= self.world_size

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Clips gradients per-layer, adds noise, reduces gradients, and steps.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step()
        return None
