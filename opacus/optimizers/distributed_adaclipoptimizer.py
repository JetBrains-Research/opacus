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
Unified Distributed AdaClip Optimizer for all distributed modes: DDP, FSDP, TP, CP.

This module provides a single optimizer class that handles adaptive clipping
in distributed training scenarios, automatically detecting sharded (DTensor)
vs replicated parameters.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .adaclipoptimizer import AdaClipDPOptimizer
from .optimizer import _check_processed_flag, _generate_noise, _mark_as_processed


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


class DistributedAdaClipDPOptimizer(AdaClipDPOptimizer):
    """
    Unified AdaClipDPOptimizer for all distributed modes: DDP, FSDP, TP, CP.

    This optimizer implements adaptive clipping (https://arxiv.org/pdf/1905.03871.pdf)
    for distributed training. It automatically detects sharded (DTensor) vs
    replicated parameters and applies the appropriate noise strategy:

    - Replicated (DDP): Only rank 0 adds gradient noise
    - Sharded (FSDP/TP): All ranks add noise to their local shards

    The adaptive clipping statistics (sample_size, unclipped_num) are aggregated
    across all ranks before updating the clipping bound, ensuring all ranks
    maintain the same max_grad_norm.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        target_unclipped_quantile: float,
        clipbound_learning_rate: float,
        max_clipbound: float,
        min_clipbound: float,
        unclipped_num_std: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        """
        Args:
            optimizer: The wrapped optimizer
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise
                to the L2-sensitivity of the function to which the noise is added
            target_unclipped_quantile: Target fraction of samples that should remain unclipped
            clipbound_learning_rate: Learning rate for updating the clipping bound
            max_clipbound: Maximum allowed value for the clipping bound
            min_clipbound: Minimum allowed value for the clipping bound
            unclipped_num_std: Standard deviation of noise added to unclipped count for privacy
            max_grad_norm: Initial maximum L2 norm of per-sample gradients
            expected_batch_size: Expected batch size for scaling
            loss_reduction: How to reduce the loss ("mean" or "sum")
            generator: Random number generator for reproducibility
            secure_mode: If True, uses secure random number generation
        """
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            target_unclipped_quantile=target_unclipped_quantile,
            clipbound_learning_rate=clipbound_learning_rate,
            max_clipbound=max_clipbound,
            min_clipbound=min_clipbound,
            unclipped_num_std=unclipped_num_std,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        # Detect if we have sharded parameters (lazy, on first use)
        self._has_sharded_params = None

    def _detect_sharded_params(self) -> bool:
        """Detect if any parameters are DTensors (sharded)."""
        if self._has_sharded_params is None:
            self._has_sharded_params = any(_is_dtensor(p) for p in self.params)
        return self._has_sharded_params

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Handle both regular and DTensor grad_samples.

        For DTensor parameters, the grad_sample is computed on the local shard,
        so we need to handle the shape correctly.
        """
        grad_sample = p.grad_sample
        if isinstance(grad_sample, list):
            grad_sample = torch.cat(grad_sample, dim=0)
        return grad_sample.reshape(len(grad_sample), -1)

    def add_noise(self):
        """
        Add noise with automatic detection of sharded vs replicated params.

        - Sharded (FSDP/TP): All ranks add noise to local shards
        - Replicated (DDP): Only rank 0 adds gradient noise

        Noise is always added to unclipped_num on all ranks for privacy.
        """
        is_sharded = self._detect_sharded_params()

        # For replicated params (DDP), only rank 0 adds gradient noise
        # For sharded params (FSDP/TP), all ranks add noise
        should_add_grad_noise = is_sharded or (self.rank == 0)

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            if _is_dtensor(p):
                # Sharded parameter - add noise to local shard
                local_p = _get_local_tensor(p)
                noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                local_grad = (p.summed_grad + noise).view_as(local_p)
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                _get_local_tensor(p.grad).copy_(local_grad)
            elif should_add_grad_noise:
                # Replicated parameter on rank 0, or sharded mode
                noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                p.grad = (p.summed_grad + noise).view_as(p)
            else:
                # Replicated parameter on non-zero rank - no noise
                p.grad = p.summed_grad.view_as(p)

            _mark_as_processed(p.summed_grad)

        # Add noise to unclipped_num (for privacy, per the paper)
        # This happens on all ranks before aggregation
        device = next(iter(self.params)).device
        unclipped_num_noise = _generate_noise(
            std=self.unclipped_num_std,
            reference=torch.tensor(float(self.unclipped_num), device=device),
            generator=self.generator,
        )
        self.unclipped_num = float(self.unclipped_num) + float(unclipped_num_noise)

    def reduce_gradients(self):
        """All-reduce gradients, handling both DTensor and regular tensors."""
        for p in self.params:
            if not p.requires_grad or p.grad is None:
                continue

            if _is_dtensor(p.grad):
                local_grad = _get_local_tensor(p.grad)
                torch.distributed.all_reduce(
                    local_grad, op=torch.distributed.ReduceOp.SUM
                )
                if self.loss_reduction == "mean":
                    local_grad /= self.world_size
            else:
                torch.distributed.all_reduce(
                    p.grad, op=torch.distributed.ReduceOp.SUM
                )
                if self.loss_reduction == "mean":
                    p.grad /= self.world_size

    def _aggregate_clipping_stats(self):
        """
        Aggregate sample_size and unclipped_num across all ranks.

        This ensures all ranks have the same global statistics for
        computing the new clipping bound.
        """
        device = next(iter(self.params)).device
        stats = torch.tensor(
            [float(self.sample_size), float(self.unclipped_num)],
            dtype=torch.float32,
            device=device,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        self.sample_size = stats[0].item()
        self.unclipped_num = stats[1].item()

    def update_max_grad_norm(self):
        """
        Update clipping bound based on GLOBAL unclipped fraction.

        First aggregates statistics across all ranks, then computes
        the new max_grad_norm (same on all ranks).
        """
        # Aggregate stats across all ranks first
        self._aggregate_clipping_stats()

        # Compute new max_grad_norm (same on all ranks)
        unclipped_frac = self.unclipped_num / self.sample_size
        self.max_grad_norm *= torch.exp(
            torch.tensor(
                -self.clipbound_learning_rate
                * (unclipped_frac - self.target_unclipped_quantile)
            )
        ).item()

        # Clamp to bounds
        if self.max_grad_norm > self.max_clipbound:
            self.max_grad_norm = self.max_clipbound
        elif self.max_grad_norm < self.min_clipbound:
            self.max_grad_norm = self.min_clipbound

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step.

        This method:
        1. Clips per-sample gradients and tracks clipping statistics
        2. Adds noise to gradients and unclipped count
        3. Updates the clipping bound based on global statistics
        4. All-reduces gradients across ranks
        5. Performs the optimizer step
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():  # Calls clip_and_accumulate, add_noise, update_max_grad_norm
            self.reduce_gradients()
            return self.original_optimizer.step()
        return None
