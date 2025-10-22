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
DPOptimizer for Tensor Parallelism with GradSampleControllerTP.

This optimizer extends DPOptimizer to properly handle per-sample gradient norm
computation when using Tensor Parallelism (DTensor-based model parallelism).
"""

from typing import Optional

import torch
from torch.distributed._tensor.experimental import implicit_replication
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from .optimizer import DPOptimizer, _generate_noise, _mark_as_processed


class DPOptimizerTP(DPOptimizer):
    """
    DP Optimizer for Tensor Parallelism.

    Extends DPOptimizer to properly compute per-sample gradient norms when model
    parameters are DTensors. Uses the GradSampleControllerTP's `get_per_sample_norms()`
    method for correct norm aggregation across TP ranks.

    Key differences from DPOptimizer:
    - Uses controller.get_per_sample_norms() for TP-aware norm computation
    - Wraps operations in implicit_replication() context for DTensor compatibility
    - Properly handles merge_flag logic for sharded vs replicated parameters

    Args:
        optimizer: Wrapped optimizer
        controller: GradSampleControllerTP instance for norm computation
        noise_multiplier: Noise multiplier for DP
        max_grad_norm: Maximum gradient norm for clipping
        expected_batch_size: Expected batch size for gradient averaging
        loss_reduction: "mean" or "sum" loss reduction
        generator: Random number generator for noise
        secure_mode: Use secure RNG for noise generation
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        controller,  # GradSampleControllerTP instance
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )

        self.controller = controller

        # For TP, we need world_size for noise scaling
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

        # Disable foreach operations to avoid mixed DTensor/Tensor issues
        # When model has both sharded (DTensor) and replicated (Tensor) parameters,
        # PyTorch's _foreach operations fail with mixed types
        for param_group in self.original_optimizer.param_groups:
            param_group.setdefault("foreach", False)

    def clip_and_accumulate(self):
        """
        Performs gradient clipping with TP-aware norm computation.

        Uses controller.get_per_sample_norms() to get norms that are properly
        aggregated across TP ranks, then clips and accumulates gradients.
        """
        # Use controller's TP-aware norm computation
        try:
            per_sample_norms = self.controller.get_per_sample_norms()
        except RuntimeError as e:
            # Handle case where no grad_samples exist (empty batch, etc.)
            if "No parameters with grad_sample found" in str(e):
                per_sample_norms = torch.zeros((0,), device=next(self.params).device)
            else:
                raise

        # Compute clipping factors
        if len(per_sample_norms) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,), device=next(self.params).device)
        else:
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        # Clip and accumulate gradients for each parameter
        for p in self.params:
            if not hasattr(p, "grad_sample") or p.grad_sample is None:
                continue

            _mark_as_processed(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            # Convert DTensor grad_sample to local for einsum
            # einsum with implicit_replication can handle mixed types, but if grad_sample
            # is a DTensor with sharded dimensions, it may fail during internal reshaping
            if isinstance(grad_sample, DTensor):
                grad_sample = grad_sample.to_local()

            # Cast to parameter dtype (for mixed precision)
            grad_sample = grad_sample.to(p.dtype)
            per_sample_clip_factor_typed = per_sample_clip_factor.to(p.dtype)

            # Apply clipping: sum of (clip_factor * grad_sample)
            grad = torch.einsum("i,i...", per_sample_clip_factor_typed, grad_sample)

            # Accumulate
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            # Don't delete grad_sample yet - accumulated_iterations property needs it
            # It will be cleaned up later by the base optimizer
            # del p.grad_sample
            # p.grad_sample = None

        # Store per_sample_norms for debugging/logging
        self.per_sample_gradient_norms = per_sample_norms

    def add_noise(self):
        """
        Adds noise to clipped gradients.

        Noise is scaled by world_size for proper DP accounting in distributed setting.
        """
        for p in self.params:
            if p.summed_grad is None:
                continue

            # For DTensor summed_grad, use local tensor as reference for noise generation
            # This is necessary because _generate_noise uses shape property which gives global shape
            if isinstance(p.summed_grad, DTensor):
                noise_reference = p.summed_grad.to_local()
            else:
                noise_reference = p.summed_grad

            # Generate noise with local shape
            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=noise_reference,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )

            # Convert noise to DTensor if summed_grad is DTensor
            # Note: implicit_replication doesn't work here because noise has local shape
            # but would be treated as replicated with that shape, causing size mismatch
            if isinstance(p.summed_grad, DTensor):
                noise = DTensor.from_local(
                    noise,
                    device_mesh=p.summed_grad.device_mesh,
                    placements=p.summed_grad.placements,
                    run_check=False,
                )

            # Scale noise by world_size for distributed training
            if self.loss_reduction == "mean" and self.world_size > 1:
                noise = noise / self.world_size

            # Add noise to gradient
            noisy_grad = p.summed_grad + noise

            # Set gradient on parameter
            # IMPORTANT: To avoid mixed DTensor/Tensor issues in optimizer._foreach operations,
            # we always set gradients directly on param.grad, converting DTensors to local tensors
            if isinstance(p, DTensor):
                # Parameter is DTensor: set gradient on _local_tensor
                if isinstance(noisy_grad, DTensor):
                    p._local_tensor.grad = noisy_grad._local_tensor
                else:
                    p._local_tensor.grad = noisy_grad
            else:
                # Regular parameter: convert DTensor gradient to regular tensor if needed
                if isinstance(noisy_grad, DTensor):
                    noisy_grad = noisy_grad.to_local()
                p.grad = noisy_grad

            # Clean up
            p.summed_grad = None

    def pre_step(self, closure=None):
        """
        Perform DP operations before optimizer step.
        """
        return super().pre_step(closure)
