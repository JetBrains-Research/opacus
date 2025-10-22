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

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from torch.distributed._tensor.experimental import implicit_replication
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from .optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from .optimizer_tp import DPOptimizerTP


logger = logging.getLogger(__name__)


class AdaClipDPOptimizerTP(DPOptimizerTP):
    """
    DP Optimizer for Tensor Parallelism with Adaptive Clipping.

    Extends DPOptimizerTP to implement adaptive clipping strategy from
    https://arxiv.org/pdf/1905.03871.pdf

    The key difference from AdaClipDPOptimizer is that this version uses
    controller.get_per_sample_norms() for TP-aware norm computation instead
    of computing norms directly from grad_samples.

    Args:
        optimizer: Wrapped optimizer
        controller: GradSampleControllerTP instance for norm computation
        noise_multiplier: Noise multiplier for DP
        target_unclipped_quantile: Target fraction of unclipped gradients
        clipbound_learning_rate: Learning rate for clipbound updates
        max_clipbound: Maximum clipping bound
        min_clipbound: Minimum clipping bound
        unclipped_num_std: Std for noise added to unclipped count
        max_grad_norm: Initial maximum gradient norm for clipping
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
        super().__init__(
            optimizer=optimizer,
            controller=controller,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )

        if max_clipbound <= min_clipbound:
            raise ValueError("max_clipbound must be larger than min_clipbound.")
        if noise_multiplier >= 2 * unclipped_num_std:
            raise ValueError(
                "noise_multiplier must be smaller than 2 * unclipped_num_std. "
                "This is a requirement stemming from Theorem 1 in "
                "https://arxiv.org/pdf/1905.03871.pdf"
            )

        self.target_unclipped_quantile = target_unclipped_quantile
        self.clipbound_learning_rate = clipbound_learning_rate
        self.max_clipbound = max_clipbound
        self.min_clipbound = min_clipbound
        self.unclipped_num_std = unclipped_num_std

        # Theorem 1. in https://arxiv.org/pdf/1905.03871.pdf
        if self.noise_multiplier > 0:  # if noise_multiplier = 0 then it stays zero
            self.noise_multiplier = (
                self.noise_multiplier ** (-2) - (2 * unclipped_num_std) ** (-2)
            ) ** (-1 / 2)

        self.sample_size = 0
        self.unclipped_num = 0

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """
        super().zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):
        """
        Performs gradient clipping with TP-aware norm computation and adaptive clipping.

        Uses controller.get_per_sample_norms() to get norms that are properly
        aggregated across TP ranks, then clips gradients and tracks unclipped fraction.
        """
        # Get per-sample norms from controller (TP-aware)
        per_sample_norms = self.controller.get_per_sample_norms()

        if per_sample_norms is None:
            raise ValueError("controller.get_per_sample_norms() returned None")

        with implicit_replication():
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

            # Track sample size and unclipped count for adaptive clipping
            self.sample_size += len(per_sample_clip_factor)
            self.unclipped_num += (
                len(per_sample_clip_factor) - (per_sample_clip_factor < 1).sum()
            )

            # Clip and accumulate gradients
            for p in self.params:
                _check_processed_flag(p.grad_sample)

                # Handle both DTensor and regular Tensor grad_samples
                grad_sample = self._get_flat_grad_sample(p)

                # Convert DTensor to local for einsum (same as parent class)
                # einsum with implicit_replication can handle mixed types, but if grad_sample
                # is a DTensor with sharded dimensions, it may fail during internal reshaping
                if isinstance(grad_sample, DTensor):
                    grad_sample = grad_sample.to_local()

                # Cast to parameter dtype (for mixed precision)
                grad_sample = grad_sample.to(p.dtype)
                per_sample_clip_factor_typed = per_sample_clip_factor.to(p.dtype)

                grad = torch.einsum("i,i...", per_sample_clip_factor_typed, grad_sample)

                if p.summed_grad is not None:
                    p.summed_grad += grad
                else:
                    p.summed_grad = grad

                _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Add noise to gradients and to unclipped count.
        """
        super().add_noise()

        # Add noise to unclipped count for privacy
        unclipped_num_noise = _generate_noise(
            std=self.unclipped_num_std,
            reference=self.unclipped_num.float(),
            generator=self.generator,
        )

        self.unclipped_num = float(self.unclipped_num)
        self.unclipped_num += unclipped_num_noise

    def update_max_grad_norm(self):
        """
        Update clipping bound based on unclipped fraction.
        """
        with implicit_replication():
            unclipped_frac = self.unclipped_num / self.sample_size
            self.max_grad_norm *= torch.exp(
                -self.clipbound_learning_rate
                * (unclipped_frac - self.target_unclipped_quantile)
            )
            if self.max_grad_norm > self.max_clipbound:
                self.max_grad_norm = self.max_clipbound
            elif self.max_grad_norm < self.min_clipbound:
                self.max_grad_norm = self.min_clipbound

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform pre-step operations including clipbound update.
        """
        pre_step_result = super().pre_step()
        if pre_step_result:
            self.update_max_grad_norm()
        return pre_step_result
