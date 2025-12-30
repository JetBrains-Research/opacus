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

from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .ddpoptimizer import DistributedDPOptimizer
from .optimizer import _check_processed_flag, _generate_noise, _mark_as_processed


def _is_dtensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a DTensor."""
    try:
        return type(tensor) is torch.distributed.tensor.DTensor
    except AttributeError:
        return False


def _get_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Get the local tensor from a DTensor, or return the tensor as-is."""
    if _is_dtensor(tensor):
        return tensor.to_local()
    return tensor


class DistributedDPOptimizerTP(DistributedDPOptimizer):
    """
    :class:`~opacus.optimizers.ddpoptimizer.DistributedDPOptimizer` compatible with
    Tensor Parallelism (TP).
    
    This optimizer handles DTensor parameters correctly by using local tensor shapes
    for grad_sample operations while maintaining proper distributed gradient aggregation.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
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

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Override to handle DTensor parameters.
        
        For DTensor parameters, the grad_sample is computed on the local shard,
        so we need to handle the shape correctly.
        """
        # Get grad_sample - it may be a list if accumulated over multiple batches
        grad_sample = p.grad_sample
        if isinstance(grad_sample, list):
            grad_sample = torch.cat(grad_sample, dim=0)
        
        # Flatten to [batch_size, -1]
        return grad_sample.reshape(len(grad_sample), -1)

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``.
        
        Override to handle DTensor parameters by using local tensor shape for view_as.
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            # For DTensor parameters, use the local tensor as reference for shape
            # The summed_grad is computed from local grad_sample, so it has local shape
            if _is_dtensor(p):
                local_p = _get_local_tensor(p)
                noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                # View as local tensor shape
                local_grad = (p.summed_grad + noise).view_as(local_p)
                # For DTensor parameters, p.grad is also a DTensor
                # We need to copy to its local tensor to avoid DTensor/Tensor mixing
                if p.grad is None:
                    # Initialize grad as zeros with same shape as parameter
                    # This will create a DTensor grad for DTensor params
                    p.grad = torch.zeros_like(p)
                # Copy to the local tensor of the DTensor grad
                _get_local_tensor(p.grad).copy_(local_grad)
            else:
                # Non-DTensor parameter - use standard logic
                noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

    def reduce_gradients(self):
        """
        Override to handle DTensor parameters.
        
        For DTensor parameters, the gradient is stored in the local tensor.
        We need to all_reduce the local tensor across TP ranks.
        """
        for p in self.params:
            if not p.requires_grad:
                continue
            
            grad = p.grad
            if grad is None:
                continue
            
            # For DTensor parameters, all_reduce the local tensor
            # DTensor's all_reduce requires DeviceMesh, so we use the local tensor
            if _is_dtensor(grad):
                local_grad = _get_local_tensor(grad)
                torch.distributed.all_reduce(local_grad, op=torch.distributed.ReduceOp.SUM)
                if self.loss_reduction == "mean":
                    local_grad /= self.world_size
            else:
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
                if self.loss_reduction == "mean":
                    grad /= self.world_size
