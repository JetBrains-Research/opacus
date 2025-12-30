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

from __future__ import annotations

import logging
import warnings
from typing import List

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module_fsdp import GradSampleHooksFSDP
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.utils.module_utils import requires_grad, trainable_parameters

try:
    from torch.distributed.tensor.experimental import implicit_replication
except ImportError:
    implicit_replication = None


logger = logging.getLogger(__name__)


class GradSampleHooksTP(GradSampleHooksFSDP):
    """
    Hooks-based implementation for computing per-sample gradients with Tensor Parallelism support.

    Inherits from GradSampleHooksFSDP to support combined FSDP + TP (2D parallelism).
    Works with PyTorch's tensor parallelism (parallelize_module, DTensor).
    Designed to work with models parallelized using ColwiseParallel/RowwiseParallel.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        strict: bool = True,
        force_functorch: bool = False,
    ):
        """
        Args:
            m: nn.Module to be attached to
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            strict: If set to True, the input module will be validated to make sure that
                it does not have buffers in all its submodules.
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
        """
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )
        self._analyze_tp_placements()
        warnings.warn(
            "Opacus TP hooks mode is currently in beta. "
            "Custom model output placements may cause unexpected behavior."
        )

    def _is_dtensor(self, param: torch.Tensor) -> bool:
        """Check if a parameter is a DTensor."""
        try:
            return type(param) is torch.distributed.tensor.DTensor
        except AttributeError:
            return False

    def _analyze_tp_placements(self):
        """
        Analyze tensor parallelism state of model parameters to determine
        whether per-sample gradient norms should be merged across devices.

        Sets `_tp_merge_flag` on each trainable parameter:
        - True: Parameter is sharded, norms should be aggregated across ranks
        - False: Parameter is replicated or non-DTensor, only rank 0 contributes

        Specifically, we should NOT merge (merge_flag=False) when:
        1. The parameter is not a DTensor
        2. The model is nn.Embedding with RowWiseParallel (shard on dim 0)
        3. The parameter is replicated (not sharded) - these are auto-frozen

        Replicated parameters are automatically frozen (requires_grad=False) because
        they would cause gradient norms to be double-counted during all_reduce,
        breaking privacy accounting.
        """
        frozen_params = []
        for module in self.iterate_submodules(self._module):
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    if not self._is_dtensor(param):
                        param._tp_merge_flag = False
                    elif type(module) is nn.Embedding and param.placements[0].is_shard(0):
                        # Embedding with RowWiseParallel - don't merge
                        param._tp_merge_flag = False
                    elif param.placements[0].is_replicate():
                        # Replicated parameters would cause double-counting of gradient norms
                        # Auto-freeze them to avoid breaking privacy accounting
                        param.requires_grad = False
                        param._tp_merge_flag = False
                        # Get full parameter name for logging
                        module_name = ""
                        for n, m in self._module.named_modules():
                            if m is module:
                                module_name = n
                                break
                        full_name = f"{module_name}.{name}" if module_name else name
                        frozen_params.append(full_name)
                    else:
                        param._tp_merge_flag = True
        
        if frozen_params:
            warnings.warn(
                f"Replicated parameters are not supported with TP and have been automatically "
                f"frozen (requires_grad=False) to avoid double-counting gradient norms. "
                f"Frozen parameters: {frozen_params}. "
                f"To avoid this warning, either disable bias in linear layers (bias=False) "
                f"or manually freeze these parameters before wrapping with GradSampleHooksTP."
            )

    def _to_local_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert DTensor to local tensor, or return tensor as-is if not a DTensor."""
        if self._is_dtensor(tensor):
            return tensor.to_local()
        return tensor

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
            or not self.hooks_enabled
        ):
            return

        # Ensure parameters have necessary attributes for TP
        module_type = self._get_module_type(module)
        recurse = self.force_functorch or module_type not in self.GRAD_SAMPLERS

        for _, p in trainable_parameters(module, recurse=recurse):
            if not hasattr(p, "grad_sample"):
                p.grad_sample = None
            if not hasattr(p, "_forward_counter"):
                p._forward_counter = 0
            # Ensure _tp_merge_flag exists (may be set during _analyze_tp_placements)
            if not hasattr(p, "_tp_merge_flag"):
                p._tp_merge_flag = self._is_dtensor(p)

        # Convert DTensor inputs to local tensors before storing activations
        # This ensures grad_sample computation uses local shapes matching sharded params
        local_forward_input = [self._to_local_tensor(t) for t in forward_input]

        super().capture_activations_hook(module, local_forward_input, _forward_output)

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        if not self.hooks_enabled:
            return

        # Ensure parameters have necessary attributes for TP
        module_type = self._get_module_type(module)
        recurse = self.force_functorch or module_type not in self.GRAD_SAMPLERS

        for _, p in trainable_parameters(module, recurse=recurse):
            if not hasattr(p, "grad_sample"):
                p.grad_sample = None
            if not hasattr(p, "_forward_counter"):
                p._forward_counter = 0
            if not hasattr(p, "_tp_merge_flag"):
                p._tp_merge_flag = self._is_dtensor(p)

        # Convert DTensor backprops to local tensors
        # forward_output is a tuple where [0] contains the backprops
        if isinstance(forward_output, tuple):
            local_forward_output = tuple(
                self._to_local_tensor(t) if isinstance(t, torch.Tensor) else t
                for t in forward_output
            )
        else:
            local_forward_output = self._to_local_tensor(forward_output)

        super().capture_backprops_hook(
            module, _forward_input, local_forward_output, loss_reduction, batch_first
        )

    def get_per_sample_norms(self) -> torch.Tensor:
        """
        Compute per-sample gradient norms with TP-aware aggregation.

        For sharded parameters (merge_flag=True), norms are aggregated across all ranks.
        For replicated/non-DTensor parameters (merge_flag=False), only rank 0 contributes.

        Returns:
            Tensor of per-sample gradient norms with shape [batch_size]
        """
        if not torch.distributed.is_initialized():
            # Fall back to local computation if not distributed
            return self._compute_local_norms()

        current_rank = torch.distributed.get_rank()
        squared_norms = []

        for _, p in trainable_parameters(self._module):
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                # Compute local squared norm for this parameter
                # grad_sample shape: [batch_size, *param_shape]
                local_norm_sq = p.grad_sample.flatten(1).norm(2, dim=1).square()

                merge_flag = getattr(p, "_tp_merge_flag", False)
                if not merge_flag and current_rank != 0:
                    # Non-sharded param on non-zero rank - contribute zeros
                    squared_norms.append(torch.zeros_like(local_norm_sq))
                else:
                    squared_norms.append(local_norm_sq)

        if not squared_norms:
            raise RuntimeError("No grad_sample found on any trainable parameters")

        # Sum squared norms across parameters
        total_squared_norm = torch.stack(squared_norms, dim=0).sum(dim=0)

        # Aggregate across TP ranks
        torch.distributed.all_reduce(
            total_squared_norm, op=torch.distributed.ReduceOp.SUM
        )

        return total_squared_norm.sqrt()

    def _compute_local_norms(self) -> torch.Tensor:
        """Compute per-sample gradient norms locally (non-distributed fallback)."""
        squared_norms = []

        for _, p in trainable_parameters(self._module):
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                local_norm_sq = p.grad_sample.flatten(1).norm(2, dim=1).square()
                squared_norms.append(local_norm_sq)

        if not squared_norms:
            raise RuntimeError("No grad_sample found on any trainable parameters")

        total_squared_norm = torch.stack(squared_norms, dim=0).sum(dim=0)
        return total_squared_norm.sqrt()


class GradSampleModuleTP(GradSampleHooksTP, AbstractGradSampleModule):
    """
    Hooks-based implementation of GradSampleModule with Tensor Parallelism support.

    This class combines the TP-aware hooks with the AbstractGradSampleModule interface,
    allowing it to be used as a drop-in replacement for GradSampleModule in TP settings.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        strict: bool = True,
        force_functorch: bool = False,
    ):
        nn.Module.__init__(self)
        GradSampleHooksTP.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )
        self.grad_accumulation_hook = None
