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
Unified FSDP grad sample module with automatic TP and optional CP support.

This module provides a single class that handles all distributed parallelism modes:
- FSDP (Fully Sharded Data Parallel) - base functionality
- TP (Tensor Parallelism) - auto-detected via DTensor parameters
- CP (Context Parallelism) - optional via cp_group parameter
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Type

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module import GradSampleHooks
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.utils.module_utils import requires_grad, trainable_parameters


logger = logging.getLogger(__name__)


class GradSampleHooksFSDP(GradSampleHooks):
    """
    Hooks-based implementation for computing per-sample gradients with FSDP support.
    
    This class automatically handles:
    - FSDP (Fully Sharded Data Parallel) - module type unwrapping, parameter attribute management
    - TP (Tensor Parallelism) - auto-detected when parameters are DTensors
    - CP (Context Parallelism) - enabled when cp_group is provided
    
    TP is automatically detected by checking if model parameters are DTensor instances.
    When DTensors are detected:
    - Sharded parameters have their gradient norms aggregated across TP ranks
    - Replicated parameters are auto-frozen to avoid double-counting gradient norms
    - DTensor activations/backprops are converted to local tensors for grad_sample computation
    
    CP support is enabled by passing a cp_group parameter. When enabled, gradient norms
    are aggregated across CP ranks after local computation.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        strict: bool = True,
        force_functorch: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """
        Args:
            m: nn.Module to be attached to
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch.
            loss_reduction: Indicates if the loss reduction is a sum or mean operation.
            strict: If True, validates that module doesn't have buffers.
            force_functorch: If True, uses functorch for all per-sample gradients.
            cp_group: Optional process group for Context Parallelism. If provided,
                gradient norms will be aggregated across CP ranks.
        """
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )
        
        # CP configuration
        self.cp_group = cp_group
        self._cp_world_size = None
        self._cp_rank = None
        if cp_group is not None:
            self._initialize_cp_info()
        
        # TP configuration - will be set lazily on first use
        self._tp_analyzed = False
        self._has_dtensor_params = False

    # =========================================================================
    # FSDP Support
    # =========================================================================

    def _get_module_type(self, module: nn.Module) -> Type[nn.Module]:
        """Return the underlying nn.Module type for grad-sampler lookup.

        For composable FSDP2, modules are transformed into subclasses of
        `torch.distributed.fsdp.FSDPModule` (e.g. `FSDPLinear`, `FSDPSequential`).
        We need the original module class to find the registered grad-sampler.
        """
        fsdp_mod_cls = None
        try:
            from torch.distributed._composable.fsdp import FSDPModule as _FSDPModule
            fsdp_mod_cls = _FSDPModule
        except Exception:
            fsdp_pkg = getattr(getattr(torch, "distributed", None), "fsdp", None)
            fsdp_mod_cls = getattr(fsdp_pkg, "FSDPModule", None)

        if fsdp_mod_cls is not None and isinstance(module, fsdp_mod_cls):
            for base in type(module).__mro__[1:]:
                if base is fsdp_mod_cls:
                    continue
                if base is nn.Module:
                    continue
                if isinstance(base, type) and issubclass(base, nn.Module):
                    return base

        return type(module)

    # =========================================================================
    # Tensor Parallelism (TP) Support - Auto-detected
    # =========================================================================

    def _is_dtensor(self, tensor: torch.Tensor) -> bool:
        """Check if a tensor is a DTensor."""
        try:
            return type(tensor) is torch.distributed.tensor.DTensor
        except AttributeError:
            return False

    def _analyze_tp_placements(self):
        """
        Analyze tensor parallelism state of model parameters.
        Called lazily on first forward pass.
        
        Sets `_tp_merge_flag` on each trainable parameter:
        - True: Parameter is sharded, norms should be aggregated across ranks
        - False: Parameter is replicated or non-DTensor
        
        Replicated parameters are auto-frozen to avoid double-counting gradient norms.
        """
        if self._tp_analyzed:
            return
        
        self._tp_analyzed = True
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
                        # Replicated parameters would cause double-counting
                        param.requires_grad = False
                        param._tp_merge_flag = False
                        module_name = ""
                        for n, m in self._module.named_modules():
                            if m is module:
                                module_name = n
                                break
                        full_name = f"{module_name}.{name}" if module_name else name
                        frozen_params.append(full_name)
                    else:
                        param._tp_merge_flag = True
                        self._has_dtensor_params = True
        
        if frozen_params:
            warnings.warn(
                f"Replicated DTensor parameters are not supported and have been automatically "
                f"frozen (requires_grad=False) to avoid double-counting gradient norms. "
                f"Frozen parameters: {frozen_params}. "
                f"To avoid this warning, disable bias in linear layers (bias=False) "
                f"or manually freeze these parameters before wrapping."
            )
        
        if self._has_dtensor_params:
            logger.info("TP (Tensor Parallelism) auto-detected via DTensor parameters")

    def _to_local_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert DTensor to local tensor, or return tensor as-is."""
        if self._is_dtensor(tensor):
            return tensor.to_local()
        return tensor

    # =========================================================================
    # Context Parallelism (CP) Support - Optional
    # =========================================================================

    def _initialize_cp_info(self):
        """Initialize CP rank and world size information."""
        if self.cp_group is not None:
            self._cp_world_size = torch.distributed.get_world_size(self.cp_group)
            self._cp_rank = torch.distributed.get_rank(self.cp_group)
            logger.info(f"CP (Context Parallelism) enabled with world_size={self._cp_world_size}")

    def set_cp_group(self, cp_group: torch.distributed.ProcessGroup):
        """
        Set the context parallelism process group after initialization.
        
        Args:
            cp_group: The process group for context parallelism.
        """
        self.cp_group = cp_group
        self._cp_world_size = torch.distributed.get_world_size(cp_group)
        self._cp_rank = torch.distributed.get_rank(cp_group)

    def _should_aggregate_across_cp(self) -> bool:
        """Check if we should aggregate gradient norms across CP ranks."""
        if not torch.distributed.is_initialized():
            return False
        if self.cp_group is not None:
            return self._cp_world_size is not None and self._cp_world_size > 1
        return False

    # =========================================================================
    # Hook Overrides
    # =========================================================================

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

        # Lazy TP analysis on first forward
        if not self._tp_analyzed:
            self._analyze_tp_placements()

        # Ensure parameters have necessary attributes
        module_type = self._get_module_type(module)
        recurse = self.force_functorch or module_type not in self.GRAD_SAMPLERS

        for _, p in trainable_parameters(module, recurse=recurse):
            if not hasattr(p, "grad_sample"):
                p.grad_sample = None
            if not hasattr(p, "_forward_counter"):
                p._forward_counter = 0
            if not hasattr(p, "_tp_merge_flag"):
                p._tp_merge_flag = self._is_dtensor(p)

        # Convert DTensor inputs to local tensors for grad_sample computation
        if self._has_dtensor_params:
            forward_input = [self._to_local_tensor(t) for t in forward_input]

        super().capture_activations_hook(module, forward_input, _forward_output)

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

        # Ensure parameters have necessary attributes
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
        if self._has_dtensor_params:
            if isinstance(forward_output, tuple):
                forward_output = tuple(
                    self._to_local_tensor(t) if isinstance(t, torch.Tensor) else t
                    for t in forward_output
                )
            else:
                forward_output = self._to_local_tensor(forward_output)

        super().capture_backprops_hook(
            module, _forward_input, forward_output, loss_reduction, batch_first
        )

    # =========================================================================
    # Per-Sample Norm Computation (unified for TP and CP)
    # =========================================================================

    def get_per_sample_norms(self) -> torch.Tensor:
        """
        Compute per-sample gradient norms with TP and/or CP-aware aggregation.

        This method handles all combinations:
        - FSDP only: Local norm computation
        - TP: Aggregate across TP ranks based on _tp_merge_flag (auto-detected)
        - CP: Aggregate across CP group (if cp_group provided)
        - TP+CP: Both aggregations

        Returns:
            Tensor of per-sample gradient norms with shape [batch_size]
        """
        if not torch.distributed.is_initialized():
            return self._compute_local_norms()

        current_rank = torch.distributed.get_rank()
        squared_norms = []

        for _, p in trainable_parameters(self._module):
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                local_norm_sq = p.grad_sample.flatten(1).norm(2, dim=1).square()

                # For TP: handle merge flag
                merge_flag = getattr(p, "_tp_merge_flag", False)
                if merge_flag is False and self._has_dtensor_params and current_rank != 0:
                    # Non-sharded param on non-zero rank - contribute zeros
                    squared_norms.append(torch.zeros_like(local_norm_sq))
                else:
                    squared_norms.append(local_norm_sq)

        if not squared_norms:
            raise RuntimeError("No grad_sample found on any trainable parameters")

        total_squared_norm = torch.stack(squared_norms, dim=0).sum(dim=0)

        # Aggregate across TP ranks (if TP detected)
        if self._has_dtensor_params:
            torch.distributed.all_reduce(
                total_squared_norm, op=torch.distributed.ReduceOp.SUM
            )

        # Aggregate across CP ranks (if CP enabled)
        if self._should_aggregate_across_cp():
            torch.distributed.all_reduce(
                total_squared_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=self.cp_group,
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


class GradSampleModuleFSDP(GradSampleHooksFSDP, AbstractGradSampleModule):
    """
    Hooks-based implementation of GradSampleModule with FSDP, TP, and CP support.
    
    TP is auto-detected via DTensor parameters. CP is enabled via cp_group parameter.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        strict: bool = True,
        force_functorch: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        nn.Module.__init__(self)
        GradSampleHooksFSDP.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
            cp_group=cp_group,
        )
        self.grad_accumulation_hook = None
