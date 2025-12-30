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
from typing import List, Optional

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module_fsdp import GradSampleHooksFSDP
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.utils.module_utils import trainable_parameters


logger = logging.getLogger(__name__)


class GradSampleHooksCP(GradSampleHooksFSDP):
    """
    Hooks-based implementation for computing per-sample gradients with Context Parallelism support.

    Inherits from GradSampleHooksFSDP since CP requires FSDP2.
    
    Context Parallelism (CP) splits the sequence dimension across devices using ring attention.
    Unlike Tensor Parallelism, CP does NOT use DTensors for parameters - parameters remain
    regular tensors. However, per-sample gradient norms must be aggregated across CP ranks
    because each rank only processes a portion of the sequence.
    
    Key differences from TP:
    - Parameters are NOT DTensors (no sharding of weights)
    - Activations/inputs are split along sequence dimension
    - Gradient norms need all-reduce across CP dimension
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
            cp_group: The process group for context parallelism. If None, will try to
                detect from the accelerator state or use the default process group.

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
        self.cp_group = cp_group
        self._cp_world_size = None
        self._cp_rank = None
        self._initialize_cp_info()
        
        warnings.warn(
            "Opacus CP (Context Parallelism) hooks mode is currently in beta. "
            "Ensure your model is compatible with context parallelism."
        )

    def _initialize_cp_info(self):
        """Initialize CP rank and world size information."""
        if self.cp_group is not None:
            self._cp_world_size = torch.distributed.get_world_size(self.cp_group)
            self._cp_rank = torch.distributed.get_rank(self.cp_group)
        elif torch.distributed.is_initialized():
            # Will be set later when CP group is available
            self._cp_world_size = None
            self._cp_rank = None

    def set_cp_group(self, cp_group: torch.distributed.ProcessGroup):
        """
        Set the context parallelism process group.
        
        This can be called after initialization if the CP group wasn't available
        at construction time.
        
        Args:
            cp_group: The process group for context parallelism.
        """
        self.cp_group = cp_group
        self._cp_world_size = torch.distributed.get_world_size(cp_group)
        self._cp_rank = torch.distributed.get_rank(cp_group)

    def get_per_sample_norms(self) -> torch.Tensor:
        """
        Compute per-sample gradient norms with CP-aware aggregation.

        In Context Parallelism, each rank processes a portion of the sequence.
        The per-sample gradient norms must be aggregated (summed) across all
        CP ranks to get the correct total norm for each sample.

        Returns:
            Tensor of per-sample gradient norms with shape [batch_size]
        """
        # First compute local norms
        squared_norms = []
        
        for _, p in trainable_parameters(self._module):
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                # grad_sample shape: [batch_size, *param_shape]
                local_norm_sq = p.grad_sample.flatten(1).norm(2, dim=1).square()
                squared_norms.append(local_norm_sq)

        if not squared_norms:
            raise RuntimeError("No grad_sample found on any trainable parameters")

        # Sum squared norms across parameters (local computation)
        total_squared_norm = torch.stack(squared_norms, dim=0).sum(dim=0)

        # Aggregate across CP ranks if CP is enabled
        if self._should_aggregate_across_cp():
            torch.distributed.all_reduce(
                total_squared_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=self.cp_group,
            )

        return total_squared_norm.sqrt()

    def _should_aggregate_across_cp(self) -> bool:
        """Check if we should aggregate gradient norms across CP ranks."""
        if not torch.distributed.is_initialized():
            return False
        
        if self.cp_group is not None:
            return self._cp_world_size is not None and self._cp_world_size > 1
        
        # If no explicit CP group, check if we can detect it from accelerator
        return False


class GradSampleModuleCP(GradSampleHooksCP, AbstractGradSampleModule):
    """
    Hooks-based implementation of GradSampleModule with Context Parallelism support.

    This class combines the CP-aware hooks with the AbstractGradSampleModule interface,
    allowing it to be used as a drop-in replacement for GradSampleModule in CP settings.
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
        GradSampleHooksCP.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
            cp_group=cp_group,
        )
        self.grad_accumulation_hook = None
