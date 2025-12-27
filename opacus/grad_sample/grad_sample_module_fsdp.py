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
from typing import List, Type

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module import GradSampleHooks
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.utils.module_utils import requires_grad, trainable_parameters


logger = logging.getLogger(__name__)


class GradSampleHooksFSDP(GradSampleHooks):
    """
    Hooks-based implementation for computing per-sample gradients with FSDP support.
    Designed to work robustly with LoRA and broadcasted inputs.
    """

    def _get_module_type(self, module: nn.Module) -> Type[nn.Module]:
        """Return the underlying nn.Module type for grad-sampler lookup.

        For composable FSDP2, modules are transformed into subclasses of
        `torch.distributed.fsdp.FSDPModule` (e.g. `FSDPLinear`, `FSDPSequential`).
        Those classes typically inherit from both `FSDPModule` and the original
        module class. We need the original module class to find the registered
        grad-sampler.
        """

        fsdp_mod_cls = None
        try:
            # Composable FSDP2
            from torch.distributed._composable.fsdp import FSDPModule as _FSDPModule

            fsdp_mod_cls = _FSDPModule
        except Exception:
            # Fallback for older/alternate builds
            fsdp_pkg = getattr(getattr(torch, "distributed", None), "fsdp", None)
            fsdp_mod_cls = getattr(fsdp_pkg, "FSDPModule", None)

        if fsdp_mod_cls is not None and isinstance(module, fsdp_mod_cls):
            # Walk MRO to find the first base class that looks like the original
            # module type (e.g. nn.Linear, nn.Sequential).
            for base in type(module).__mro__[1:]:
                if base is fsdp_mod_cls:
                    continue
                if base is nn.Module:
                    continue
                if isinstance(base, type) and issubclass(base, nn.Module):
                    return base

        return type(module)

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

        # Ensure parameters have necessary attributes, handling FSDP replacement.
        # IMPORTANT: initialize attributes for the same parameter set that the parent
        # hook will later update/promote (`recurse` depends on whether we fall back to
        # functorch for this module).
        module_type = self._get_module_type(module)
        recurse = self.force_functorch or module_type not in self.GRAD_SAMPLERS

        for _, p in trainable_parameters(module, recurse=recurse):
            if not hasattr(p, "grad_sample"):
                p.grad_sample = None
            if not hasattr(p, "_forward_counter"):
                p._forward_counter = 0

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

        # Ensure parameters have necessary attributes, handling FSDP replacement.
        # Note: Don't reset _forward_counter if it exists, as it's managed by parent hooks.
        module_type = self._get_module_type(module)
        recurse = self.force_functorch or module_type not in self.GRAD_SAMPLERS

        for _, p in trainable_parameters(module, recurse=recurse):
            if not hasattr(p, "grad_sample"):
                p.grad_sample = None
            if not hasattr(p, "_forward_counter"):
                p._forward_counter = 0

        super().capture_backprops_hook(
            module, _forward_input, forward_output, loss_reduction, batch_first
        )


class GradSampleModuleFSDP(GradSampleHooksFSDP, AbstractGradSampleModule):
    """
    Hooks-based implementation of GradSampleModule with FSDP support.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
    ):
        nn.Module.__init__(self)
        GradSampleHooksFSDP.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )
        self.grad_accumulation_hook = None
