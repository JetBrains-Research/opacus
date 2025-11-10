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
Shared mixin for GradSampleModule and GradSampleController.

This module contains the common hook logic that is shared between the module-wrapping
approach (GradSampleModule) and the controller-based approach (GradSampleController).
"""

import logging
from functools import partial
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN, RNNLinear
from opacus.utils.module_utils import (
    has_trainable_params,
    requires_grad,
    trainable_modules,
    trainable_parameters,
)
from opacus.validators.errors import UnsupportedModuleError
from torch.utils.hooks import RemovableHandle


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleHooksMixin:
    """
    Mixin class containing common hook logic shared between GradSampleModule and GradSampleController.

    This class provides the core functionality for:
    - Adding/removing hooks to compute per-sample gradients
    - Capturing activations during forward pass
    - Computing gradients during backward pass
    - Managing hook lifecycle
    """

    GRAD_SAMPLERS = {}

    def _get_target_module(self) -> nn.Module:
        """Return the module to attach hooks to. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_target_module")

    def iterate_submodules(self, module: nn.Module) -> Iterable[nn.Module]:
        """Iterate over submodules that need hooks attached."""
        if has_trainable_params(module):
            yield module

        # Don't recurse if module is handled by functorch
        if (
            has_trainable_params(module)
            and type(module) not in self.GRAD_SAMPLERS
            and type(module) not in [DPRNN, DPLSTM, DPGRU]
        ):
            return

        for m in module.children():
            yield from self.iterate_submodules(m)

    def _add_hooks_impl(
        self,
        target_module: nn.Module,
        hooks_list: List[RemovableHandle],
        *,
        batch_first: bool,
        loss_reduction: str,
        force_functorch: bool,
    ) -> None:
        """Internal implementation of hook addition."""
        for module in self.iterate_submodules(target_module):
            # Do not add hooks to DPRNN, DPLSTM or DPGRU
            if type(module) in [DPRNN, DPLSTM, DPGRU]:
                continue

            module_type = type(module)
            if force_functorch or not (module_type in self.GRAD_SAMPLERS):
                prepare_layer(module, batch_first=batch_first)

            hooks_list.append(
                module.register_forward_hook(self.capture_activations_hook)
            )

            hooks_list.append(
                module.register_full_backward_hook(
                    partial(
                        self.capture_backprops_hook,
                        loss_reduction=loss_reduction,
                        batch_first=batch_first,
                    )
                )
            )

        self.enable_hooks()

    def disable_hooks(self) -> None:
        """Globally disable all hooks installed by this library."""
        self.hooks_enabled = False

    def enable_hooks(self) -> None:
        """Enable hooks (opposite of disable_hooks)."""
        self.hooks_enabled = True

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        """Hook to capture activations during forward pass."""
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
        ):
            return

        if not self.hooks_enabled:
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])

        for _, p in trainable_parameters(module):
            p._forward_counter += 1

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes per sample gradients given the current backprops and activations
        stored by the associated forward hook.
        """
        # Import here to avoid circular dependency
        from opacus.grad_sample.grad_sample_module import (
            _get_batch_size,
            create_or_accumulate_grad_sample,
            promote_current_grad_sample,
        )

        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()
        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )

        if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
            grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        else:
            grad_sampler_fn = ft_compute_per_sample_gradient

        grad_samples = grad_sampler_fn(module, activations, backprops)
        for param, gs in grad_samples.items():
            create_or_accumulate_grad_sample(
                param=param, grad_sample=gs, max_batch_len=module.max_batch_len
            )

        # Detect end of current batch processing
        for _, p in trainable_parameters(module):
            p._forward_counter -= 1
            if p._forward_counter == 0:
                promote_current_grad_sample(p)

            if not self.grad_accumulation_allowed:
                if isinstance(p.grad_sample, list) and len(p.grad_sample) > 1:
                    raise ValueError(
                        "Poisson sampling is not compatible with grad accumulation. "
                        "You need to call optimizer.step() after every forward/backward pass "
                        "or consider using BatchMemoryManager"
                    )

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

    def rearrange_grad_samples(
        self,
        *,
        module: nn.Module,
        backprops: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rearrange activations and grad_samples based on loss reduction and batch dim."""
        # Import here to avoid circular dependency
        from opacus.grad_sample.grad_sample_module import _get_batch_size

        if not hasattr(module, "activations"):
            raise ValueError(
                f"No activations detected for {type(module)},"
                " run forward after add_hooks(model)"
            )

        batch_dim = 0 if batch_first or type(module) is RNNLinear else 1

        if not hasattr(module, "max_batch_len"):
            module.max_batch_len = _get_batch_size(
                module=module,
                batch_dim=batch_dim,
            )
        activations = module.activations.pop()

        n = module.max_batch_len
        if loss_reduction == "mean":
            backprops = backprops * n
        elif loss_reduction == "sum":
            backprops = backprops
        else:
            raise ValueError(
                f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported"
            )

        # No matter where the batch dimension was, .grad_samples will *always* put it in the first dim
        if batch_dim != 0:
            activations = [
                t.permute([batch_dim] + [x for x in range(t.dim()) if x != batch_dim])
                for t in activations
            ]
            backprops = backprops.permute(
                [batch_dim] + [x for x in range(backprops.dim()) if x != batch_dim]
            )

        return activations, backprops

    def forbid_grad_accumulation(self):
        """Forbid gradient accumulation (for Poisson sampling)."""
        self.grad_accumulation_allowed = False

    def allow_grad_accumulation(self):
        """Allow gradient accumulation."""
        self.grad_accumulation_allowed = True

    @classmethod
    def validate(
        cls, module: nn.Module, *, strict: bool = False
    ) -> List[UnsupportedModuleError]:
        """Check if per sample gradients can be fully computed for a given model."""
        errors = []
        errors.extend(
            [
                UnsupportedModuleError(
                    f"Model contains a trainable layer with buffers "
                    f"that Opacus doesn't currently support ({m_name}:{m}). "
                )
                for m_name, m in trainable_modules(module)
                if len(list(m.buffers())) > 0
            ]
        )
        if strict and len(errors) > 0:
            raise UnsupportedModuleError(errors)
        else:
            return errors


def create_norm_sample(
    *, param: torch.Tensor, grad_sample: torch.Tensor, max_batch_len: int
) -> None:
    """
    Creates a ``_norm_sample`` attribute in the given parameter

    Args:
        param: Parameter to which ``_norm_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        max_batch_len: Maximum batch length for handling empty batches
    """
    if param.requires_grad:
        if (
            max_batch_len == 0
        ):  # To handle the case of empty batch that may arise from Poisson sampling
            param._norm_sample = torch.tensor(
                [], device=grad_sample.device, dtype=grad_sample.dtype
            )
        else:
            param._norm_sample = torch.zeros(
                torch.Size([max_batch_len, 1]),
                device=grad_sample.device,
                dtype=grad_sample.dtype,
            )
            param._norm_sample = grad_sample.reshape(len(grad_sample), -1).norm(
                2, dim=-1
            )


class GradSampleFastGradientClippingMixin(GradSampleHooksMixin):
    """
    Mixin for Fast Gradient and Ghost Clipping support.

    Extends GradSampleHooksMixin to add ghost clipping capabilities for
    memory-efficient gradient norm computation.
    """

    NORM_SAMPLERS = {}

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes per sample gradient norms given the current backprops and activations.

        Supports both:
        - Ghost Clipping: Direct norm computation without materializing full gradients
        - Fast Gradient Clipping: Full gradient computation followed by norm computation
        """
        # Import here to avoid circular dependency
        from opacus.grad_sample.grad_sample_module import (
            create_or_accumulate_grad_sample,
            promote_current_grad_sample,
        )

        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()
        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )

        # Handle DTensor if needed
        activations = [
            temp.to_local() if type(temp) is torch.distributed.tensor.DTensor else temp
            for temp in activations
        ]

        if self.use_ghost_clipping and type(module) in self.NORM_SAMPLERS:
            # Ghost clipping: compute norms directly
            norm_sampler_fn = self.NORM_SAMPLERS[type(module)]
            norm_samples = norm_sampler_fn(module, activations, backprops)

            for param, ns in norm_samples.items():
                if param.requires_grad:
                    param._norm_sample = ns
                    param._forward_counter -= 1

        else:
            # Fast gradient clipping: materialize gradients then compute norms
            if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
                grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
            else:
                grad_sampler_fn = ft_compute_per_sample_gradient

            grad_samples = grad_sampler_fn(module, activations, backprops)
            for param, gs in grad_samples.items():
                create_or_accumulate_grad_sample(
                    param=param, grad_sample=gs, max_batch_len=module.max_batch_len
                )
                # Also create norm sample for fast gradient clipping
                create_norm_sample(
                    param=param, grad_sample=gs, max_batch_len=module.max_batch_len
                )

            # Detect end of current batch processing
            for _, p in trainable_parameters(module):
                p._forward_counter -= 1
                if p._forward_counter == 0:
                    promote_current_grad_sample(p)

                if not self.grad_accumulation_allowed:
                    if isinstance(p.grad_sample, list) and len(p.grad_sample) > 1:
                        raise ValueError(
                            "Poisson sampling is not compatible with grad accumulation. "
                            "You need to call optimizer.step() after every forward/backward pass "
                            "or consider using BatchMemoryManager"
                        )

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len
