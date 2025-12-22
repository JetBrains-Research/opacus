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
from typing import List

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module import GradSampleHooks, GradSampleModule
from opacus.utils.module_utils import (
    has_trainable_params,
    requires_grad,
    trainable_parameters,
)


logger = logging.getLogger(__name__)
logger.disabled = True


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


class FastGradientHooks(GradSampleHooks):
    """
    Hooks for Fast Gradient and Ghost Clipping support.

    Extends GradSampleHooks to add ghost clipping capabilities for
    memory-efficient gradient norm computation.
    """

    NORM_SAMPLERS = {}

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
        max_grad_norm=1,
        use_ghost_clipping=True,
        **kwargs,
    ):
        super().__init__(
            m=m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
            **kwargs,
        )

        self.max_grad_norm = max_grad_norm
        self.use_ghost_clipping = use_ghost_clipping
        self._per_sample_gradient_norms = None

        self.trainable_parameters = [p for _, p in trainable_parameters(self._module)]

        if logger.isEnabledFor(logging.INFO):
            self.log_module_gradient_sample_mode(
                module=m,
                force_functorch=force_functorch,
                use_ghost_clipping=use_ghost_clipping,
            )

    def initialize_parameter(self, p: nn.Parameter):
        super().initialize_parameter(p)
        p._norm_sample = None

    def _set_param_grad_sample_to_none(self, p: nn.Parameter):
        super()._set_param_grad_sample_to_none(p)
        if hasattr(p, "_norm_sample"):
            p._norm_sample = None

    def _del_param_grad_sample(self, p: nn.Parameter):
        super()._del_param_grad_sample(p)
        if hasattr(p, "_norm_sample"):
            delattr(p, "_norm_sample")

    def get_clipping_coef(self) -> torch.Tensor:
        """Get clipping coefficient for ghost clipping."""
        norm_sample = self.get_norm_sample()
        return (self.max_grad_norm / (norm_sample + 1e-6)).clamp(max=1.0)

    def get_norm_sample(self) -> torch.Tensor:
        """Get per-example gradient norms."""
        squared_norm_sample = (
            torch.stack(
                [param._norm_sample for param in self.trainable_parameters],
                dim=0,
            )
            .norm(2, dim=0)
            .square()
        )

        return squared_norm_sample.sqrt()

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        """Hook to capture activations and check for parameter tying in ghost clipping."""
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
            if (
                self.use_ghost_clipping
                and p._forward_counter > 1
                and type(module) in self.NORM_SAMPLERS
            ):
                raise NotImplementedError(
                    "Parameter tying is not supported with Ghost Clipping"
                )

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

        self.compute_sample_gradients(module, activations, backprops)

    def compute_sample_gradients(self, module, activations, backprops):
        if self.use_ghost_clipping and type(module) in self.NORM_SAMPLERS:
            self._compute_ghost_grad_sample_norms(module, activations, backprops)
            self._on_gradients_computed(module)
        else:
            super().compute_sample_gradients(module, activations, backprops)

    def _process_grad_sample(
        self, param: nn.Parameter, grad_sample: torch.Tensor, max_batch_len: int
    ):
        super()._process_grad_sample(param, grad_sample, max_batch_len)
        # Also create norm sample for fast gradient clipping
        create_norm_sample(
            param=param, grad_sample=grad_sample, max_batch_len=max_batch_len
        )

    def _compute_ghost_grad_sample_norms(self, module, activations, backprops):
        # Ghost clipping: compute norms directly
        norm_sampler_fn = self.NORM_SAMPLERS[type(module)]
        norm_samples = norm_sampler_fn(module, activations, backprops)

        for param, ns in norm_samples.items():
            if param.requires_grad:
                param._norm_sample = ns

    def log_module_gradient_sample_mode(
        self, module: nn.Module, *, force_functorch=False, use_ghost_clipping=True
    ):
        """
        Check if the module is compatible with the requested gradient sample mode.
        """
        # Do not add hooks to DPRNN, DPLSTM or DPGRU
        from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN

        if type(module) in [DPRNN, DPLSTM, DPGRU]:
            return

        module_type = type(module)
        if use_ghost_clipping and module_type in self.NORM_SAMPLERS:
            # Ghost clipping: module has a registered norm sampler
            pass
        elif not force_functorch and module_type in self.GRAD_SAMPLERS:
            # Fast gradient clipping: module has a registered grad sampler
            pass
        elif force_functorch or (
            has_trainable_params(module) and module_type not in self.GRAD_SAMPLERS
        ):
            # Functorch will be used
            pass

    @property
    def per_sample_gradient_norms(self) -> torch.Tensor:
        """Get per-example gradient norms."""
        if not hasattr(self, "_per_sample_gradient_norms"):
            self._per_sample_gradient_norms = self.get_norm_sample()
        return self._per_sample_gradient_norms

    @per_sample_gradient_norms.setter
    def per_sample_gradient_norms(self, value: torch.Tensor):
        self._per_sample_gradient_norms = value


class GradSampleModuleFastGradientClipping(FastGradientHooks, GradSampleModule):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping

    Computes norms of gradients without gradient instantiation
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
        max_grad_norm=1,
        use_ghost_clipping=True,
        **kwargs,
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            max_grad_norm: The value at which gradients are to be clipped.
            strict: If set to True, the input module will be validated to make sure that
                it does not have buffers in all its submodules.
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.
            use_ghost_clipping: If set to ``True``, Ghost Clipping
                will be used for clipping gradients of supported layers. If ``False``, Fast
                Gradient Clipping will be used for all layers.
            **kwargs: Additional keyword arguments passed to the parent class

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
        """
        super().__init__(
            m=m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=use_ghost_clipping,
            **kwargs,
        )
