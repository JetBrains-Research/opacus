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
from functools import partial
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
from opacus.grad_sample.grad_sample_hooks_mixin import GradSampleHooksMixin
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN, RNNLinear
from opacus.utils.module_utils import (
    has_trainable_params,
    requires_grad,
    trainable_modules,
    trainable_parameters,
)


logger = logging.getLogger(__name__)
logger.disabled = True


def create_or_accumulate_grad_sample(
    *, param: torch.Tensor, grad_sample: torch.Tensor, max_batch_len: int
) -> None:
    """
    Creates a ``_current_grad_sample`` attribute in the given parameter, or adds to it
    if the ``_current_grad_sample`` attribute already exists.


    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        layer: nn.Module parameter belongs to
    """
    if param.requires_grad:
        if hasattr(param, "_current_grad_sample"):
            param._current_grad_sample[: grad_sample.shape[0]] += grad_sample
        else:
            param._current_grad_sample = torch.zeros(
                torch.Size([max_batch_len]) + grad_sample.shape[1:],
                device=grad_sample.device,
                dtype=grad_sample.dtype,
            )
            param._current_grad_sample[: grad_sample.shape[0]] = grad_sample


def promote_current_grad_sample(p: nn.Parameter) -> None:
    if p.requires_grad:
        if p.grad_sample is not None:
            if isinstance(p.grad_sample, list):
                p.grad_sample.append(p._current_grad_sample)
            else:
                p.grad_sample = [p.grad_sample, p._current_grad_sample]
        else:
            p.grad_sample = p._current_grad_sample

        del p._current_grad_sample


class GradSampleModule(AbstractGradSampleModule, GradSampleHooksMixin):
    """
    Hooks-based implementation of AbstractGradSampleModule

    Computes per-sample gradients using custom-written methods for each layer.
    See README.md for more details
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
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
            strict: If set to ``True``, the input module will be validated to make sure that none of its submodules includes buffers,
                which is not currently supported by Opacus.
                If set to ``False``, per sample gradients will
                be computed on "best effort" basis - they will be available where
                possible and set to None otherwise. This is not recommended, because
                some unsupported modules (e.g. BatchNorm) affect other parameters and
                invalidate the concept of per sample gradients for the entire model.
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
        )

        errors = self.validate(module=m, strict=strict)
        if errors and not strict:
            logger.info(
                f"GradSampleModule found the following errors: {errors}."
                "Using non-strict mode, continuing"
            )

        self.hooks_enabled = False
        self.grad_accumulation_allowed = True
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.force_functorch = force_functorch
        self.add_hooks(
            loss_reduction=loss_reduction,
            batch_first=batch_first,
            force_functorch=force_functorch,
        )

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def _get_target_module(self) -> nn.Module:
        """Return the module to attach hooks to."""
        return self._module

    def add_hooks(
        self,
        *,
        loss_reduction: str = "mean",
        batch_first: bool = True,
        force_functorch: bool = False,
    ) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.

        Args:
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            force_functorch: If set to ``True``, will use functorch to compute all per sample gradients.
                Otherwise, functorch will be used only for layers without registered grad sampler methods.
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        # Delegate to mixin implementation
        self._add_hooks_impl(
            target_module=self._module,
            hooks_list=self.autograd_grad_sample_hooks,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            force_functorch=force_functorch,
        )

    def remove_hooks(self) -> None:
        """
        Removes hooks added by ``add_hooks()``
        """
        self.disable_hooks()

        for p in self.parameters():
            if hasattr(p, "ddp_hooks"):
                while p.ddp_hooks:
                    handle = p.ddp_hooks.pop()
                    handle.remove()
                delattr(p, "ddp_hooks")

        if not hasattr(self, "autograd_grad_sample_hooks"):
            raise ValueError("Asked to remove hooks, but no hooks found")
        else:
            while self.autograd_grad_sample_hooks:
                handle = self.autograd_grad_sample_hooks.pop()
                handle.remove()
            delattr(self, "autograd_grad_sample_hooks")
            delattr(self._module, "autograd_grad_sample_hooks")

        # Remove functorch hooks
        for _module_name, module in trainable_modules(self._module):
            if hasattr(module, "ft_compute_sample_grad"):
                delattr(module, "ft_compute_sample_grad")

    def disable_hooks(self) -> None:
        r"""
        Globally disable all hooks installed by this library.
        Why is this needed? As per https://github.com/pytorch/pytorch/issues/25723, there is
        a bug in Autograd that makes removing hooks do nothing if the graph was already
        constructed. For this reason, we have this method to at least turn them off.
        """
        self.hooks_enabled = False

    def enable_hooks(self) -> None:
        r"""
        The opposite of ``disable_hooks()``. Hooks are always enabled unless you explicitly
        disable them so you don't need to call this unless you want to re-enable them.
        """
        self.hooks_enabled = True

    def _close(self):
        super()._close()
        self.remove_hooks()

    # Override base class no-op methods to use mixin implementations
    def forbid_grad_accumulation(self):
        """Forbid gradient accumulation (for Poisson sampling)."""
        GradSampleHooksMixin.forbid_grad_accumulation(self)

    def allow_grad_accumulation(self):
        """Allow gradient accumulation."""
        GradSampleHooksMixin.allow_grad_accumulation(self)

    # Note: The following methods are inherited from GradSampleHooksMixin:
    # - capture_activations_hook
    # - capture_backprops_hook
    # - rearrange_grad_samples
    # - validate


def _get_batch_size(*, module: nn.Module, batch_dim: int) -> int:
    """
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over module.activations, where module.activations is
    a list.

    Args:
        module: input module
        batch_dim: batch dimension

    Returns:
        Maximum sequence length in a batch
    """
    max_batch_len = 0
    for out in module.activations:
        # out is typically a tuple of one element (x)
        # for embedding bag, it is a tuple of two elements (x, offsets)
        # where len(offsets) = batch_size
        if out[-1].shape[batch_dim] > max_batch_len:
            max_batch_len = out[-1].shape[batch_dim]

    return max_batch_len
