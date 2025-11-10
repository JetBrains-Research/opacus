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
GradSampleController: Manages privacy hooks on models without wrapping them.

This module provides a GradSampleModule-less approach to attaching hooks
directly to model parameters for computing per-sample gradients.
"""

import logging
from functools import partial
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
from opacus.grad_sample.grad_sample_hooks_mixin import GradSampleHooksMixin
from opacus.grad_sample.grad_sample_module import (
    _get_batch_size,
    create_or_accumulate_grad_sample,
    promote_current_grad_sample,
)
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


OPACUS_PARAM_MONKEYPATCH_ATTRS = [
    "grad_sample",
    "_forward_counter",
    "_current_grad_sample",
    "_norm_sample",
]


# GradSampleHooksMixin is now imported from grad_sample_hooks_mixin.py to avoid circular imports


class GradSampleController(GradSampleHooksMixin):
    """
    Controller for managing privacy hooks on models without wrapping them

    Computes per-sample gradients using custom-written methods for each layer.
    See README.md for more details

    This class attaches hooks directly to model modules and manages their lifecycle,
    providing an alternative to GradSampleModule wrapping that's more compatible
    with transformers and other complex models.
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
            m: nn.Module to attach hooks to
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
        errors = self.validate(module=m, strict=strict)
        if errors and not strict:
            logger.info(
                f"GradSampleController found the following errors: {errors}."
                "Using non-strict mode, continuing"
            )

        self.module = m
        self.hooks_enabled = False
        self.grad_accumulation_allowed = True
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.force_functorch = force_functorch

        self.autograd_grad_sample_hooks: List[RemovableHandle] = []

        # Initialize parameters with required attributes
        for _, p in trainable_parameters(self.module):
            p.grad_sample = None
            p._forward_counter = 0

        # Add the hooks
        self.add_hooks()

    def _get_target_module(self) -> nn.Module:
        """Return the module to attach hooks to."""
        return self.module

    def add_hooks(self) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.
        """
        self._add_hooks_impl(
            target_module=self.module,
            hooks_list=self.autograd_grad_sample_hooks,
            batch_first=self.batch_first,
            loss_reduction=self.loss_reduction,
            force_functorch=self.force_functorch,
        )

    def remove_hooks(self) -> None:
        """
        Removes hooks added by ``add_hooks()``
        """
        self.disable_hooks()

        while self.autograd_grad_sample_hooks:
            handle = self.autograd_grad_sample_hooks.pop()
            handle.remove()

        # Remove functorch hooks
        for _module_name, module in trainable_modules(self.module):
            if hasattr(module, "ft_compute_sample_grad"):
                delattr(module, "ft_compute_sample_grad")
            if hasattr(module, "activations"):
                delattr(module, "activations")

    def cleanup(self):
        """
        Clean up all hooks and attributes added to the model.
        """
        self.remove_hooks()

        # Clean up parameter attributes
        for attr in OPACUS_PARAM_MONKEYPATCH_ATTRS:
            for p in self.module.parameters():
                if hasattr(p, attr):
                    delattr(p, attr)
