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

import torch.nn as nn
from opacus.grad_sample.grad_sample_hooks_mixin import FastGradientHooksHandler
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.utils.module_utils import trainable_parameters


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleModuleFastGradientClipping(GradSampleModule, FastGradientHooksHandler):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping

    Computes norms of gradients without gradient instantiation
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

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
        """
        if logger.isEnabledFor(logging.INFO):
            self.log_module_gradient_sample_mode(
                module=m,
                force_functorch=force_functorch,
                use_ghost_clipping=use_ghost_clipping,
            )

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )
        self.trainable_parameters = [p for _, p in trainable_parameters(self._module)]
        self.max_grad_norm = max_grad_norm
        self.use_ghost_clipping = use_ghost_clipping
        self._per_sample_gradient_norms = None
