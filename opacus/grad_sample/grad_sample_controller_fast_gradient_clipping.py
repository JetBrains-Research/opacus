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
GradSampleControllerFastGradientClipping: Controller-based Fast Gradient and Ghost Clipping.

This module provides a GradSampleModule-less approach with ghost clipping support,
combining the benefits of:
- Controller-based hook management (no model wrapping)
- Ghost clipping (memory-efficient gradient norm computation)
"""

import logging

import torch.nn as nn
from opacus.grad_sample.grad_sample_controller import GradSampleController
from opacus.grad_sample.grad_sample_hooks_mixin import FastGradientHooksHandler
from opacus.utils.module_utils import trainable_parameters


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleControllerFastGradientClipping(
    GradSampleController, FastGradientHooksHandler
):
    """
    Controller for managing privacy hooks with Fast Gradient and Ghost Clipping support

    Extends GradSampleController to add ghost clipping support for memory-efficient
    gradient norm computation. Supports both:
    - Ghost Clipping: Direct norm computation without materializing full gradients
    - Fast Gradient Clipping: Full gradient computation followed by norm computation

    This class attaches hooks directly to model modules and manages their lifecycle,
    providing an alternative to GradSampleModule wrapping that's more compatible
    with transformers and other complex models.
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
            m: nn.Module to attach hooks to
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            max_grad_norm: The value at which gradients are to be clipped.
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
            use_ghost_clipping: If set to ``True``, Ghost Clipping
                will be used for clipping gradients of supported layers. If ``False``, Fast
                Gradient Clipping will be used for all layers.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
        """
        # Call parent constructor
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )

        # Add ghost clipping specific attributes
        self.max_grad_norm = max_grad_norm
        self.use_ghost_clipping = use_ghost_clipping
        self._per_sample_gradient_norms = None

        # Initialize _norm_sample attribute for parameters
        for _, p in trainable_parameters(self.module):
            p._norm_sample = None

        self.trainable_parameters = [p for _, p in trainable_parameters(self.module)]

        if logger.isEnabledFor(logging.INFO):
            self.log_module_gradient_sample_mode(
                module=m,
                force_functorch=force_functorch,
                use_ghost_clipping=use_ghost_clipping,
            )
