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

import logging
from abc import ABC, abstractmethod

import torch.nn as nn
from opacus.utils.module_utils import trainable_parameters


logger = logging.getLogger(__name__)


class AbstractGradSampleHooks(ABC):
    """
    Abstract base class for managing grad sample computation via hooks.

    Defines the interface for:
    - Clearing/deleting grad sample attributes
    - Managing gradient accumulation
    - Cleanup
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        **kwargs,
    ):
        """
        Initialize hooks with the module and configuration.

        Args:
            m: nn.Module to attach hooks to
            batch_first: Flag to indicate if the input tensor has batch as first dimension
            loss_reduction: Indicates if the loss reduction is "sum" or "mean"
            **kwargs: Extra arguments
        """
        self._module = m
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction

        for _, p in trainable_parameters(self._module):
            self.initialize_parameter(p)
        super().__init__(**kwargs)

    def initialize_parameter(self, p: nn.Parameter):
        """
        Initializes parameters with required attributes.
        Can be overridden by subclasses to add more attributes.

        Args:
            p: nn.Parameter to initialize
        """
        p.grad_sample = None
        p._forward_counter = 0

    def set_grad_sample_to_none(self):
        """
        Sets ``.grad_sample`` and related attributes to None.
        """
        for p in self._module.parameters():
            self._set_param_grad_sample_to_none(p)

    def _set_param_grad_sample_to_none(self, p: nn.Parameter):
        """
        Sets ``.grad_sample`` and related attributes to None for a given parameter.
        Can be overridden by subclasses to handle their specific attributes.

        Args:
            p: nn.Parameter
        """
        if hasattr(p, "grad_sample"):
            p.grad_sample = None
        if hasattr(p, "_forward_counter"):
            p._forward_counter = 0

    def del_grad_sample(self):
        """
        Deletes ``.grad_sample`` and related attributes from all model parameters.
        """
        for p in self._module.parameters():
            self._del_param_grad_sample(p)

    def _del_param_grad_sample(self, p: nn.Parameter):
        """
        Deletes ``.grad_sample`` and related attributes from a given parameter.
        Can be overridden by subclasses to handle their specific attributes.

        Args:
            p: nn.Parameter
        """
        if hasattr(p, "grad_sample"):
            delattr(p, "grad_sample")
        if hasattr(p, "_forward_counter"):
            delattr(p, "_forward_counter")

    def forbid_grad_accumulation(self):
        """
        Forbid gradient accumulation (multiple backward passes without zero_grad).
        """
        pass

    def allow_grad_accumulation(self):
        """
        Allow gradient accumulation.
        """
        pass

    def cleanup(self):
        """
        Remove all hooks and clean up all attributes.
        """
        self.del_grad_sample()


class AbstractGradSampleModule(nn.Module, AbstractGradSampleHooks, ABC):
    r"""
    Lightweight nn.Module wrapper that delegates to AbstractGradSampleHooks.

    This class provides the nn.Module interface (forward, zero_grad, parameters, etc.)
    while delegating all grad sample management to the hooks implementation.

    The hooks implementation (AbstractGradSampleHooks) owns the actual module and
    manages all grad sample attributes.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        **kwargs,
    ):
        """
        Initialize the nn.Module wrapper.

        Args:
            m: nn.Module to wrap
            batch_first: Flag to indicate if the input tensor has batch as first dimension
            loss_reduction: Indicates if the loss reduction is "sum" or "mean"
            **kwargs: Extra arguments passed to AbstractGradSampleHooks
        """
        nn.Module.__init__(self)
        AbstractGradSampleHooks.__init__(
            self, m, batch_first=batch_first, loss_reduction=loss_reduction, **kwargs
        )

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass. Should delegate to self._module which is provided
        by the hooks implementation.
        """
        pass

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as e:
            if hasattr(self, "_module"):
                submodules = dict(self._module.named_modules())
                if item and item in submodules:
                    return submodules[item]
            raise e

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad`` and ``p.grad_sample`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` is
            never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """
        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample to None due to "
                "non-trivial gradient accumulation behaviour"
            )
        self.set_grad_sample_to_none()
        super().zero_grad(set_to_none)

    def to_standard_module(self) -> nn.Module:
        """
        Returns the standard nn.Module wrapped by this, eliminating all traces
        of grad samples and hooks

        Returns:
            The wrapped module
        """
        self._close()
        return self._module

    def _close(self):
        """Clean up by calling hooks cleanup method."""
        self.cleanup()

    def __repr__(self):
        return f"{type(self).__name__}({self._module.__repr__()})"
