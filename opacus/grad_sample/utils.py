# !/usr/bin/env python3
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

from typing import Sequence, Type, Union

import torch.nn as nn

from .grad_sample_controller import GradSampleController
from .grad_sample_controller_fast_gradient_clipping import (
    GradSampleControllerFastGradientClipping,
)
from .grad_sample_module import GradSampleModule
from .grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from .grad_sample_module_fast_gradient_clipping_fsdp import (
    GradSampleModuleFastGradientClippingFSDP,
)
from .gsm_base import AbstractGradSampleModule
from .gsm_exp_weights import GradSampleModuleExpandedWeights
from .gsm_no_op import GradSampleModuleNoOp


def register_grad_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]],
):
    """
    Registers the decorated function as the ``grad_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient
    of ``target_class_or_classes``. The signature of every grad_sampler is always the same:

    >>> @register_grad_sampler(MyCustomModel)
    ... def compute_grad_sample(module, activations, backprops):
    ...    pass

    It may help you to take a look at the existing grad_samplers inside Opacus, under ``opacus.grad_sample.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            GradSampleModule.GRAD_SAMPLERS[target_class] = f
            GradSampleModuleFastGradientClipping.GRAD_SAMPLERS[target_class] = f
            GradSampleController.GRAD_SAMPLERS[target_class] = f
            GradSampleControllerFastGradientClipping.GRAD_SAMPLERS[target_class] = f
        return f

    return decorator


def register_norm_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]],
):
    """
    Registers the decorated function as the ``norm_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient norm
    of ``target_class_or_classes``. The signature of every norm_sampler is always the same:

    >>> @register_norm_sampler(MyCustomModel)
    ... def compute_grad_norm_sample(module, activations, backprops):
    ...    pass
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            GradSampleModuleFastGradientClipping.NORM_SAMPLERS[target_class] = f
            GradSampleControllerFastGradientClipping.NORM_SAMPLERS[target_class] = f
        return f

    return decorator


def get_gsm_class(grad_sample_mode: str) -> Type[AbstractGradSampleModule]:
    """
    Returns AbstractGradSampleModule subclass corresponding to the input mode.

    This is used for the wrapping approach where the model is wrapped in a
    GradSampleModule subclass.

    See README for detailed comparison between grad sample modes.

    Args:
        grad_sample_mode: Mode for computing per-sample gradients. Supported values:
            - "hooks": Standard hook-based computation (GradSampleModule)
            - "functorch": Functorch-based computation (GradSampleModule with force_functorch=True)
            - "ew": Expanded weights approach (GradSampleModuleExpandedWeights)
            - "ghost": Ghost clipping with wrapping (GradSampleModuleFastGradientClipping)
            - "ghost_fsdp": Ghost clipping with FSDP (GradSampleModuleFastGradientClippingFSDP)
            - "no_op": No-op implementation (GradSampleModuleNoOp)

    Returns:
        AbstractGradSampleModule subclass

    Raises:
        ValueError: If grad_sample_mode is not recognized
    """
    if grad_sample_mode in ["hooks", "functorch"]:
        return GradSampleModule
    elif grad_sample_mode == "ew":
        return GradSampleModuleExpandedWeights
    elif grad_sample_mode == "ghost":
        return GradSampleModuleFastGradientClipping
    elif grad_sample_mode == "ghost_fsdp":
        return GradSampleModuleFastGradientClippingFSDP
    elif grad_sample_mode == "no_op":
        return GradSampleModuleNoOp
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Allowed values: hooks, functorch, ew, ghost, ghost_fsdp, no_op"
        )


def get_gsc_class(grad_sample_mode: str):
    """
    Returns GradSampleController subclass corresponding to the input mode.

    This is used for the controller-based approach where hooks are attached
    directly to the model without wrapping.

    See README for a detailed comparison between grad sample modes.

    Args:
        grad_sample_mode: Mode for computing per-sample gradients. Supported values:
            - "hooks": Standard hook-based computation (GradSampleController)
            - "functorch": Functorch-based computation (GradSampleController with force_functorch=True)
            - "ghost": Ghost clipping without wrapping (GradSampleControllerFastGradientClipping)

    Returns:
        GradSampleController subclass

    Raises:
        ValueError: If grad_sample_mode is not recognized or not supported by controllers
    """
    if grad_sample_mode in ["hooks", "functorch"]:
        return GradSampleController
    elif grad_sample_mode == "ghost":
        return GradSampleControllerFastGradientClipping
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Controller-based approach supports: hooks, functorch, ghost"
        )


def wrap_model(
    model: nn.Module,
    grad_sample_mode: str,
    use_controller: bool = False,
    *args,
    **kwargs,
):
    """
    Wraps a model for per-sample gradient computation.

    This is a unified interface that supports both wrapping-based and controller-based
    approaches for computing per-sample gradients.

    Args:
        model: PyTorch module to be wrapped or controlled
        grad_sample_mode: Mode for computing per-sample gradients
        use_controller: If True, uses controller-based approach (no wrapping).
            If False (default), wraps model in GradSampleModule subclass.
        *args: Additional positional arguments passed to the wrapper/controller
        **kwargs: Additional keyword arguments passed to the wrapper/controller

    Returns:
        Either:
        - GradSampleModule subclass instance (if use_controller=False)
        - GradSampleController instance (if use_controller=True)

    Notes:
        - When use_controller=True, the original model is NOT wrapped and can be used
          as-is. The controller manages hooks on the side.
        - When use_controller=False, the model is wrapped and should be used via the
          returned wrapper object.
    """
    # Set force_functorch flag for functorch mode
    if grad_sample_mode == "functorch":
        kwargs["force_functorch"] = True

    if use_controller:
        cls = get_gsc_class(grad_sample_mode)
    else:
        cls = get_gsm_class(grad_sample_mode)

    return cls(model, *args, **kwargs)
