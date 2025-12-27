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

from typing import Iterable, Optional, Set

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (
    GradSampleModuleFastGradientClippingFSDP,
)
from opacus.utils.module_utils import has_trainable_params

try:
    # Composable FSDP2 API
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
except Exception:  # pragma: no cover
    MixedPrecisionPolicy = None  # type: ignore[assignment]
    fully_shard = None  # type: ignore[assignment]


def has_params(module: nn.Module) -> bool:
    return len(list(module.parameters(recurse=False))) > 0


def iterate_submodules(module: nn.Module) -> Iterable[nn.Module]:
    if has_params(module):
        yield module

    for m in module.children():
        yield from iterate_submodules(m)


def FSDP2Wrapper(
    model: nn.Module,
    *,
    ignore_trainable_params: bool = False,
    **kwargs,
) -> nn.Module:
    """Wrap a model with composable FSDP2 (`fully_shard`).

    Notes:
        Composable FSDP2 may swap sharded parameter objects across the iteration.
        Opacus' hooks-based per-sample gradients are stored as attributes on
        parameter objects (e.g. `p.grad_sample`), so those attributes can be lost
        if the parameter objects are swapped.

        For LoRA-only (or other sparse fine-tuning) where the trainable parameter
        set is small, setting `ignore_trainable_params=True` keeps trainable
        parameters unsharded/stable while still allowing sharding of frozen
        parameters.
    """

    if fully_shard is None or MixedPrecisionPolicy is None:
        raise ImportError(
            "Composable FSDP2 is not available in this PyTorch build; "
            "`FSDP2Wrapper` requires `torch.distributed._composable.fsdp`."
        )

    sampler_classes = set(
        list(GradSampleModuleFastGradientClippingFSDP.GRAD_SAMPLERS.keys())
        + list(GradSampleModuleFastGradientClippingFSDP.NORM_SAMPLERS.keys())
    )
    mp_policy = kwargs.get("mp_policy", MixedPrecisionPolicy())
    opacus_high_precision_layers = kwargs.get("opacus_high_precision_layers", [])
    for module in iterate_submodules(model):
        # The root module is sharded at the end; avoid applying fully_shard twice
        # for models that have parameters directly on the root (e.g. `nn.Linear`).
        if module is model:
            continue
        ignored_params: Optional[Set[nn.Parameter]] = None
        if ignore_trainable_params:
            # Opacus stores per-sample gradients as attributes on parameter objects.
            # Composable FSDP2 may swap sharded parameter objects across the iteration,
            # which would drop these attributes. For LoRA-only (or any sparse fine-tune)
            # this is typically acceptable: keep trainable params unsharded/stable and
            # shard only frozen params.
            ignored_params = {
                p for p in module.parameters(recurse=False) if p.requires_grad
            }
        if (type(module) in sampler_classes) or (not has_trainable_params(module)):
            if len(opacus_high_precision_layers) > 0 and isinstance(
                module, opacus_high_precision_layers
            ):
                # For certain layers, higher precision is needed to stablize the training of DP-SGD.
                fully_shard(
                    module,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.get_default_dtype()
                    ),
                    ignored_params=ignored_params,
                )
            else:
                fully_shard(module, mp_policy=mp_policy, ignored_params=ignored_params)

    root_ignored_params: Optional[Set[nn.Parameter]] = None
    if ignore_trainable_params:
        root_ignored_params = {p for p in model.parameters() if p.requires_grad}

    model = fully_shard(model, mp_policy=mp_policy, ignored_params=root_ignored_params)
    return model
