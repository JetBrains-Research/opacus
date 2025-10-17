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
HookController: Manages privacy hooks on models without wrapping them.

This module provides a GradSampleModule-less approach to attaching hooks
directly to model parameters for computing per-sample gradients.
"""

import logging
from functools import partial
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN, RNNLinear
from opacus.utils.module_utils import (
    has_trainable_params,
    requires_grad,
    trainable_modules,
    trainable_parameters,
)
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.hooks import RemovableHandle


logger = logging.getLogger(__name__)
logger.disabled = True


OPACUS_PARAM_MONKEYPATCH_ATTRS = [
    "grad_sample",
    "_forward_counter",
    "_current_grad_sample",
    "_norm_sample",
]


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
        max_batch_len: Maximum batch length
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
    """
    Promotes _current_grad_sample to grad_sample after backward pass.

    Args:
        p: Parameter whose grad_sample should be promoted
    """
    if p.requires_grad:
        if p.grad_sample is not None:
            if isinstance(p.grad_sample, list):
                p.grad_sample.append(p._current_grad_sample)
            else:
                p.grad_sample = [p.grad_sample, p._current_grad_sample]
        else:
            p.grad_sample = p._current_grad_sample

        del p._current_grad_sample


def _get_batch_size(*, module: nn.Module, batch_dim: int) -> int:
    """
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over module.activations.

    Args:
        module: input module
        batch_dim: batch dimension

    Returns:
        Maximum sequence length in a batch
    """
    max_batch_len = 0
    for out in module.activations:
        if out[-1].shape[batch_dim] > max_batch_len:
            max_batch_len = out[-1].shape[batch_dim]

    return max_batch_len


class HookController:
    """
    Controller for managing privacy hooks on models without wrapping them.

    This class attaches hooks directly to model modules and manages their lifecycle,
    providing an alternative to GradSampleModule wrapping that's more compatible
    with transformers and other complex models.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        force_functorch: bool = False,
        grad_samplers: Dict = None,
        strict: bool = False,
    ):
        """
        Initialize the HookController.

        Args:
            model: The PyTorch model to attach hooks to
            batch_first: Flag to indicate if the input tensor has batch dimension first
            loss_reduction: Indicates if the loss reduction is "sum" or "mean"
            force_functorch: If True, use functorch for all grad sampling
            grad_samplers: Dictionary mapping module types to grad sampler functions
            strict: If True, validate that module has no buffers
        """
        self.model = model
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.force_functorch = force_functorch

        # If model is wrapped in DDP/DPDDP, work with the underlying module
        # FSDP is handled differently (requires specialized implementation)
        if isinstance(model, (DDP, DPDDP)):
            self.target_module = model.module
        elif isinstance(model, FSDPModule):
            # FSDP requires specialized grad sample computation
            # For now, work with the model as-is
            # TODO: Implement FSDP-specific hook controller
            self.target_module = model
        else:
            self.target_module = model

        # If grad_samplers not provided, use the ones from GradSampleModule
        if grad_samplers is None:
            from opacus.grad_sample.grad_sample_module import GradSampleModule
            self.grad_samplers = GradSampleModule.GRAD_SAMPLERS
        else:
            self.grad_samplers = grad_samplers

        self.hooks_enabled = False
        self.grad_accumulation_allowed = True
        self.hook_handles: List[RemovableHandle] = []

        # Validate if strict mode
        if strict:
            errors = self.validate(module=self.target_module, strict=strict)
            if errors:
                raise NotImplementedError(errors)

        # Initialize parameters with required attributes
        for _, p in trainable_parameters(self.target_module):
            p.grad_sample = None
            p._forward_counter = 0

        # Add the hooks
        self.add_hooks()

    def iterate_submodules(self, module: nn.Module):
        """
        Iterate over submodules that should have hooks attached.

        Args:
            module: Module to iterate over

        Yields:
            Submodules that should have hooks
        """
        if has_trainable_params(module):
            yield module

        # Don't recurse if module is handled by functorch
        if (
            has_trainable_params(module)
            and type(module) not in self.grad_samplers
            and type(module) not in [DPRNN, DPLSTM, DPGRU]
        ):
            return

        for m in module.children():
            yield from self.iterate_submodules(m)

    def add_hooks(self) -> None:
        """
        Adds forward and backward hooks to model modules.
        """
        for module in self.iterate_submodules(self.target_module):
            # Do not add hooks to DPRNN, DPLSTM or DPGRU
            if type(module) in [DPRNN, DPLSTM, DPGRU]:
                continue

            module_type = type(module)
            if self.force_functorch or not (module_type in self.grad_samplers):
                prepare_layer(module, batch_first=self.batch_first)

            # Register forward hook
            self.hook_handles.append(
                module.register_forward_hook(self.capture_activations_hook)
            )

            # Register backward hook
            self.hook_handles.append(
                module.register_full_backward_hook(
                    partial(
                        self.capture_backprops_hook,
                        loss_reduction=self.loss_reduction,
                        batch_first=self.batch_first,
                    )
                )
            )

        self.enable_hooks()

    def remove_hooks(self) -> None:
        """
        Removes all hooks from the model.
        """
        self.disable_hooks()

        # Remove all hook handles
        while self.hook_handles:
            handle = self.hook_handles.pop()
            handle.remove()

        # Remove functorch attributes
        for _module_name, module in trainable_modules(self.target_module):
            if hasattr(module, "ft_compute_sample_grad"):
                delattr(module, "ft_compute_sample_grad")
            if hasattr(module, "activations"):
                delattr(module, "activations")

    def disable_hooks(self) -> None:
        """
        Disable all hooks without removing them.
        """
        self.hooks_enabled = False

    def enable_hooks(self) -> None:
        """
        Enable all hooks.
        """
        self.hooks_enabled = True

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        """
        Forward hook that captures activations.

        Args:
            module: The module this hook is attached to
            forward_input: Input to the module
            _forward_output: Output from the module
        """
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
        Backward hook that computes per-sample gradients.

        Args:
            module: The module this hook is attached to
            _forward_input: Input to the module
            forward_output: Output from the module
            loss_reduction: "sum" or "mean"
            batch_first: Whether batch dimension is first
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

        # Choose grad sampler function
        if (
            not self.force_functorch
            and type(module) in self.grad_samplers
        ):
            grad_sampler_fn = self.grad_samplers[type(module)]
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
        """
        Rearrange activations and grad_samples based on loss reduction and batch dim.

        Args:
            module: The module for which per-sample gradients are computed
            backprops: The captured backprops
            loss_reduction: Either "mean" or "sum"
            batch_first: True if batch dimension is first

        Returns:
            Tuple of (activations, backprops)
        """
        if not hasattr(module, "activations"):
            raise ValueError(
                f"No activations detected for {type(module)},"
                " run forward after adding hooks"
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

        # Always put batch dimension in the first dim for grad_samples
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
        """
        Forbid gradient accumulation (for Poisson sampling).
        """
        self.grad_accumulation_allowed = False

    def allow_grad_accumulation(self):
        """
        Allow gradient accumulation.
        """
        self.grad_accumulation_allowed = True

    def set_grad_sample_to_none(self):
        """
        Sets ``.grad_sample`` to None for all parameters.
        """
        for p in self.target_module.parameters():
            p.grad_sample = None

    @classmethod
    def validate(
        cls, module: nn.Module, *, strict: bool = False
    ) -> List[NotImplementedError]:
        """
        Check if per sample gradients can be fully computed for a given model.

        Args:
            module: nn.Module to be checked
            strict: If True, raise error if module has buffers

        Returns:
            Empty list if validation is successful.
            List of validation errors if unsupported modules are found.

        Raises:
            NotImplementedError
                If ``strict=True`` and unsupported modules are found
        """
        errors = []
        errors.extend(
            [
                NotImplementedError(
                    f"Model contains a trainable layer with buffers "
                    f"that Opacus doesn't currently support ({m_name}:{m}). "
                )
                for m_name, m in trainable_modules(module)
                # With functorch, all modules are trainable
                # We still want to avoid modules that have buffers (e.g. BatchNorm)
                # as the buffers are not private
                if len(list(m.buffers())) > 0
            ]
        )
        # raise or return errors as needed
        if strict and len(errors) > 0:
            raise NotImplementedError(errors)
        else:
            return errors

    def cleanup(self):
        """
        Clean up all hooks and attributes added to the model.
        """
        self.remove_hooks()

        # Clean up parameter attributes
        for attr in OPACUS_PARAM_MONKEYPATCH_ATTRS:
            for p in self.target_module.parameters():
                if hasattr(p, attr):
                    delattr(p, attr)
