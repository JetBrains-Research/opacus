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
from functools import partial
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
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


def create_norm_sample(
    *, param: torch.Tensor, grad_sample: torch.Tensor, max_batch_len: int
) -> None:
    """
    Creates a ``_norm_sample`` attribute in the given parameter


    Args:
        param: Parameter to which ``_norm_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
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


class GradSampleControllerFastGradientClipping:
    """
    Controller for managing privacy hooks with Fast Gradient and Ghost Clipping support

    Computes per-sample gradient norms using custom-written methods for each layer,
    without wrapping the model. Supports both:
    - Ghost Clipping: Direct norm computation without materializing full gradients
    - Fast Gradient Clipping: Full gradient computation followed by norm computation

    This class attaches hooks directly to model modules and manages their lifecycle,
    providing an alternative to GradSampleModule wrapping that's more compatible
    with transformers and other complex models.
    """

    GRAD_SAMPLERS = {}
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
        errors = self.validate(module=m, strict=strict)
        if errors and not strict:
            logger.info(
                f"GradSampleControllerFastGradientClipping found the following errors: {errors}."
                "Using non-strict mode, continuing"
            )

        if logger.isEnabledFor(logging.INFO):
            self.log_module_gradient_sample_mode(
                module=m,
                force_functorch=force_functorch,
                use_ghost_clipping=use_ghost_clipping,
            )

        self.module = m
        self.hooks_enabled = False
        self.grad_accumulation_allowed = True
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.force_functorch = force_functorch
        self.max_grad_norm = max_grad_norm
        self.use_ghost_clipping = use_ghost_clipping
        self._per_sample_gradient_norms = None

        self.autograd_grad_sample_hooks: List[RemovableHandle] = []

        # Initialize parameters with required attributes
        for _, p in trainable_parameters(self.module):
            p.grad_sample = None
            p._forward_counter = 0
            p._norm_sample = None

        self.trainable_parameters = [p for _, p in trainable_parameters(self.module)]

        # Add the hooks
        self.add_hooks()

    def iterate_submodules(self, module: nn.Module) -> Iterable[nn.Module]:
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

    def add_hooks(self) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradient norms in params._norm_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.
        """
        for module in self.iterate_submodules(self.module):
            # Do not add hooks to DPRNN, DPLSTM or DPGRU
            if type(module) in [DPRNN, DPLSTM, DPGRU]:
                continue

            module_type = type(module)
            if self.force_functorch or not (module_type in self.GRAD_SAMPLERS):
                prepare_layer(module, batch_first=self.batch_first)

            self.autograd_grad_sample_hooks.append(
                module.register_forward_hook(self.capture_activations_hook)
            )

            self.autograd_grad_sample_hooks.append(
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

    def get_clipping_coef(self) -> torch.Tensor:
        """Get per-example gradient scaling factor for clipping."""
        norm_sample = self.get_norm_sample()
        return (self.max_grad_norm / (norm_sample + 1e-6)).clamp(max=1.0)

    def get_norm_sample(self) -> torch.Tensor:
        """Get per-example gradient norms."""
        norm_sample = torch.stack(
            [param._norm_sample for param in self.trainable_parameters], dim=0
        ).norm(2, dim=0)
        self.per_sample_gradient_norms = norm_sample
        return norm_sample

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
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
        module.activations.append([t.detach() for t in forward_input])  # pyre-ignore

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
        Computes per sample gradient norms given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradient norms are
        stored in ``_norm_sample`` field in each parameter.

        For non-recurrent layers the process is straightforward: for each
        ``loss.backward()`` call this hook will be called exactly one. For recurrent
        layers, however, this is more complicated and the hook will be called multiple
        times, while still processing the same batch of data.

        For this reason we first accumulate the gradients from *the same batch* in
        ``p._current_grad_sample`` and then, when we detect the end of a full backward
        pass - we store accumulated result on ``p.grad_sample`` (for fast gradient clipping)
        or ``p._norm_sample`` (for ghost clipping).

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
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

        if self.use_ghost_clipping and type(module) in self.NORM_SAMPLERS:
            # Ghost clipping: compute norms directly
            norm_sampler_fn = self.NORM_SAMPLERS[type(module)]
            norm_samples = norm_sampler_fn(module, activations, backprops)

            for param, ns in norm_samples.items():
                if param.requires_grad:
                    param._norm_sample = ns
                    param._forward_counter -= 1

        else:
            # Fast gradient clipping: compute full gradients then norms
            if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
                grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
            else:
                grad_sampler_fn = ft_compute_per_sample_gradient

            grad_samples = grad_sampler_fn(module, activations, backprops)
            for param, gs in grad_samples.items():
                create_or_accumulate_grad_sample(
                    param=param, grad_sample=gs, max_batch_len=module.max_batch_len
                )
            del grad_samples

            # Detect end of current batch processing and switch accumulation
            # mode from sum to stacking. Used for RNNs and tied parameters
            # (See #417 for details)
            for _, p in trainable_parameters(module):
                p._forward_counter -= 1
                if p._forward_counter == 0:
                    promote_current_grad_sample(p)
                    create_norm_sample(
                        param=p,
                        grad_sample=p.grad_sample,
                        max_batch_len=module.max_batch_len,
                    )
                    p.grad_sample = None

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
        Rearrange activations and grad_samples based on loss reduction and batch dim

        Args:
            module: the module for which per-sample gradients are computed
            backprops: the captured backprops
            loss_reduction: either "mean" or "sum" depending on whether backpropped
                loss was averaged or summed over batch
            batch_first: True is batch dimension is first

        Returns:
            Tuple of (activations, backprops)
        """
        if not hasattr(module, "activations"):
            raise ValueError(
                f"No activations detected for {type(module)},"
                " run forward after add_hooks(model)"
            )

        batch_dim = 0 if batch_first or type(module) is RNNLinear else 1

        if not hasattr(module, "max_batch_len"):
            # For packed sequences, max_batch_len is set in the forward of the model (e.g. the LSTM)
            # Otherwise we infer it here
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

    @classmethod
    def validate(
        cls, module: nn.Module, *, strict: bool = False
    ) -> List[UnsupportedModuleError]:
        """
        Check if per sample gradients can be fully computed for a given model

        Args:
            module: nn.Module to be checked
            strict: If True, raise error if module has buffers

        Returns:
            Empty list if validation is successful.
            List of validation errors if unsupported modules are found.

        Raises:
            UnsupportedModuleError
                If ``strict=True`` and unsupported modules are found
        """
        errors = []
        errors.extend(
            [
                UnsupportedModuleError(
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
            raise UnsupportedModuleError(errors)
        else:
            return errors

    def forbid_grad_accumulation(self):
        self.grad_accumulation_allowed = False

    def allow_grad_accumulation(self):
        self.grad_accumulation_allowed = True

    def log_module_gradient_sample_mode(
        self, module: nn.Module, *, force_functorch=False, use_ghost_clipping=True
    ):
        """
        Add logs to track gradient sample mode for each part of the module, including 1) Ghost Clipping, 2) Fast Gradient Clipping (hook mode), and 3) Fast Gradient Clipping (functorch mode).

        Args:
            module: nn.Module to be checked
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.
            use_ghost_clipping: If set to ``True``, Ghost Clipping
                will be used for clipping gradients of supported layers. If ``False``, Fast
                Gradient Clipping will be used for all layers.
        """
        for m_name, m in trainable_modules(module):
            if type(m) in [DPRNN, DPLSTM, DPGRU]:
                logger.info(
                    f"Module name: {m_name}, module type: {type(m)}. No hook or functorch is added."
                )

            elif use_ghost_clipping and type(m) in self.NORM_SAMPLERS:
                logger.info(
                    f"Module name: {m_name}, module type: {type(m)}, under Ghost Clipping."
                )

            else:
                if not force_functorch and type(m) in self.GRAD_SAMPLERS:
                    # When functorch is not enforced, use FGC (hook mode) if the layer has a registered grad_sampler (supported). Otherwise, use FGC (functorch mode).
                    logger.info(
                        f"Module name: {m_name}, module type: {type(m)}, under Fast Gradient Clipping (hook mode)."
                    )
                else:
                    logger.info(
                        f"Module name: {m_name}, module type: {type(m)}, under Fast Gradient Clipping (functorch mode)."
                    )

    @property
    def per_sample_gradient_norms(self) -> torch.Tensor:
        """Returns per sample gradient norms. Note that these are not privatized and should only be used for debugging purposes or in non-private settings"""
        if self._per_sample_gradient_norms is not None:
            return self._per_sample_gradient_norms
        else:
            raise AttributeError(
                "per_sample_gradient_norms is not set. Please call forward and backward on the model before accessing this property."
            )

    @per_sample_gradient_norms.setter
    def per_sample_gradient_norms(self, value):
        self._per_sample_gradient_norms = value
