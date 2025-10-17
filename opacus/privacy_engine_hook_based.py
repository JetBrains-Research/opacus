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
PrivacyEngineHookBased: A GradSampleModule-less implementation of PrivacyEngine.

This module provides an alternative approach to the standard PrivacyEngine that
attaches hooks directly to models without wrapping them in GradSampleModule.
This improves compatibility with transformers and other complex models.
"""

import os
import warnings
from itertools import chain
from typing import Any, BinaryIO, Dict, IO, List, Optional, Tuple, Union

import torch
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import GradSampleModule
from opacus.hook_controller import HookController
from opacus.optimizers import DPOptimizer, get_optimizer_class
from opacus.schedulers import _GradClipScheduler, _NoiseScheduler
from opacus.validators.module_validator import ModuleValidator
from torch import nn, optim
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


class PrivacyEngineHookBased:
    """
    Alternative PrivacyEngine that uses direct hook attachment instead of module wrapping.

    This implementation provides the same functionality as PrivacyEngine but attaches
    hooks directly to the model without wrapping it in a GradSampleModule. This approach
    is more compatible with transformers and other models that have complex module
    introspection or custom __getattr__ behavior.

    Can be used as a context manager for automatic cleanup:

    Example:
        >>> model = MyCustomModel()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> dataloader = demo_dataloader
        >>>
        >>> # Standard usage
        >>> privacy_engine = PrivacyEngineHookBased()
        >>> model, optimizer, dataloader = privacy_engine.make_private(
        ...    module=model,
        ...    optimizer=optimizer,
        ...    data_loader=dataloader,
        ...    noise_multiplier=1.0,
        ...    max_grad_norm=1.0,
        ... )
        >>> # ... training ...
        >>> privacy_engine.cleanup()  # Manual cleanup
        >>>
        >>> # Context manager usage (recommended)
        >>> with PrivacyEngineHookBased() as privacy_engine:
        ...     model, optimizer, dataloader = privacy_engine.make_private(
        ...         module=model,
        ...         optimizer=optimizer,
        ...         data_loader=dataloader,
        ...         noise_multiplier=1.0,
        ...         max_grad_norm=1.0,
        ...     )
        ...     # ... training ...
        ... # Automatic cleanup on exit
    """

    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False):
        """
        Initialize the PrivacyEngineHookBased.

        Args:
            accountant: Accounting mechanism. Currently supported:
                - rdp (:class:`~opacus.accountants.RDPAccountant`)
                - gdp (:class:`~opacus.accountants.GaussianAccountant`)
                - prv (:class`~opacus.accountants.PRVAccountant`)
            secure_mode: Set to ``True`` if cryptographically strong DP guarantee is
                required. ``secure_mode=True`` uses secure random number generator for
                noise and shuffling (as opposed to pseudo-rng in vanilla PyTorch).
                When set to ``True`` requires ``torchcsprng`` to be installed
        """
        self.accountant = create_accountant(mechanism=accountant)
        self.secure_mode = secure_mode
        self.secure_rng = None
        self.dataset = None
        self.hook_controller = None  # Will be created in make_private

        if self.secure_mode:
            try:
                import torchcsprng as csprng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            self.secure_rng = csprng.create_random_device_generator("/dev/urandom")
        else:
            warnings.warn(
                "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
                "for much faster training performance, but remember to turn it on and retrain "
                "one last time before production with ``secure_mode`` turned on."
            )

    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        **kwargs,
    ) -> DPOptimizer:
        """
        Prepare the optimizer for DP training.

        Args:
            optimizer: The optimizer to wrap
            noise_multiplier: Noise multiplier for DP
            max_grad_norm: Maximum gradient norm for clipping
            expected_batch_size: Expected batch size
            loss_reduction: "mean" or "sum"
            distributed: Whether using distributed training
            clipping: Clipping strategy ("flat", "per_layer", or "adaptive")
            noise_generator: Random number generator for noise
            grad_sample_mode: Mode for computing grad samples

        Returns:
            DPOptimizer wrapping the original optimizer
        """
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
        )

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            **kwargs,
        )

    def _prepare_data_loader(
        self,
        data_loader: DataLoader,
        *,
        poisson_sampling: bool,
        distributed: bool,
        batch_first: bool = True,
        rand_on_empty: bool = False,
    ) -> DataLoader:
        """
        Prepare the data loader for DP training.

        Args:
            data_loader: The data loader to prepare
            poisson_sampling: Whether to use Poisson sampling
            distributed: Whether using distributed training
            batch_first: Whether batch dimension is first
            rand_on_empty: Whether to return random data on empty batches

        Returns:
            Prepared data loader
        """
        if self.dataset is None:
            self.dataset = data_loader.dataset
        elif self.dataset != data_loader.dataset:
            warnings.warn(
                f"PrivacyEngine detected new dataset object. "
                f"Was: {self.dataset}, got: {data_loader.dataset}. "
                f"Privacy accounting works per dataset, please initialize "
                f"new PrivacyEngine if you're using different dataset. "
                f"You can ignore this warning if two datasets above "
                f"represent the same logical dataset"
            )

        if poisson_sampling:
            return DPDataLoader.from_data_loader(
                data_loader,
                generator=self.secure_rng,
                distributed=distributed,
                batch_first=batch_first,
                rand_on_empty=rand_on_empty,
            )
        elif self.secure_mode:
            return switch_generator(data_loader=data_loader, generator=self.secure_rng)
        else:
            return data_loader

    def _prepare_model(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        force_functorch: bool = False,
        strict: bool = False,
    ) -> nn.Module:
        """
        Prepare the model by attaching hooks directly (no wrapping).

        Args:
            module: The module to prepare
            batch_first: Whether batch dimension is first
            loss_reduction: "mean" or "sum"
            force_functorch: Whether to force functorch for all modules
            strict: If True, validate that module has no buffers

        Returns:
            The same module with hooks attached
        """
        # Validate the module (will be done again in HookController if strict=True)
        self.validate(module=module, optimizer=None, data_loader=None)

        # Create hook controller and attach hooks
        # HookController will automatically get GRAD_SAMPLERS from GradSampleModule
        self.hook_controller = HookController(
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            force_functorch=force_functorch,
            strict=strict,
        )

        return module

    def is_compatible(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ) -> bool:
        """
        Check if task components are compatible with DP.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Returns:
            ``True`` if compatible, ``False`` otherwise
        """
        return ModuleValidator.is_valid(module)

    def validate(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ):
        """
        Validate that task components are compatible with DP.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Raises:
            UnsupportedModuleError
                If one or more modules found to be incompatible
        """
        ModuleValidator.validate(module, strict=True)

    @classmethod
    def get_compatible_module(cls, module: nn.Module) -> nn.Module:
        """
        Return a privacy engine compatible module.

        Args:
            module: module to be modified

        Returns:
            Module with some submodules replaced for their deep copies or
            close equivalents.
        """
        module = ModuleValidator.fix(module)
        ModuleValidator.validate(module, strict=True)
        return module

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        rand_on_empty: bool = False,
        force_functorch: bool = False,
        **kwargs,
    ) -> Tuple[nn.Module, DPOptimizer, DataLoader]:
        """
        Add privacy-related responsibilities to the main PyTorch training objects.

        Unlike the standard PrivacyEngine, this method does NOT wrap the model in a
        GradSampleModule. Instead, it attaches hooks directly to the model and manages
        them through a HookController.

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise
            max_grad_norm: The maximum norm of the per-sample gradients
            batch_first: Flag to indicate if the input tensor has batch dimension first
            loss_reduction: Indicates if the loss reduction is a sum or a mean operation
            poisson_sampling: ``True`` if you want to use standard sampling required for DP
            clipping: Per sample gradient clipping mechanism
            noise_generator: torch.Generator() object used as a source of randomness
            grad_sample_mode: mode for computing per sample gradients
            rand_on_empty: Return random batch when encountering empty batches
            force_functorch: Force use of functorch for all modules

        Returns:
            Tuple of (model, optimizer, data_loader).
            Note: Model is NOT wrapped - it's the original module with hooks attached.
        """
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # Validate that optimizer parameters match module parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP, FSDPModule))

        # Prepare model (attach hooks directly, no wrapping)
        module = self._prepare_model(
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            force_functorch=force_functorch,
            strict=kwargs.get("strict", False),
        )

        if poisson_sampling:
            self.hook_controller.forbid_grad_accumulation()

        data_loader = self._prepare_data_loader(
            data_loader,
            distributed=distributed,
            poisson_sampling=poisson_sampling,
            batch_first=batch_first,
            rand_on_empty=rand_on_empty,
        )

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            **kwargs,
        )

        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )

        return module, optimizer, data_loader

    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        force_functorch: bool = False,
        **kwargs,
    ) -> Tuple[nn.Module, DPOptimizer, DataLoader]:
        """
        Version of make_private that calculates privacy parameters based on a given privacy budget.

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            target_epsilon: Target epsilon to be achieved
            target_delta: Target delta to be achieved
            epochs: Number of training epochs
            max_grad_norm: The maximum norm of the per-sample gradients
            batch_first: Flag to indicate if the input tensor has batch dimension first
            loss_reduction: Indicates if the loss reduction is a sum or a mean operation
            poisson_sampling: ``True`` if you want to use standard sampling required for DP
            clipping: Per sample gradient clipping mechanism
            noise_generator: torch.Generator() object used as a source of randomness
            grad_sample_mode: mode for computing per sample gradients
            force_functorch: Force use of functorch for all modules

        Returns:
            Tuple of (model, optimizer, data_loader)
        """
        sample_rate = 1 / len(data_loader)

        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                accountant=self.accountant.mechanism(),
                **kwargs,
            ),
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            force_functorch=force_functorch,
            **kwargs,
        )

    def get_epsilon(self, delta):
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        Args:
            delta: The target delta.

        Returns:
            Privacy budget (epsilon) expended so far.
        """
        return self.accountant.get_epsilon(delta)

    def save_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: nn.Module,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        module_state_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_save_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Saves the state_dict of module, optimizer, and accountant at path.

        Note: Since the model is not wrapped, state_dict is directly from the module.

        Args:
            path: Path to save the state dict objects.
            module: Module to save (not wrapped)
            optimizer: DPOptimizer to save; wrapped optimizer's state_dict is saved.
            noise_scheduler: _NoiseScheduler whose state we should save.
            grad_clip_scheduler: _GradClipScheduler whose state we should save.
            checkpoint_dict: Dict[str, Any]; an already-filled checkpoint dict.
            module_state_dict_kwargs: dict of kwargs to pass to ``module.state_dict()``
            torch_save_kwargs: dict of kwargs to pass to ``torch.save()``
        """
        checkpoint_dict = checkpoint_dict or {}
        checkpoint_dict["module_state_dict"] = module.state_dict(
            **(module_state_dict_kwargs or {})
        )
        checkpoint_dict["privacy_accountant_state_dict"] = self.accountant.state_dict()
        if optimizer is not None:
            checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
        if noise_scheduler is not None:
            checkpoint_dict["noise_scheduler_state_dict"] = noise_scheduler.state_dict()
        if grad_clip_scheduler is not None:
            checkpoint_dict["grad_clip_scheduler_state_dict"] = (
                grad_clip_scheduler.state_dict()
            )

        torch.save(checkpoint_dict, path, **(torch_save_kwargs or {}))

    def load_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: nn.Module,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        module_load_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Loads checkpoint from path.

        Args:
            path: Path to load the checkpoint from
            module: Module to load state into (not wrapped)
            optimizer: DPOptimizer to load state into
            noise_scheduler: _NoiseScheduler to load state into
            grad_clip_scheduler: _GradClipScheduler to load state into
            module_load_dict_kwargs: dict of kwargs to pass to ``module.load_state_dict()``
            torch_load_kwargs: dict of kwargs to pass to ``torch.load()``

        Returns:
            The loaded checkpoint dict
        """
        checkpoint = torch.load(path, **(torch_load_kwargs or {}), weights_only=False)
        module.load_state_dict(
            checkpoint["module_state_dict"], **(module_load_dict_kwargs or {})
        )
        self.accountant.load_state_dict(checkpoint["privacy_accountant_state_dict"])

        optimizer_state_dict = checkpoint.pop("optimizer_state_dict", {})
        if optimizer is not None and len(optimizer_state_dict) > 0:
            optimizer.load_state_dict(optimizer_state_dict)
        elif (optimizer is not None) ^ (len(optimizer_state_dict) > 0):
            warnings.warn(
                f"optimizer_state_dict has {len(optimizer_state_dict)} items"
                f" but optimizer is {'' if optimizer else 'not'} provided."
            )

        noise_scheduler_state_dict = checkpoint.pop("noise_scheduler_state_dict", {})
        if noise_scheduler is not None and len(noise_scheduler_state_dict) > 0:
            noise_scheduler.load_state_dict(noise_scheduler_state_dict)

        grad_clip_scheduler_state_dict = checkpoint.pop(
            "grad_clip_scheduler_state_dict", {}
        )
        if grad_clip_scheduler is not None and len(grad_clip_scheduler_state_dict) > 0:
            grad_clip_scheduler.load_state_dict(grad_clip_scheduler_state_dict)

        return checkpoint

    def cleanup(self):
        """
        Clean up all hooks and attributes added to the model.
        Call this when you're done with DP training.

        Note: This is called automatically when using the privacy engine as a context manager.
        """
        if self.hook_controller is not None:
            self.hook_controller.cleanup()
            self.hook_controller = None

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            self: The privacy engine instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            False: Don't suppress exceptions
        """
        self.cleanup()
        return False
