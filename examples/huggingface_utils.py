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

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from accelerate.utils import DistributedType
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.grad_sample.utils import wrap_model
from opacus.optimizers import (
    AdaClipDPOptimizer,
    DPOptimizer,
    get_optimizer_class,
)
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import nn
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
    logging,
)
from transformers.training_args import ParallelMode
from transformers.trainer_utils import FSDPOption
from transformers.trainer_callback import ExportableState


logger = logging.get_logger(__name__)

# Maximum depth for unwrapping nested optimizer wrappers
MAX_WRAPPER_DEPTH = 10


def unwrap_optimizer(optimizer, target_class=DPOptimizer):
    """
    Unwrap nested optimizer to find the target class.

    Args:
        optimizer: The optimizer to unwrap
        target_class: The class to search for (default: DPOptimizer)

    Returns:
        The unwrapped optimizer if found, otherwise the last optimizer in the chain
    """
    for _ in range(MAX_WRAPPER_DEPTH):
        if isinstance(optimizer, target_class):
            return optimizer
        elif hasattr(optimizer, "optimizer"):  # accelerate.Optimizer
            optimizer = optimizer.optimizer
        elif hasattr(optimizer, "_optimizer"):
            optimizer = optimizer._optimizer
        else:
            break
    return optimizer


class OptimizerProxy:
    """
    Lazy proxy for accessing DPOptimizer from trainer.

    This is necessary because:
    1. HF Trainer creates the dataloader before the optimizer
    2. BatchMemoryManager needs optimizer reference at dataloader creation time
    3. The optimizer might be wrapped by Accelerate after creation

    This proxy defers optimizer access until it's actually needed (during iteration),
    ensuring the optimizer exists and is properly unwrapped.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def _get_optimizer(self):
        optimizer = self.trainer.optimizer
        if optimizer is None:
            raise AttributeError("Optimizer not yet created in DPTrainer")
        return unwrap_optimizer(optimizer)

    def __getattr__(self, name):
        return getattr(self._get_optimizer(), name)

    def signal_skip_step(self, do_skip: bool):
        return self._get_optimizer().signal_skip_step(do_skip)


@dataclass
class PrivacyArguments:
    """
    Arguments for differentially private training.
    """

    accountant: str = "rdp"
    grad_sample_mode: str = "hooks"
    per_sample_max_grad_norm: float = 0.5
    clipping: str = "flat"
    poisson_sampling: bool = True
    min_clipbound: float = 0.05
    max_clipbound: float = 1e8
    clipbound_learning_rate: float = 0.2
    target_unclipped_quantile: float = 0.5
    unclipped_num_std: float = 1.0
    noise_multiplier: Optional[float] = None
    target_epsilon: Optional[float] = None
    target_delta: Optional[float] = None

    def precalculate(self, num_samples: int, sample_rate: float, steps: int):
        """
        Precalculate noise multiplier if not provided.
        """
        if self.target_delta is None:
            self.target_delta = 1.0 / num_samples

        if self.noise_multiplier is not None:
            return

        if self.target_epsilon is not None:
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                sample_rate=sample_rate,
                steps=steps,
                accountant=self.accountant,
            )
        else:
            raise ValueError(
                "Either noise_multiplier or target_epsilon must be specified."
            )


class DPCallback(TrainerCallback, ExportableState):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with Opacus.
    """

    def __init__(
        self,
        accountant: str,
        gradient_accumulation_steps: int,
        target_delta: float,
        max_epsilon: float = None,
    ) -> None:
        self.accountant = create_accountant(accountant)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.target_delta = target_delta
        self.max_epsilon = max_epsilon

    def get_optimizer_callback(self, sample_rate):
        return self.accountant.get_optimizer_hook_fn(sample_rate)

    def on_train_begin(self, args, state, control, **kwargs):
        return self._check_max_privacy_budget_exceeded(control)

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        optimizer = self._get_dp_optimizer(optimizer)

        # trainer samples one extra element at the beginning of each epoch, cleaning it up if present
        while len(optimizer._step_skip_queue) > self.gradient_accumulation_steps:
            optimizer._step_skip_queue.pop(0)

    def on_substep_end(self, args, state, control, optimizer=None, **kwargs):
        optimizer = self._get_dp_optimizer(optimizer)

        # gradients should be cleared after each substep with poisson sampling
        # precalculated grad_sample will stay until the final aggregation
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        optimizer = self._get_dp_optimizer(optimizer)

        # gradients should be cleared after each substep with poisson sampling
        # precalculated grad_sample will stay until the final aggregation
        # optimizer.step() is executed by the trainer
        optimizer.zero_grad(set_to_none=True)

    def on_evaluate(self, args, state, control, optimizer=None, metrics=None, **kwargs):
        return self._check_max_privacy_budget_exceeded(control)

    def get_privacy_metrics(self):
        metrics = {}
        if self.target_delta is not None:
            metrics["privacy_epsilon"] = (
                self.accountant.get_epsilon(self.target_delta)
                if len(self.accountant.history) > 0
                else 0.0
            )

        return metrics

    def _get_dp_optimizer(self, optimizer) -> DPOptimizer:
        optimizer = unwrap_optimizer(optimizer, DPOptimizer)
        if not isinstance(optimizer, DPOptimizer):
            raise ValueError(f"Expected DPOptimizer, got {type(optimizer)}")
        return optimizer

    def _check_max_privacy_budget_exceeded(
        self, control: TrainerControl
    ) -> TrainerControl:
        metrics = self.get_privacy_metrics()
        if (
            "privacy_epsilon" in metrics
            and self.max_epsilon is not None
            and metrics["privacy_epsilon"] >= self.max_epsilon
        ):
            logger.warning(
                f"Max epsilon exceeded: {metrics['privacy_epsilon']} >= {self.max_epsilon}."
                "Stopping training..."
            )
            control.should_training_stop = True

        return control

    @property
    def _accountant_state_dict(self):
        return self.accountant.state_dict()

    @_accountant_state_dict.setter
    def _accountant_state_dict(self, state_dict):
        self.accountant.load_state_dict(state_dict)

    def state(self) -> dict:
        return {
            "args": {
                "accountant": self.accountant.mechanism(),
                "target_delta": self.target_delta,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_epsilon": self.max_epsilon,
            },
            "attributes": {
                "_accountant_state_dict": self._accountant_state_dict,
            },
        }


class DPTrainer(Trainer):
    def __init__(
        self,
        model: Union[nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: torch.utils.data.Dataset = None,
        privacy_args: PrivacyArguments = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        **kwargs,
    ):
        """Huggingface Trainer with Differential Privacy support.

        Args:
            model: Model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            privacy_args: Privacy arguments for differential private training.
            compute_metrics: Custom evaluation metrics.
            callbacks: Training callbacks.
            kwargs: Additional keyword arguments passed to Trainer.
        """
        self.privacy_args = privacy_args
        if not self.privacy_args:
            raise ValueError("Privacy arguments must be provided.")

        # Store the original grad_sample_mode - will be adjusted in create_accelerator_and_postprocess
        # based on actual Accelerator state (not TrainingArguments which may be inaccurate)
        self._original_grad_sample_mode = self.privacy_args.grad_sample_mode

        dataset_size = len(train_dataset)
        if (
            hasattr(train_dataset, "__iter__")
            and not hasattr(train_dataset, "__len__")
            and privacy_args.poisson_sampling
        ):
            raise ValueError(
                "IterableDataset is not supported by DPTrainer when poisson_sampling is True."
            )

        if (
            args.save_strategy
            and args.save_steps
            and not args.restore_callback_states_from_checkpoint
        ):
            save_strategy = getattr(args, "save_strategy", None)
            save_strategy_value = (
                save_strategy.value if hasattr(save_strategy, "value") else str(save_strategy)
            )
            if str(save_strategy_value) == "no":
                # No checkpointing requested, no callback state restoration needed.
                pass
            else:
                warnings.warn(
                    "Save strategy is set but restore_callback_states_from_checkpoint is false. "
                    "Accountant states will not be restored from the checkpoint leading to the incorrect "
                    "privacy budget estimates. Setting restore_callback_states_from_checkpoint to True"
                )
                args.restore_callback_states_from_checkpoint = True

        sample_rate = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            / dataset_size
        )

        self.privacy_args.precalculate(
            num_samples=dataset_size,
            sample_rate=sample_rate,
            steps=(
                args.max_steps // args.gradient_accumulation_steps
                if args.max_steps and args.max_steps != -1
                else math.ceil(1 / sample_rate) * args.num_train_epochs
            ),
        )

        logger.info(
            f"Using privacy noise multiplier: {self.privacy_args.noise_multiplier}"
        )

        self.dp_callback = DPCallback(
            accountant=self.privacy_args.accountant,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            target_delta=self.privacy_args.target_delta,
            max_epsilon=self.privacy_args.target_epsilon,
        )
        callbacks = callbacks or []
        callbacks.append(self.dp_callback)

        def compute_privacy_metrics(*args, **kwargs):
            if compute_metrics:
                metrics = compute_metrics(*args, **kwargs)
            else:
                metrics = {}

            privacy_metrics = self.dp_callback.get_privacy_metrics()
            metrics.update(privacy_metrics)

            return metrics

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            callbacks=callbacks,
            compute_metrics=compute_privacy_metrics,
            **kwargs,
        )

        self.sample_rate = sample_rate
        self.hooks = None
        self._hooks_attached = False

        # IMPORTANT: do not attach hooks here.
        # When launched via `accelerate launch --use_fsdp` the model wrapping happens
        # later (inside Trainer/Accelerate), and attaching hooks before wrapping can
        # bind to parameters that will be replaced by FSDP.

    def create_accelerator_and_postprocess(self):
        """Override to auto-configure and validate distributed configuration.
        
        This method hooks into the Trainer's Accelerator creation to:
        1. Auto-configure grad_sample_mode based on actual distributed state
        2. Auto-configure FSDP settings for DP compatibility
        3. Validate the final configuration
        
        Auto-configuration follows the same pattern as HF Trainer (modifying
        fsdp_plugin after Accelerator creation), making it a clean integration point.
        
        Note: Currently only LoRA-style fine-tuning is supported with FSDP.
        """
        super().create_accelerator_and_postprocess()
        
        # Now we have access to actual Accelerator state - adjust grad_sample_mode accordingly
        self._configure_grad_sample_mode()
        
        # Auto-configure FSDP for DP training
        if self.is_fsdp_enabled:
            self._auto_configure_fsdp_for_dp()
        
        # Validate the final configuration
        self._validate_distributed_config()
    
    def _configure_grad_sample_mode(self):
        """Configure grad_sample_mode based on actual Accelerator state.
        
        This must be called after Accelerator is created, as TrainingArguments
        may indicate FSDP/distributed based on environment variables, but the
        actual Accelerator configuration may differ (e.g., num_processes=1).
        """
        state = self.accelerator.state
        
        # Determine actual distributed/FSDP state from Accelerator
        is_actually_distributed = state.distributed_type in {
            DistributedType.MULTI_GPU,
            DistributedType.FSDP,
        }
        is_actually_fsdp = state.distributed_type == DistributedType.FSDP
        
        original_mode = self._original_grad_sample_mode
        
        if is_actually_distributed or is_actually_fsdp:
            # Distributed/FSDP mode - need hooks_fsdp
            if original_mode == "hooks":
                self.privacy_args.grad_sample_mode = "hooks_fsdp"
                logger.info(
                    f"Auto-adjusted grad_sample_mode: hooks -> hooks_fsdp "
                    f"(distributed_type={state.distributed_type})"
                )
            elif original_mode == "hooks_fsdp":
                self.privacy_args.grad_sample_mode = "hooks_fsdp"
            else:
                raise ValueError(
                    f"Unsupported grad_sample_mode '{original_mode}' for distributed training "
                    f"(distributed_type={state.distributed_type}). "
                    f"Only 'hooks' and 'hooks_fsdp' are supported."
                )
        else:
            # Single-GPU mode
            if original_mode not in ["hooks", "functorch"]:
                raise ValueError(
                    f"Unsupported grad_sample_mode '{original_mode}' for single-GPU training. "
                    f"Only 'hooks' and 'functorch' are supported."
                )
            self.privacy_args.grad_sample_mode = original_mode
            logger.info(
                f"Single-GPU mode detected (distributed_type={state.distributed_type}), "
                f"using grad_sample_mode: {original_mode}"
            )

    def _auto_configure_fsdp_for_dp(self):
        """Auto-configure FSDP plugin settings for DP training compatibility.
        
        This method modifies the FSDP plugin after Accelerator creation,
        following the same pattern used by HF Trainer itself.
        
        Auto-configurations applied:
        1. cpu_ram_efficient_loading=False (required for DP)
        2. ignored_modules for trainable params (FSDP2 only, required for LoRA)
        
        Note: Currently only LoRA-style fine-tuning is supported with FSDP+DP.
        """
        plugin = self.accelerator.state.fsdp_plugin
        if plugin is None:
            return
        
        fsdp_version = getattr(plugin, "fsdp_version", 1)
        auto_configured = []
        
        # Auto-fix cpu_ram_efficient_loading
        if getattr(plugin, "cpu_ram_efficient_loading", False):
            plugin.cpu_ram_efficient_loading = False
            auto_configured.append("cpu_ram_efficient_loading=False")
            logger.warning(
                "Auto-configured FSDP: cpu_ram_efficient_loading=False "
                "(required for DP training to preserve parameter references)"
            )
        
        # Auto-configure ignored_modules for FSDP2 (required for DP training with LoRA)
        if fsdp_version == 2:
            current_ignored = getattr(plugin, "ignored_modules", None)
            if current_ignored in (None, [], ""):
                ignored_regex = self._detect_trainable_modules_regex()
                
                if ignored_regex:
                    plugin.ignored_modules = ignored_regex
                    auto_configured.append(f"ignored_modules='{ignored_regex}'")
                    logger.info(
                        f"Auto-configured FSDP2: ignored_modules='{ignored_regex}' "
                        f"(excluding trainable modules from sharding for DP compatibility)"
                    )
        
        if auto_configured:
            logger.info(f"FSDP auto-configuration complete: {', '.join(auto_configured)}")
    
    def _detect_trainable_modules_regex(self) -> Optional[str]:
        """Detect trainable module names and generate a regex pattern.
        
        Returns:
            Regex pattern matching trainable module names, or None if detection fails.
        """
        trainable_module_names = set()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Extract module path (everything except the last component which is the param name)
                parts = name.rsplit(".", 1)
                if len(parts) > 1:
                    module_name = parts[0]
                    # Get the leaf module name (last component of the path)
                    leaf_name = module_name.rsplit(".", 1)[-1]
                    trainable_module_names.add(leaf_name)
        
        if not trainable_module_names:
            return None
        
        # Common patterns for LoRA and other adapters
        common_patterns = {"lora", "adapter", "ia3", "prefix", "prompt"}
        detected_patterns = trainable_module_names & common_patterns
        
        if detected_patterns:
            # Use detected common patterns
            pattern = "|".join(f".*{p}.*" for p in sorted(detected_patterns))
        else:
            # Fall back to exact module names
            pattern = "|".join(f".*\\.{name}$" for name in sorted(trainable_module_names))
        
        return pattern

    def _validate_distributed_config(self):
        """Validate that the distributed configuration is supported for DP training.
        
        This method checks the Accelerate configuration and raises clear errors
        if the setup is incompatible with differential privacy training.
        """
        state = self.accelerator.state
        
        # Supported distributed types for DP training
        supported_types = {
            DistributedType.NO,        # Single GPU
            DistributedType.MULTI_GPU, # DDP
            DistributedType.FSDP,      # FSDP (with constraints)
        }
        
        if state.distributed_type not in supported_types:
            raise ValueError(
                f"Distributed type '{state.distributed_type}' is not supported for DP training. "
                f"Supported types: {[t.value for t in supported_types]}. "
                f"Please configure your environment using 'accelerate config' or 'accelerate launch' arguments."
            )
        
        # Check for DataParallel case: multiple GPUs visible but not using distributed training
        # In this case, HF Trainer wraps the model in nn.DataParallel which breaks DP hooks
        if state.distributed_type == DistributedType.NO and self.args.n_gpu > 1:
            raise ValueError(
                f"Multiple GPUs detected (n_gpu={self.args.n_gpu}) but not using distributed training. "
                f"HuggingFace Trainer will wrap the model in nn.DataParallel, which is incompatible "
                f"with Opacus per-sample gradient computation.\n\n"
                f"Please use one of the following options:\n"
                f"  1. Limit to single GPU: CUDA_VISIBLE_DEVICES=0 python your_script.py\n"
                f"  2. Use FSDP2 distributed training:\n"
                f"     accelerate launch --use_fsdp --fsdp_version 2 --num_processes N your_script.py\n"
                f"  3. Use DDP distributed training:\n"
                f"     accelerate launch --multi_gpu --num_processes N your_script.py"
            )
        
        # FSDP-specific validation
        if state.distributed_type == DistributedType.FSDP:
            self._validate_fsdp_config(state)
        
        logger.info(
            f"Distributed configuration validated for DP training: "
            f"distributed_type={state.distributed_type}, num_processes={state.num_processes}"
        )

    def _validate_fsdp_config(self, state):
        """Validate FSDP configuration for DP training.
        
        Args:
            state: AcceleratorState containing FSDP plugin configuration.
            
        Note: Currently only LoRA-style fine-tuning is supported with FSDP+DP.
        """
        plugin = state.fsdp_plugin
        if plugin is None:
            return
        
        fsdp_version = getattr(plugin, "fsdp_version", 1)
        
        # FSDP1 is not supported - it uses FlatParameter which flattens multiple parameters
        # into a single tensor, making per-sample gradient computation impossible.
        # FSDP2 preserves individual parameters and is compatible with Opacus.
        if fsdp_version == 1:
            raise ValueError(
                "FSDP1 is not supported for DP training. FSDP1 uses FlatParameter which "
                "flattens multiple model parameters into a single tensor, making per-sample "
                "gradient computation impossible. Please use FSDP2 instead:\n"
                "  accelerate launch --use_fsdp --fsdp_version 2 --num_processes N your_script.py\n"
                "FSDP2 preserves individual parameters and is compatible with Opacus "
                "per-sample gradient computation."
            )
        
        # Check cpu_ram_efficient_loading (should have been auto-fixed, but verify)
        if getattr(plugin, "cpu_ram_efficient_loading", False):
            raise ValueError(
                "FSDP cpu_ram_efficient_loading=True is not compatible with DP training. "
                "It moves the model to meta device, which invalidates parameter references "
                "needed for per-sample gradient computation. "
                "Please disable it via accelerate launch:\n"
                "  accelerate launch --use_fsdp --fsdp_cpu_ram_efficient_loading false ..."
            )
        
        # Check ignored_modules for FSDP2 (required for DP training with LoRA)
        if fsdp_version == 2:
            ignored_modules = getattr(plugin, "ignored_modules", None)
            if ignored_modules in (None, [], ""):
                warnings.warn(
                    "FSDP2 with DP training requires 'ignored_modules' to be configured "
                    "to exclude trainable parameters from sharding. Auto-detection may have failed. "
                    "You can manually specify ignored modules via accelerate config or by setting "
                    "the FSDP_IGNORED_MODULES environment variable (e.g., '.*lora.*' for LoRA adapters)."
                )
        
        logger.info(
            f"FSDP configuration: fsdp_version={fsdp_version}, "
            f"cpu_ram_efficient_loading={getattr(plugin, 'cpu_ram_efficient_loading', False)}, "
            f"ignored_modules={getattr(plugin, 'ignored_modules', None)}"
        )

    def _get_model_for_hook_attachment(self, model: nn.Module) -> nn.Module:
        """Unwrap common parallel wrappers (DDP/FSDP/Accelerate) to reach the base module."""

        m = model
        # Most wrappers expose the wrapped module as `.module` (DDP, FSDP1).
        for _ in range(8):
            inner = getattr(m, "module", None)
            if isinstance(inner, nn.Module):
                m = inner
            else:
                break

        return m

    def _get_accelerate_fsdp_plugin(self):
        """Best-effort access to Accelerate's FSDP plugin (if present)."""

        accelerator = getattr(self, "accelerator", None)
        state = getattr(accelerator, "state", None)
        return getattr(state, "fsdp_plugin", None)

    def _attach_hooks(self, model=None):
        if self.hooks is None:
            model = model or self.model
            model = self._get_model_for_hook_attachment(model)
            self.hooks = wrap_model(
                model,
                grad_sample_mode=self.privacy_args.grad_sample_mode,
                wrap_model=False,
            )

    def _wrap_model(self, model, training=True, dataloader=None):
        wrapped_model = super()._wrap_model(
            model, training=training, dataloader=dataloader
        )

        # Attach hooks after wrapping so we see the final parameter objects.
        if training and not self._hooks_attached:
            self._attach_hooks(wrapped_model)
            self._hooks_attached = True

        return wrapped_model

    def create_optimizer(self):
        if self.optimizer:
            return self.optimizer

        self.optimizer = super().create_optimizer()

        optim_class = get_optimizer_class(
            clipping=self.privacy_args.clipping,
            distributed=bool(
                self.args.parallel_mode == ParallelMode.DISTRIBUTED
                or (self.args.fsdp and len(self.args.fsdp) > 0)
            ),
            grad_sample_mode=self.privacy_args.grad_sample_mode,
        )

        kwargs = {
            "optimizer": self.optimizer,
            "noise_multiplier": self.privacy_args.noise_multiplier,
            "expected_batch_size": self.args.per_device_train_batch_size,
            "max_grad_norm": self.privacy_args.per_sample_max_grad_norm,
            "loss_reduction": "mean",
        }

        if issubclass(optim_class, AdaClipDPOptimizer):
            kwargs.update(
                {
                    "max_clipbound": self.privacy_args.max_clipbound,
                    "min_clipbound": self.privacy_args.min_clipbound,
                    "clipbound_learning_rate": self.privacy_args.clipbound_learning_rate,
                    "target_unclipped_quantile": self.privacy_args.target_unclipped_quantile,
                    "unclipped_num_std": self.privacy_args.unclipped_num_std,
                }
            )

        self.optimizer = optim_class(**kwargs)

        # Attach privacy accounting hook
        self.optimizer.attach_step_hook(
            self.dp_callback.get_optimizer_callback(sample_rate=self.sample_rate)
        )

        logger.info(f"Created {self.optimizer.__class__.__name__} for grad_sample_mode={self.privacy_args.grad_sample_mode}")

        return self.optimizer

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        data_loader = self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

        if self.privacy_args.poisson_sampling:
            is_fsdp = self.args.fsdp and len(self.args.fsdp) > 0
            data_loader = DPDataLoader.from_data_loader(
                data_loader,
                distributed=bool(
                    self.args.parallel_mode == ParallelMode.DISTRIBUTED or is_fsdp
                ),
            )

        data_loader = wrap_data_loader(
            data_loader=data_loader,
            optimizer=OptimizerProxy(self),
            max_batch_size=self._train_batch_size,
        )

        return data_loader

    def detach_model(self) -> nn.Module:
        """Detach the model from the hooks and return the model."""
        if hasattr(self, "hooks") and self.hooks:
            self.hooks.cleanup()
        return self.model
