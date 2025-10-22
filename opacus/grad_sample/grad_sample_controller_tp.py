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
GradSampleControllerTP: Tensor Parallelism support for GradSampleController.

This module extends GradSampleController to support Tensor Parallelism using
DTensors, providing proper per-sample gradient norm aggregation across devices.
"""

import logging
import warnings
from typing import Optional

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_controller import GradSampleController
from opacus.utils.module_utils import trainable_parameters
from torch.distributed.tensor import DTensor


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleControllerTP(GradSampleController):
    """
    GradSampleController with Tensor Parallelism support.

    Extends GradSampleController to properly aggregate per-sample gradient norms
    across DTensor-sharded parameters. This implementation does NOT use ghost
    clipping, instead computing full grad_sample tensors and then aggregating
    their norms in a TP-aware manner.

    Key features:
    - Detects DTensor parameters and sets merge_flag for aggregation control
    - Handles RowWiseParallel embeddings (no merge)
    - Handles sharded linear layers (merge with all_reduce)
    - Raises error for replicated parameters (unsupported)

    Example:
        >>> model = MyTransformerModel()
        >>> # Apply tensor parallelism to model
        >>> tp_mesh = init_device_mesh("cuda", (world_size,))
        >>> model = parallelize_module(model, tp_mesh, parallelization_plan)
        >>>
        >>> # Create TP-aware controller
        >>> controller = GradSampleControllerTP(model, batch_first=True, loss_reduction="mean")
        >>>
        >>> # Use with PrivacyEngineGradSampleController
        >>> privacy_engine = PrivacyEngineGradSampleController()
        >>> controller, optimizer, dataloader = privacy_engine.make_private(
        ...     module=model,
        ...     optimizer=optimizer,
        ...     data_loader=dataloader,
        ...     grad_sample_mode="tp",
        ...     return_controller=True
        ... )
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
            m: nn.Module to attach hooks to (should already have TP applied)
            batch_first: Flag to indicate if the input tensor has batch dimension first
            loss_reduction: "sum" or "mean" for loss aggregation
            strict: If True, validate that module has no unsupported buffers
            force_functorch: If True, use functorch for all layers

        Raises:
            NotImplementedError: If module has replicated DTensor parameters
        """
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )
        self.set_pattern_param_sample_norm_sum()
        warnings.warn(
            "Opacus TP support is currently in beta. "
            "Custom model output placements may cause unexpected behavior."
        )

    def set_pattern_param_sample_norm_sum(self):
        """
        Investigate tensor parallelism state of model parameters and set merge_flag.

        The merge_flag determines whether per-sample gradient norms should be
        aggregated across devices using all_reduce:

        - merge_flag=True: Parameter is a sharded DTensor, all ranks contribute
        - merge_flag=False: Only rank 0 contributes (non-DTensor, RowWise Embedding, etc.)

        Exceptions where we don't merge (merge_flag=False):
        1. The parameter is not a DTensor
        2. The parameter belongs to nn.Embedding under RowWiseParallel
        3. The parameter is a replicated DTensor (raises NotImplementedError)

        Raises:
            NotImplementedError: If any parameter is a replicated DTensor
        """
        for module in self.iterate_submodules(self.module):
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                # Check if parameter is a DTensor
                if type(param) is not torch.distributed.tensor.DTensor:
                    param.merge_flag = False
                    logger.debug(f"Parameter {name} is not a DTensor, merge_flag=False")
                    continue

                # Check for Embedding with RowWiseParallel (sharded on dim 0)
                if type(module) is torch.nn.Embedding and param.placements[0].is_shard(
                    0
                ):
                    param.merge_flag = False
                    logger.debug(
                        f"Parameter {name} is Embedding with RowWiseParallel, merge_flag=False"
                    )
                    continue

                # Check for replicated parameters (unsupported)
                if param.placements[0].is_replicate():
                    param.merge_flag = False
                    raise NotImplementedError(
                        f"Parameter {name} is a replicated DTensor, which is not supported. "
                        "This is due to constraints of FSDPOptimizer. "
                        "Please disable nn.Linear.bias or configure it to be sharded."
                    )

                # Default: sharded DTensor, merge across ranks
                param.merge_flag = True
                logger.debug(f"Parameter {name} is sharded DTensor, merge_flag=True")

    def get_per_sample_norms(self) -> torch.Tensor:
        """
        Compute per-sample gradient norms with TP-aware aggregation.

        This method aggregates per-sample gradient norms across all parameters,
        taking into account the tensor parallelism configuration. Parameters with
        merge_flag=True contribute from all ranks, while merge_flag=False parameters
        only contribute from rank 0.

        Returns:
            torch.Tensor: Per-sample gradient norms of shape [batch_size]

        Note:
            This method assumes grad_sample has been populated on all parameters
            via the standard GradSampleController hooks (inherited from base class).
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed not initialized. GradSampleControllerTP requires "
                "torch.distributed to be initialized."
            )

        current_rank = torch.distributed.get_rank()

        # Collect per-sample norms from all parameters
        param_norms = []

        for _, param in trainable_parameters(self.module):
            if param.grad_sample is None:
                continue

            # Convert DTensor to local for reshape operation
            # Reshape on sharded dimensions requires redistribution, so convert first
            grad_sample = param.grad_sample
            if isinstance(grad_sample, DTensor):
                grad_sample = grad_sample.to_local()

            # Compute per-sample norm for this parameter: shape [batch_size]
            per_sample_norm = grad_sample.reshape(len(grad_sample), -1).norm(2, dim=-1)

            # Apply merge_flag logic
            if not param.merge_flag and current_rank != 0:
                # Non-mergeable parameters: only rank 0 contributes
                # Other ranks contribute zeros
                param_norms.append(torch.zeros_like(per_sample_norm))
            else:
                # Mergeable parameters: all ranks contribute
                # Non-mergeable on rank 0: only rank 0 contributes
                param_norms.append(per_sample_norm)

        if len(param_norms) == 0:
            raise ValueError(
                "No parameters with grad_sample found. "
                "Ensure backward pass has been executed."
            )

        # Stack norms and compute total norm across parameters
        # param_norms: list of [batch_size] tensors
        # stacked: [num_params, batch_size]
        # after norm(2, dim=0): [batch_size] - norm across parameters for each sample
        stacked_norms = torch.stack(param_norms, dim=0)
        per_sample_total_norm = stacked_norms.norm(2, dim=0)

        # Square before all_reduce to get sum of squared norms
        squared_norm_sample = per_sample_total_norm.square()

        # Convert DTensor to local tensor for all_reduce
        # DTensors use their own distributed operations, but we need standard torch.distributed.all_reduce
        if isinstance(squared_norm_sample, DTensor):
            squared_norm_sample = squared_norm_sample.to_local()

        # All-reduce to aggregate across ranks
        torch.distributed.all_reduce(
            squared_norm_sample, op=torch.distributed.ReduceOp.SUM
        )

        # Return square root for final per-sample norms
        return squared_norm_sample.sqrt()
