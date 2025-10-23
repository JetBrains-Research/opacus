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

from typing import Dict, List

import torch
import torch.nn as nn
from opacus.grad_sample import embedding_norm_sample
from torch.distributed._tensor.experimental import implicit_replication

from .utils import register_grad_sampler, register_norm_sampler


@register_grad_sampler(nn.Embedding)
def compute_embedding_grad_sample(
    layer: nn.Embedding, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Embedding`` layer.

    Supports both regular tensors and RowwiseParallel sharded embeddings (TP).
    For RowwiseParallel, the vocabulary is sharded across devices, and we need
    to convert global token indices to local indices for the shard.

    Args:
        layer: Layer
        activations: Activations (token indices)
        backprops: Backpropagations
    """
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        saved = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

        batch_size = activations.shape[0]
        if batch_size == 0:
            ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
            return ret

        # Check if weight is a sharded DTensor (RowwiseParallel)
        from torch.distributed._tensor import DTensor
        is_dtensor = isinstance(layer.weight, DTensor)

        if is_dtensor and layer.weight.placements[0].is_shard(0):
            # RowwiseParallel sharding: vocabulary dimension is sharded across devices
            # Need to convert global token indices to local vocab indices
            import torch.distributed as dist
            from torch.distributed._tensor import distribute_tensor
            from torch.distributed._tensor.placement_types import Shard

            mesh = layer.weight.device_mesh
            rank = dist.get_rank(mesh.get_group())

            # Get local vocab shard info (DTensor.shape returns global shape)
            local_vocab_size = layer.weight.to_local().shape[0]
            vocab_start = rank * local_vocab_size
            vocab_end = vocab_start + local_vocab_size

            # Use implicit_replication to work with local tensors without DTensor dispatch
            with implicit_replication():
                # Convert DTensors to local tensors for computation
                local_activations = activations.to_local() if isinstance(activations, DTensor) else activations
                local_backprops = backprops.to_local() if isinstance(backprops, DTensor) else backprops

                # Get actual device from local weight tensor
                local_weight = layer.weight.to_local()
                device = local_weight.device

                # Mask: True for tokens in this rank's vocab shard
                vocab_mask = (local_activations >= vocab_start) & (local_activations < vocab_end)

                # Convert global vocab indices to local (subtract vocab_start)
                # Masked out indices become 0 (won't affect grad since backprops are masked)
                local_indices = torch.where(
                    vocab_mask,
                    local_activations - vocab_start,
                    torch.zeros_like(local_activations)
                )

                # Prepare indices for scatter_add: [batch, seq*positions, embed_dim]
                scatter_indices = (
                    local_indices.unsqueeze(-1)
                    .expand(*local_indices.shape, layer.embedding_dim)
                    .reshape(batch_size, -1, layer.embedding_dim)
                )

                # Initialize grad_sample for local vocab shard as a regular tensor
                grad_sample = torch.zeros(
                    batch_size,
                    local_vocab_size,
                    layer.embedding_dim,
                    device=device,
                    dtype=local_backprops.dtype
                )

                # Mask backprops: only accumulate gradients for tokens in local vocab
                vocab_mask_expanded = vocab_mask.unsqueeze(-1).expand_as(local_backprops)
                masked_backprops = torch.where(
                    vocab_mask_expanded,
                    local_backprops,
                    torch.zeros_like(local_backprops)
                )

                # Accumulate per-sample gradients via scatter_add on regular tensor
                grad_sample.scatter_add_(
                    1,
                    scatter_indices,
                    masked_backprops.reshape(batch_size, -1, layer.embedding_dim)
                )

            # Convert to DTensor with vocab dimension sharded
            # grad_sample: [batch, local_vocab, embed] -> shard on dim 1 (vocab)
            grad_sample = distribute_tensor(
                grad_sample,
                device_mesh=mesh,
                placements=[Shard(1)]
            )
        else:
            # Regular tensor or replicated DTensor - use existing logic
            # Use implicit_replication to handle mixed Tensor/DTensor operations
            with implicit_replication():
                index = (
                    activations.unsqueeze(-1)
                    .expand(*activations.shape, layer.embedding_dim)
                    .reshape(batch_size, -1, layer.embedding_dim)
                )
                grad_sample = torch.zeros(
                    batch_size,
                    *layer.weight.shape,
                    device=layer.weight.device,
                    dtype=backprops.dtype
                )
                grad_sample.scatter_add_(
                    1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
                )

        torch.backends.cudnn.deterministic = saved
        ret[layer.weight] = grad_sample
    return ret


@register_grad_sampler(nn.EmbeddingBag)
def compute_embeddingbag_gradsampler(layer, inputs, backprops):
    index, offset = inputs
    batch_size = offset.shape[0]

    # Use implicit_replication to handle mixed Tensor/DTensor operations
    with implicit_replication():
        gsm = torch.zeros(
            batch_size,
            layer.num_embeddings,
            layer.embedding_dim,
            device=index.device,
            dtype=backprops.dtype,
        )

        for i in range(batch_size):
            begin = offset[i]
            if i < batch_size - 1:
                end = offset[i + 1]
            else:
                end = index.shape[0]

            if layer.mode == "sum":
                gsm[i][index[begin:end]] += backprops[i]
            elif layer.mode == "mean":
                gsm[i][index[begin:end]] += backprops[i] / (end - begin)

    ret = {}
    ret[layer.weight] = gsm

    return ret


@register_norm_sampler(nn.Embedding)
def compute_embedding_norm_sample(
    layer: nn.Embedding,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """Computes gradient norms for ``nn.Embedding`` layer.

    Args:
      layer: Layer
      activations: Activations
      backprops: Backpropagations

    Returns:
      A dictionary of parameter gradients
    """
    return embedding_norm_sample.compute_embedding_norm_sample(
        layer, activations, backprops
    )
