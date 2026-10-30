"""Fused projection helpers for EVA01 Unified All-Task V1."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import GenerationRouteCompatibleLoRA

def alltask_batched_lora_triplet(
    features: torch.Tensor,
    deltas: Sequence[nn.Module],
    branch_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate Q/K/V LoRA with one A projection and one batched B."""

    if len(deltas) != 3:
        raise ValueError("All-task QKV requires exactly three LoRA deltas.")
    if any(isinstance(delta, GenerationRouteCompatibleLoRA) for delta in deltas):
        # The generation Q bank uses rank 64 while native K/V retain rank 32.
        return torch.stack(
            tuple(delta(features) for delta in deltas),
            dim=0,
        )
    ranks = {int(delta.rank) for delta in deltas}
    in_features = {int(delta.in_features) for delta in deltas}
    out_features = {int(delta.out_features) for delta in deltas}
    if len(ranks) != 1 or len(in_features) != 1 or len(out_features) != 1:
        raise ValueError("All-task QKV LoRA triplet has incompatible shapes.")
    rank = ranks.pop()
    hidden = out_features.pop()
    if int(features.shape[-1]) != in_features.pop():
        raise ValueError("All-task QKV LoRA input width drifted.")

    a_weight = torch.cat(
        tuple(delta.lora_a.weight for delta in deltas),
        dim=0,
    )
    b_weight = torch.stack(
        tuple(delta.lora_b.weight for delta in deltas),
        dim=0,
    )
    input_dtype = features.dtype
    compute_features = (
        features
        if torch.is_autocast_enabled()
        else features.to(dtype=a_weight.dtype)
    )
    low_rank = F.linear(compute_features, a_weight)
    prefix = tuple(int(value) for value in features.shape[:-1])
    low_rank = low_rank.reshape(*prefix, 3, rank).movedim(-2, 0)
    projected = torch.bmm(
        low_rank.reshape(3, -1, rank),
        b_weight.transpose(1, 2),
    ).reshape(3, *prefix, hidden)
    scales = tuple(float(delta.scaling) for delta in deltas)
    if scales[1:] != scales[:1] * 2:
        raise RuntimeError("All-task fused Q/K/V LoRA scales must match.")
    # A Python scalar avoids the tiny pageable H2D scale tensor that otherwise
    # creates a synchronization point for every family and slot.
    return (projected * scales[0]).to(dtype=input_dtype)


__all__ = ["alltask_batched_lora_triplet"]
