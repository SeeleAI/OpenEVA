"""Family-native condition projection helpers for Unified All-Task V2."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .adapters import linear_with_lora


def validate_native_condition_projections(model: Any) -> None:
    """Fail closed unless every GEN retains its pretrained native cross-KV."""

    for family, base in (
        ("ss", model.ss_xl),
        ("shape", model.shape_model),
        ("material", model.material_model),
    ):
        spec = model.adapters.condition_slot(family, 0).spec
        for index, block in enumerate(base.blocks):
            projection = getattr(getattr(block, "cross_attn", None), "to_kv", None)
            if not isinstance(projection, nn.Linear):
                raise RuntimeError(
                    f"{family} block {index} lost its pretrained native cross-KV."
                )
            if (
                int(projection.in_features) != int(spec.condition_dim)
                or int(projection.out_features) != 2 * int(spec.hidden_size)
            ):
                raise RuntimeError(
                    f"{family} block {index} native cross-KV ABI drifted."
                )


def project_condition_view(
    model: Any,
    *,
    family: str,
    slot: int,
    canonical_condition: torch.Tensor,
    branch_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one shared canonical state through one family's native cross-KV."""

    base = {
        "ss": model.ss_xl,
        "shape": model.shape_model,
        "material": model.material_model,
    }[str(family)]
    block = base.blocks[int(slot)]
    adapters = model.adapters.condition_slot(family, slot)
    context = model.adapters.family_condition_states(family, canonical_condition)
    key_value = linear_with_lora(
        context,
        block.cross_attn.to_kv,
        adapters.cross_kv,
        branch_mask=branch_mask,
    )
    token_count = int(key_value.shape[1])
    key, value = key_value.view(
        1,
        token_count,
        2,
        adapters.spec.num_heads,
        adapters.spec.head_dim,
    ).unbind(dim=2)
    if getattr(block.cross_attn, "qk_rms_norm", False):
        key = block.cross_attn.k_rms_norm(key)
    return adapters.k_bridge(key), adapters.v_bridge(value)


__all__ = ["project_condition_view", "validate_native_condition_projections"]
