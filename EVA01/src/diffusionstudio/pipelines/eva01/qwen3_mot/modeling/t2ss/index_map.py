"""Unified token metadata and exact visibility rules for EVA01 generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch


FAMILY_PADDING = -1
FAMILY_UND = 0
FAMILY_SS = 1
FAMILY_SHAPE = 2
FAMILY_MATERIAL = 3
FAMILY_SS_CONDITION_VIEW = 11
FAMILY_SHAPE_CONDITION_VIEW = 12
FAMILY_MATERIAL_CONDITION_VIEW = 13

FAMILY_CODES: Mapping[str, int] = {
    "und": FAMILY_UND,
    "ss": FAMILY_SS,
    "shape": FAMILY_SHAPE,
    "material": FAMILY_MATERIAL,
    "condition_ss": FAMILY_SS_CONDITION_VIEW,
    "condition_shape": FAMILY_SHAPE_CONDITION_VIEW,
    "condition_material": FAMILY_MATERIAL_CONDITION_VIEW,
    "padding": FAMILY_PADDING,
}
CONDITION_CODE_BY_QUERY_FAMILY: Mapping[int, int] = {
    FAMILY_SS: FAMILY_SS_CONDITION_VIEW,
    FAMILY_SHAPE: FAMILY_SHAPE_CONDITION_VIEW,
    FAMILY_MATERIAL: FAMILY_MATERIAL_CONDITION_VIEW,
}


@dataclass(frozen=True)
class UnifiedTokenMetadata:
    """Metadata for one padded physical token axis."""

    family: torch.Tensor
    sample_id: torch.Tensor
    local_index: torch.Tensor
    valid: torch.Tensor
    position_ids: torch.Tensor

    def validate(self, *, axis_name: str) -> None:
        if self.family.ndim != 2:
            raise ValueError(f"{axis_name}.family must have shape [B,T].")
        expected = self.family.shape
        for name, tensor in (
            ("sample_id", self.sample_id),
            ("local_index", self.local_index),
            ("valid", self.valid),
        ):
            if tensor.shape != expected:
                raise ValueError(
                    f"{axis_name}.{name} must have shape {tuple(expected)}, got {tuple(tensor.shape)}."
                )
        if self.position_ids.shape != (3, *expected):
            raise ValueError(
                f"{axis_name}.position_ids must have shape {(3, *expected)}, "
                f"got {tuple(self.position_ids.shape)}."
            )
        devices = {
            self.family.device,
            self.sample_id.device,
            self.local_index.device,
            self.valid.device,
            self.position_ids.device,
        }
        if len(devices) != 1:
            raise ValueError(f"All {axis_name} metadata tensors must share one device.")
        if not torch.equal(self.valid, self.family.ne(FAMILY_PADDING)):
            raise ValueError(f"{axis_name}.valid must be exactly family != FAMILY_PADDING.")


@dataclass(frozen=True)
class UnifiedIndexMap:
    """Rectangular query/KV layout plus family output slices."""

    task: str
    variant: str
    query: UnifiedTokenMetadata
    key_value: UnifiedTokenMetadata
    query_slices: Dict[str, Tuple[int, int]]
    key_value_slices: Dict[str, Tuple[int, int]]

    def validate(self) -> None:
        self.query.validate(axis_name="query")
        self.key_value.validate(axis_name="key_value")
        if self.query.family.shape[0] != self.key_value.family.shape[0]:
            raise ValueError("Query and KV metadata batch sizes must match.")
        if self.variant not in {"A", "B"}:
            raise ValueError("UnifiedIndexMap variant must be `A` or `B`.")
        if self.task not in {"t2ss", "t2shape", "t2sm"}:
            raise ValueError(f"Unsupported generation task `{self.task}`.")
        for axis_name, slices, length in (
            ("query", self.query_slices, self.query_length),
            ("key_value", self.key_value_slices, self.key_value_length),
        ):
            occupied: list[Tuple[int, int, str]] = []
            for name, (start, end) in slices.items():
                if not (0 <= int(start) <= int(end) <= length):
                    raise ValueError(f"Invalid {axis_name} slice {name}: {(start, end)}.")
                occupied.append((int(start), int(end), name))
            occupied.sort()
            for left, right in zip(occupied, occupied[1:]):
                if left[1] > right[0]:
                    raise ValueError(f"Overlapping {axis_name} slices: {left} and {right}.")

    @property
    def batch_size(self) -> int:
        return int(self.query.family.shape[0])

    @property
    def query_length(self) -> int:
        return int(self.query.family.shape[1])

    @property
    def key_value_length(self) -> int:
        return int(self.key_value.family.shape[1])

    def dense_allow_mask(self) -> torch.Tensor:
        """Materialize the exact small-test oracle; formal forward must not call this."""

        query = self.query
        key_value = self.key_value
        q_family = query.family.unsqueeze(-1)
        kv_family = key_value.family.unsqueeze(-2)
        same_sample = query.sample_id.unsqueeze(-1).eq(key_value.sample_id.unsqueeze(-2))
        valid_pair = query.valid.unsqueeze(-1) & key_value.valid.unsqueeze(-2) & same_sample

        und_query = q_family.eq(FAMILY_UND)
        und_key = kv_family.eq(FAMILY_UND)
        causal = key_value.local_index.unsqueeze(-2).le(query.local_index.unsqueeze(-1))
        allowed = und_query & und_key & causal

        for query_family, visible_native in (
            (FAMILY_SS, (FAMILY_SS,)),
            (FAMILY_SHAPE, (FAMILY_SS, FAMILY_SHAPE)),
            (FAMILY_MATERIAL, (FAMILY_SS, FAMILY_SHAPE, FAMILY_MATERIAL)),
        ):
            condition_code = (
                FAMILY_UND if self.variant == "A" else CONDITION_CODE_BY_QUERY_FAMILY[query_family]
            )
            visible = kv_family.eq(condition_code)
            for native_code in visible_native:
                visible = visible | kv_family.eq(native_code)
            allowed = allowed | (q_family.eq(query_family) & visible)

        padding = q_family.eq(FAMILY_PADDING) & kv_family.eq(FAMILY_PADDING)
        return (valid_pair & allowed) | padding


@dataclass(frozen=True)
class TokenAxis:
    """One unpadded family segment used to assemble a unified axis."""

    name: str
    sample_id: torch.Tensor
    local_index: torch.Tensor
    position_ids: torch.Tensor

    def validate(self) -> None:
        if self.name not in FAMILY_CODES or self.name == "padding":
            raise ValueError(f"Unsupported token-axis family `{self.name}`.")
        if self.sample_id.ndim != 2:
            raise ValueError("TokenAxis.sample_id must have shape [B,T].")
        if self.local_index.shape != self.sample_id.shape:
            raise ValueError("TokenAxis.local_index must match sample_id.")
        if self.position_ids.shape != (3, *self.sample_id.shape):
            raise ValueError("TokenAxis.position_ids must have shape [3,B,T].")
        if len({self.sample_id.device, self.local_index.device, self.position_ids.device}) != 1:
            raise ValueError("TokenAxis tensors must share one device.")


def _assemble_axis(
    axes: Sequence[TokenAxis],
    *,
    padded_length: Optional[int],
) -> Tuple[UnifiedTokenMetadata, Dict[str, Tuple[int, int]]]:
    if not axes:
        raise ValueError("A unified axis requires at least one non-padding segment.")
    for axis in axes:
        axis.validate()
    batch_size = int(axes[0].sample_id.shape[0])
    device = axes[0].sample_id.device
    if any(int(axis.sample_id.shape[0]) != batch_size for axis in axes):
        raise ValueError("All unified token axes must share one physical batch size.")
    slices: Dict[str, Tuple[int, int]] = {}
    family_rows = []
    sample_rows = []
    local_rows = []
    position_rows = []
    offset = 0
    for axis in axes:
        if axis.name in slices:
            raise ValueError(f"Duplicate unified axis segment `{axis.name}`.")
        token_count = int(axis.sample_id.shape[1])
        slices[axis.name] = (offset, offset + token_count)
        family_rows.append(
            torch.full(
                (batch_size, token_count),
                FAMILY_CODES[axis.name],
                device=device,
                dtype=torch.long,
            )
        )
        sample_rows.append(axis.sample_id.to(dtype=torch.long))
        local_rows.append(axis.local_index.to(dtype=torch.long))
        position_rows.append(axis.position_ids.to(dtype=torch.long))
        offset += token_count
    target_length = offset if padded_length is None else int(padded_length)
    if target_length < offset:
        raise ValueError(
            f"Unified axis has {offset} active tokens but padded_length is {target_length}."
        )
    padding_count = target_length - offset
    if padding_count:
        slices["padding"] = (offset, target_length)
        family_rows.append(
            torch.full(
                (batch_size, padding_count),
                FAMILY_PADDING,
                device=device,
                dtype=torch.long,
            )
        )
        sample_rows.append(torch.full((batch_size, padding_count), -1, device=device, dtype=torch.long))
        local_rows.append(
            torch.arange(padding_count, device=device, dtype=torch.long)
            .view(1, -1)
            .expand(batch_size, -1)
        )
        position_rows.append(torch.zeros((3, batch_size, padding_count), device=device, dtype=torch.long))
    family = torch.cat(family_rows, dim=1)
    metadata = UnifiedTokenMetadata(
        family=family,
        sample_id=torch.cat(sample_rows, dim=1),
        local_index=torch.cat(local_rows, dim=1),
        valid=family.ne(FAMILY_PADDING),
        position_ids=torch.cat(position_rows, dim=2),
    )
    return metadata, slices


def build_generation_index_map(
    *,
    task: str,
    variant: str,
    query_axes: Sequence[TokenAxis],
    key_value_axes: Sequence[TokenAxis],
    max_query_tokens: Optional[int] = None,
    max_key_value_tokens: Optional[int] = None,
) -> UnifiedIndexMap:
    """Assemble one slot's rectangular packed query/KV metadata."""

    normalized_task = str(task).strip().lower()
    normalized_variant = str(variant).strip().upper()
    query, query_slices = _assemble_axis(query_axes, padded_length=max_query_tokens)
    key_value, key_value_slices = _assemble_axis(
        key_value_axes,
        padded_length=max_key_value_tokens,
    )
    index_map = UnifiedIndexMap(
        task=normalized_task,
        variant=normalized_variant,
        query=query,
        key_value=key_value,
        query_slices=query_slices,
        key_value_slices=key_value_slices,
    )
    index_map.validate()
    return index_map


def _normalize_sample_ids(
    sample_ids: Optional[torch.Tensor],
    *,
    batch_size: int,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:
    if sample_ids is None:
        return torch.zeros((batch_size, token_count), device=device, dtype=torch.long)
    normalized = sample_ids.to(device=device, dtype=torch.long)
    if normalized.shape != (batch_size, token_count):
        raise ValueError(
            f"sample_ids must have shape {(batch_size, token_count)}, got {tuple(normalized.shape)}."
        )
    return normalized


def build_t2ss_index_map(
    *,
    variant: str,
    und_attention_mask: torch.Tensor,
    und_position_ids: torch.Tensor,
    ss_position_ids: torch.Tensor,
    und_sample_ids: Optional[torch.Tensor] = None,
    ss_sample_ids: Optional[torch.Tensor] = None,
    max_query_tokens: Optional[int] = None,
    max_condition_tokens: Optional[int] = None,
) -> UnifiedIndexMap:
    """Compatibility builder for t2ss tests and the packed formal executor."""

    normalized_variant = str(variant).strip().upper()
    if normalized_variant not in {"A", "B"}:
        raise ValueError("condition variant must be `A` or `B`.")
    if und_attention_mask.ndim != 2:
        raise ValueError("und_attention_mask must have shape [B,T_und].")
    batch_size, und_tokens = und_attention_mask.shape
    if und_position_ids.shape != (3, batch_size, und_tokens):
        raise ValueError("und_position_ids must have shape [3,B,T_und].")
    if ss_position_ids.ndim != 3 or ss_position_ids.shape[:2] != (3, batch_size):
        raise ValueError("ss_position_ids must have shape [3,B,T_ss].")
    if not bool(und_attention_mask.to(dtype=torch.bool).all()):
        # Legacy unit tests may include a masked text token. Drop it from the
        # formal segment only when every physical row has the same valid count.
        counts = und_attention_mask.to(dtype=torch.bool).sum(dim=1)
        if not bool(counts.eq(counts[0]).all()):
            raise ValueError("Packed text rows may not contain per-row padding.")
        valid_indices = und_attention_mask[0].to(dtype=torch.bool)
        und_position_ids = und_position_ids[:, :, valid_indices]
        if und_sample_ids is not None:
            und_sample_ids = und_sample_ids[:, valid_indices]
        und_tokens = int(valid_indices.sum())
    device = und_position_ids.device
    ss_tokens = int(ss_position_ids.shape[2])
    und_samples = _normalize_sample_ids(
        und_sample_ids,
        batch_size=batch_size,
        token_count=und_tokens,
        device=device,
    )
    ss_samples = _normalize_sample_ids(
        ss_sample_ids,
        batch_size=batch_size,
        token_count=ss_tokens,
        device=device,
    )
    und_local = und_position_ids[0].to(dtype=torch.long)
    ss_local = (
        torch.arange(ss_tokens, device=device, dtype=torch.long)
        .view(1, -1)
        .expand(batch_size, -1)
    )
    und_axis = TokenAxis("und", und_samples, und_local, und_position_ids)
    ss_axis = TokenAxis("ss", ss_samples, ss_local, ss_position_ids)
    query_axes = (und_axis, ss_axis)
    if normalized_variant == "A":
        key_value_axes = query_axes
        max_kv = max_query_tokens
    else:
        condition_axis = TokenAxis(
            "condition_ss",
            und_samples,
            und_local,
            und_position_ids,
        )
        key_value_axes = (und_axis, condition_axis, ss_axis)
        max_kv = (
            None
            if max_query_tokens is None
            else int(max_query_tokens) + int(max_condition_tokens or und_tokens)
        )
    return build_generation_index_map(
        task="t2ss",
        variant=normalized_variant,
        query_axes=query_axes,
        key_value_axes=key_value_axes,
        max_query_tokens=max_query_tokens,
        max_key_value_tokens=max_kv,
    )


__all__ = [
    "CONDITION_CODE_BY_QUERY_FAMILY",
    "FAMILY_CODES",
    "FAMILY_MATERIAL",
    "FAMILY_MATERIAL_CONDITION_VIEW",
    "FAMILY_PADDING",
    "FAMILY_SHAPE",
    "FAMILY_SHAPE_CONDITION_VIEW",
    "FAMILY_SS",
    "FAMILY_SS_CONDITION_VIEW",
    "FAMILY_UND",
    "TokenAxis",
    "UnifiedIndexMap",
    "UnifiedTokenMetadata",
    "build_generation_index_map",
    "build_t2ss_index_map",
]
