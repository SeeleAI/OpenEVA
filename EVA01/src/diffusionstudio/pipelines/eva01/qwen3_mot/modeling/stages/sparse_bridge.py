"""Helpers that translate between SparseTensor stage batches and flattened runtime views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import torch

from ...trellis2.modules.sparse import SparseTensor


@dataclass(frozen=True)
class SparseStageRestoreSpec:
    """Metadata required to restore flattened sparse stage outputs into SparseTensor batches."""
    mode: Literal["tensor", "tensor_list", "sparse_tensor"]


def get_trellis2_sparse_tensor_class():
    """Return the canonical TRELLIS.2 SparseTensor class used by the runtime."""
    return SparseTensor


def is_trellis2_sparse_tensor(value: Any) -> bool:
    """Check whether a value is the canonical TRELLIS.2 SparseTensor wrapper."""
    return isinstance(value, SparseTensor)


def _normalize_coord_batch(
    coords: torch.Tensor,
    *,
    stage_name: str,
) -> torch.Tensor:
    """Normalize coord batch."""
    if coords.ndim != 2 or coords.shape[-1] not in (3, 4):
        raise ValueError(
            f"{stage_name} sparse coords must have shape [N, 3] or [N, 4], got {tuple(coords.shape)}."
        )
    if coords.shape[-1] == 4:
        coords = coords[:, 1:]
    return coords.to(dtype=torch.int64)


def _split_sparse_tensor_batches(
    sparse_tensor: Any,
    *,
    stage_name: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Split sparse tensor batches."""
    feats_list, coords_list = sparse_tensor.to_tensor_list()
    if len(feats_list) != len(coords_list):
        raise ValueError(
            f"{stage_name} SparseTensor feats/coords batch sizes do not match: "
            f"{len(feats_list)} vs {len(coords_list)}."
        )
    normalized_coords = [_normalize_coord_batch(coords, stage_name=stage_name) for coords in coords_list]
    return list(feats_list), normalized_coords


def build_sparse_stage_tensor_from_batches(
    feat_batches: Sequence[torch.Tensor],
    coord_batches: Sequence[torch.Tensor],
    *,
    stage_name: str,
):
    """Build one SparseTensor batch from per-block sparse coords and features."""
    if len(feat_batches) != len(coord_batches):
        raise ValueError(
            f"{stage_name} sparse tensor rebuild requires matching feat/coord batch sizes, got "
            f"{len(feat_batches)} vs {len(coord_batches)}."
        )
    if not feat_batches:
        raise ValueError(f"{stage_name} sparse tensor rebuild requires at least one batch.")

    SparseTensor = get_trellis2_sparse_tensor_class()
    batched_coords: List[torch.Tensor] = []
    for batch_index, (feats, coords) in enumerate(zip(feat_batches, coord_batches)):
        normalized_coords = _normalize_coord_batch(coords, stage_name=stage_name)
        if feats.ndim != 2:
            raise ValueError(f"{stage_name} sparse feats batch must be 2-D [N, C], got {tuple(feats.shape)}.")
        if feats.shape[0] != normalized_coords.shape[0]:
            raise ValueError(
                f"{stage_name} sparse feats batch token count ({feats.shape[0]}) must match coord count "
                f"({normalized_coords.shape[0]})."
            )
        batch_column = torch.full(
            (normalized_coords.shape[0], 1),
            batch_index,
            dtype=torch.int32,
            device=normalized_coords.device,
        )
        batched_coords.append(torch.cat([batch_column, normalized_coords.to(dtype=torch.int32)], dim=1))

    return SparseTensor(
        feats=torch.cat(list(feat_batches), dim=0),
        coords=torch.cat(batched_coords, dim=0),
    )


def split_sparse_stage_inputs(
    stage_inputs: Optional[Union[torch.Tensor, List[torch.Tensor], Any]],
    *,
    stage_coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    stage_name: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], SparseStageRestoreSpec]:
    """Split a SparseTensor batch into per-block coord and feature views."""
    if stage_inputs is None:
        return [], [], SparseStageRestoreSpec(mode="tensor_list")

    if is_trellis2_sparse_tensor(stage_inputs):
        feat_batches, coord_batches = _split_sparse_tensor_batches(stage_inputs, stage_name=stage_name)
        return feat_batches, coord_batches, SparseStageRestoreSpec(mode="sparse_tensor")

    if isinstance(stage_inputs, torch.Tensor):
        if stage_inputs.ndim != 2:
            raise ValueError(f"{stage_name} sparse feats must be 2-D [N, C], got {tuple(stage_inputs.shape)}.")
        coord_batches: List[torch.Tensor] = []
        if stage_coords is not None:
            if not isinstance(stage_coords, torch.Tensor):
                raise TypeError(f"{stage_name}_coords must be a tensor when {stage_name}_feats is a tensor.")
            coord_batches = [_normalize_coord_batch(stage_coords, stage_name=stage_name)]
        return [stage_inputs], coord_batches, SparseStageRestoreSpec(mode="tensor")

    if not isinstance(stage_inputs, list):
        raise TypeError(
            f"{stage_name} sparse stage inputs must be a tensor, list of tensors, or SparseTensor, "
            f"got {type(stage_inputs)!r}."
        )
    if not stage_inputs:
        return [], [], SparseStageRestoreSpec(mode="tensor_list")

    feat_batches = []
    for index, item in enumerate(stage_inputs):
        if not isinstance(item, torch.Tensor) or item.ndim != 2:
            raise ValueError(
                f"{stage_name} sparse feats item at index {index} must be a 2-D tensor, got {type(item)!r}."
            )
        feat_batches.append(item)

    coord_batches = []
    if stage_coords is not None:
        if not isinstance(stage_coords, list):
            raise TypeError(f"{stage_name}_coords must be a list when {stage_name}_feats is a list.")
        if len(stage_coords) != len(stage_inputs):
            raise ValueError(
                f"{stage_name}_coords batch size ({len(stage_coords)}) must match "
                f"{stage_name}_feats batch size ({len(stage_inputs)})."
            )
        coord_batches = [_normalize_coord_batch(coords, stage_name=stage_name) for coords in stage_coords]
    return feat_batches, coord_batches, SparseStageRestoreSpec(mode="tensor_list")


def restore_sparse_stage_batches(
    feat_batches: Sequence[torch.Tensor],
    *,
    restore_spec: SparseStageRestoreSpec,
    stage_name: str,
    coord_batches: Optional[Sequence[torch.Tensor]] = None,
):
    """Restore per-block sparse predictions into canonical SparseTensor batches."""
    if restore_spec.mode == "tensor":
        if len(feat_batches) != 1:
            raise ValueError(
                f"{stage_name} tensor restore expected exactly one batch, got {len(feat_batches)}."
            )
        return feat_batches[0]
    if restore_spec.mode == "tensor_list":
        return list(feat_batches)
    if restore_spec.mode == "sparse_tensor":
        if coord_batches is None:
            raise ValueError(f"{stage_name} sparse tensor restore requires coord_batches.")
        return build_sparse_stage_tensor_from_batches(
            feat_batches,
            coord_batches,
            stage_name=stage_name,
        )
    raise ValueError(f"Unsupported sparse restore mode: {restore_spec.mode!r}.")


__all__ = [
    "SparseStageRestoreSpec",
    "build_sparse_stage_tensor_from_batches",
    "get_trellis2_sparse_tensor_class",
    "is_trellis2_sparse_tensor",
    "restore_sparse_stage_batches",
    "split_sparse_stage_inputs",
]
