"""Compact-hidden running-base 3D-RoPE layout for All-Task V2.

The virtual sequence contains pad placeholders for native GEN rows, but those
rows are never embedded by Qwen.  The builder uses the existing MoT position
oracle, then gathers compact UND positions and native family positions back
out of the virtual layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ..attention.rope import build_mot_position_ids
from .all_task_tokens import FAMILY_TOKEN_IDS


BOUNDARY_COMMON = 0
BOUNDARY_FAMILY_CODE: Mapping[str, int] = {
    "ss": 1,
    "shape": 2,
    "material": 3,
}
BOUNDARY_ROLE_CODE: Mapping[str, int] = {
    "condition": 0,
    "source_clean": 1,
    "target_noisy": 2,
}
VIRTUAL_POSITION_ABI = "compact_hidden_running_base_interleaved_3d_rope_v1"


@dataclass(frozen=True)
class AllTaskVirtualStageSpec:
    """Position-only metadata for one real native GEN chunk."""

    name: str
    family: str
    role: str
    sample_id: int
    token_count: int
    coords: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.family not in FAMILY_TOKEN_IDS:
            raise ValueError(f"Unknown virtual stage family {self.family!r}.")
        if self.role not in {"source_clean", "target_noisy"}:
            raise ValueError(f"Unknown virtual stage role {self.role!r}.")
        if int(self.sample_id) < 0 or int(self.token_count) <= 0:
            raise ValueError("Virtual stage IDs and token counts must be positive.")
        if self.family == "ss":
            if int(self.token_count) != 4096 or self.coords is not None:
                raise ValueError("SS virtual stages must be exactly 4096 dense rows.")
        else:
            if self.coords is None:
                raise ValueError("Sparse virtual stages require native XYZ coords.")
            if self.coords.ndim != 2 or tuple(self.coords.shape) != (
                int(self.token_count),
                3,
            ):
                raise ValueError("Sparse virtual stage coords must have shape [N,3].")


@dataclass(frozen=True)
class AllTaskVirtualPositionPlan:
    """Positions for compact UND rows and every real native GEN chunk."""

    compact_position_ids: torch.Tensor
    family_position_ids: Mapping[str, torch.Tensor]
    boundary_family: torch.Tensor
    boundary_role: torch.Tensor
    virtual_sample_lengths: Tuple[int, ...]
    compact_sample_lengths: Tuple[int, ...]
    next_position_bases: Tuple[int, ...]

    def validate(self) -> None:
        total = sum(int(value) for value in self.compact_sample_lengths)
        if self.compact_position_ids.shape != (3, 1, total):
            raise ValueError("Compact position plan must have shape [3,1,T].")
        if self.boundary_family.shape != (1, total):
            raise ValueError("Boundary-family metadata must have shape [1,T].")
        if self.boundary_role.shape != (1, total):
            raise ValueError("Boundary-role metadata must have shape [1,T].")
        if not (
            len(self.virtual_sample_lengths)
            == len(self.compact_sample_lengths)
            == len(self.next_position_bases)
        ):
            raise ValueError("Virtual position per-sample metadata is misaligned.")
        for name, positions in self.family_position_ids.items():
            if positions.ndim != 3 or positions.shape[:2] != (3, 1):
                raise ValueError(f"Family position rows for {name} must be [3,1,N].")


def _stage_coords(spec: AllTaskVirtualStageSpec) -> torch.Tensor:
    if spec.coords is None:
        raise ValueError(f"Sparse stage {spec.name} has no coords.")
    return spec.coords.detach().to(device="cpu", dtype=torch.long).contiguous()


def build_all_task_running_position_plan(
    *,
    token_rows: Sequence[torch.Tensor],
    image_grid_rows: Sequence[Optional[torch.Tensor]],
    stage_specs: Sequence[AllTaskVirtualStageSpec],
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> AllTaskVirtualPositionPlan:
    """Build the virtual oracle and gather positions without hidden copies."""

    if not token_rows or len(token_rows) != len(image_grid_rows):
        raise ValueError("Token rows and image-grid rows must be nonempty and aligned.")
    specs_by_sample: Dict[int, Dict[str, List[AllTaskVirtualStageSpec]]] = {
        sample_id: {family: [] for family in FAMILY_TOKEN_IDS}
        for sample_id in range(len(token_rows))
    }
    names = set()
    for spec in stage_specs:
        spec.validate()
        if spec.name in names:
            raise ValueError(f"Duplicate virtual stage name {spec.name!r}.")
        names.add(spec.name)
        if int(spec.sample_id) not in specs_by_sample:
            raise ValueError(f"Virtual stage {spec.name} references an unknown sample.")
        specs_by_sample[int(spec.sample_id)][spec.family].append(spec)

    compact_parts: List[torch.Tensor] = []
    boundary_family_parts: List[torch.Tensor] = []
    boundary_role_parts: List[torch.Tensor] = []
    family_positions: Dict[str, torch.Tensor] = {}
    virtual_lengths: List[int] = []
    compact_lengths: List[int] = []
    next_bases: List[int] = []
    consumed_names = set()

    start_to_family = {
        int(ids[0]): family for family, ids in FAMILY_TOKEN_IDS.items()
    }
    for sample_id, (token_row, image_grid) in enumerate(
        zip(token_rows, image_grid_rows)
    ):
        row = token_row.detach().to(device="cpu", dtype=torch.long).flatten()
        if row.ndim != 1 or int(row.numel()) <= 0:
            raise ValueError("Every compact UND row must be a nonempty vector.")
        compact_lengths.append(int(row.numel()))
        virtual_values: List[int] = []
        compact_to_virtual: List[int] = []
        spans: List[Tuple[AllTaskVirtualStageSpec, int, int]] = []
        boundary_family = torch.zeros((row.numel(),), dtype=torch.long)
        boundary_role = torch.zeros((row.numel(),), dtype=torch.long)
        occurrence = {family: 0 for family in FAMILY_TOKEN_IDS}
        index = 0
        while index < int(row.numel()):
            token_id = int(row[index].item())
            compact_to_virtual.append(len(virtual_values))
            virtual_values.append(token_id)
            family = start_to_family.get(token_id)
            if family is not None:
                family_specs = specs_by_sample[sample_id][family]
                ordinal = occurrence[family]
                if ordinal >= len(family_specs):
                    raise ValueError(
                        f"Compact row {sample_id} has too many {family} start tokens."
                    )
                spec = family_specs[ordinal]
                occurrence[family] += 1
                _start, end_token_id, pad_token_id = FAMILY_TOKEN_IDS[family]
                if index + 1 >= int(row.numel()) or int(row[index + 1].item()) != int(
                    end_token_id
                ):
                    raise ValueError(
                        f"Compact {family} boundary must be an adjacent start/end pair."
                    )
                role_code = BOUNDARY_ROLE_CODE[spec.role]
                boundary_family[index] = BOUNDARY_FAMILY_CODE[family]
                boundary_family[index + 1] = BOUNDARY_FAMILY_CODE[family]
                boundary_role[index] = role_code
                boundary_role[index + 1] = role_code
                content_start = len(virtual_values)
                virtual_values.extend([int(pad_token_id)] * int(spec.token_count))
                spans.append(
                    (spec, content_start, content_start + int(spec.token_count))
                )
            index += 1

        for family, family_specs in specs_by_sample[sample_id].items():
            if occurrence[family] != len(family_specs):
                raise ValueError(
                    f"Compact row {sample_id} did not consume every {family} stage."
                )

        virtual = torch.tensor(virtual_values, dtype=torch.long).view(1, -1)
        grid = None
        if image_grid is not None:
            grid = image_grid.detach().to(device="cpu", dtype=torch.long).view(-1, 3)
        ordered_specs = [spec for spec, _start, _end in spans]
        ss_grids = [
            torch.tensor((16, 16, 16), dtype=torch.long)
            for spec in ordered_specs
            if spec.family == "ss"
        ]
        shape_coords = [
            _stage_coords(spec)
            for spec in ordered_specs
            if spec.family == "shape"
        ]
        material_coords = [
            _stage_coords(spec)
            for spec in ordered_specs
            if spec.family == "material"
        ]
        virtual_positions, _deltas = build_mot_position_ids(
            input_ids=virtual,
            attention_mask=torch.ones_like(virtual),
            image_grid_thw=grid,
            video_grid_thw=None,
            ss_grid_thw=ss_grids,
            shape_coords=shape_coords,
            tex_coords=material_coords,
            spatial_merge_size=int(spatial_merge_size),
            image_token_id=int(image_token_id),
            video_token_id=int(video_token_id),
            vision_start_token_id=int(vision_start_token_id),
            ss_start_token_id=FAMILY_TOKEN_IDS["ss"][0],
            ss_pad_token_id=FAMILY_TOKEN_IDS["ss"][2],
            ss_end_token_id=FAMILY_TOKEN_IDS["ss"][1],
            shape_start_token_id=FAMILY_TOKEN_IDS["shape"][0],
            shape_pad_token_id=FAMILY_TOKEN_IDS["shape"][2],
            shape_end_token_id=FAMILY_TOKEN_IDS["shape"][1],
            tex_start_token_id=FAMILY_TOKEN_IDS["material"][0],
            tex_pad_token_id=FAMILY_TOKEN_IDS["material"][2],
            tex_end_token_id=FAMILY_TOKEN_IDS["material"][1],
        )
        compact_index = torch.tensor(compact_to_virtual, dtype=torch.long)
        compact_parts.append(virtual_positions.index_select(2, compact_index))
        for spec, start, end in spans:
            positions = virtual_positions[:, :, start:end].contiguous()
            if int(positions.shape[2]) != int(spec.token_count):
                raise RuntimeError("Virtual stage span length drifted.")
            family_positions[spec.name] = positions
            consumed_names.add(spec.name)
        boundary_family_parts.append(boundary_family.view(1, -1))
        boundary_role_parts.append(boundary_role.view(1, -1))
        virtual_lengths.append(int(virtual.shape[1]))
        next_bases.append(int(virtual_positions.max().item()) + 1)

    if consumed_names != names:
        raise RuntimeError("Virtual position plan did not consume every native chunk.")
    result = AllTaskVirtualPositionPlan(
        compact_position_ids=torch.cat(compact_parts, dim=2),
        family_position_ids=family_positions,
        boundary_family=torch.cat(boundary_family_parts, dim=1),
        boundary_role=torch.cat(boundary_role_parts, dim=1),
        virtual_sample_lengths=tuple(virtual_lengths),
        compact_sample_lengths=tuple(compact_lengths),
        next_position_bases=tuple(next_bases),
    )
    result.validate()
    return result


def gather_all_task_reference_position_plan(
    reference: AllTaskVirtualPositionPlan,
    reference_indices: Sequence[torch.Tensor],
) -> AllTaskVirtualPositionPlan:
    """Gather compact null rows while retaining the conditional virtual layout.

    Removed CFG condition tokens do not retain hidden states or K/V rows.  The
    surviving compact UND rows and all native family chunks nevertheless keep
    the positions assigned by the conditional reference sequence.
    """

    reference.validate()
    if len(reference_indices) != len(reference.compact_sample_lengths):
        raise ValueError("CFG reference maps must align with logical samples.")
    absolute_parts: List[torch.Tensor] = []
    compact_lengths: List[int] = []
    offset = 0
    for sample_id, (indices, reference_length) in enumerate(
        zip(reference_indices, reference.compact_sample_lengths)
    ):
        local = indices.detach().to(device="cpu", dtype=torch.long).flatten()
        if int(local.numel()) <= 0:
            raise ValueError(f"Logical sample {sample_id} has an empty CFG reference map.")
        if (
            int(local[0]) < 0
            or int(local[-1]) >= int(reference_length)
            or bool(local[1:].le(local[:-1]).any())
        ):
            raise ValueError("CFG reference maps must be strictly increasing and in range.")
        absolute_parts.append(local + int(offset))
        compact_lengths.append(int(local.numel()))
        offset += int(reference_length)
    absolute = torch.cat(absolute_parts).to(
        device=reference.compact_position_ids.device,
        dtype=torch.long,
    )
    result = AllTaskVirtualPositionPlan(
        compact_position_ids=reference.compact_position_ids.index_select(
            2,
            absolute,
        ),
        family_position_ids=reference.family_position_ids,
        boundary_family=reference.boundary_family.index_select(1, absolute),
        boundary_role=reference.boundary_role.index_select(1, absolute),
        virtual_sample_lengths=reference.virtual_sample_lengths,
        compact_sample_lengths=tuple(compact_lengths),
        next_position_bases=reference.next_position_bases,
    )
    result.validate()
    return result


__all__ = [
    "AllTaskVirtualPositionPlan",
    "AllTaskVirtualStageSpec",
    "BOUNDARY_COMMON",
    "BOUNDARY_FAMILY_CODE",
    "BOUNDARY_ROLE_CODE",
    "VIRTUAL_POSITION_ABI",
    "build_all_task_running_position_plan",
    "gather_all_task_reference_position_plan",
]
