"""Token layout and direct BlockMask ABI for EVA01 Unified All-Task V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ...trellis2.modules import sparse as sp
from .all_task_adapters import ALL_TASK_FAMILIES
from .flex_only import DirectBlockMaskStats, require_flex_attention


ROLE_NONE = 0
ROLE_SOURCE = 1
ROLE_TARGET = 2
ROLE_CODE = {
    "condition": ROLE_NONE,
    "source_clean": ROLE_SOURCE,
    "target_noisy": ROLE_TARGET,
}
FAMILY_CODE = {
    "und": 0,
    "ss": 1,
    "shape": 2,
    "material": 3,
    "condition_ss": 4,
    "condition_shape": 5,
    "condition_material": 6,
    "padding": -1,
}


def condition_family_code(family: str) -> int:
    return FAMILY_CODE[f"condition_{str(family)}"]


@dataclass(frozen=True)
class AllTaskTokenAxis:
    name: str
    family: str
    role: int
    sample_id: torch.Tensor
    local_index: torch.Tensor
    position_ids: torch.Tensor

    def validate(self) -> None:
        if self.family not in FAMILY_CODE or self.family == "padding":
            raise ValueError(f"Unknown all-task family {self.family}.")
        if self.sample_id.ndim != 2 or self.local_index.shape != self.sample_id.shape:
            raise ValueError("All-task axis IDs must have shape [1,T] and match.")
        if self.position_ids.shape != (3, *self.sample_id.shape):
            raise ValueError("All-task axis position IDs must have shape [3,1,T].")
        if int(self.role) not in {ROLE_NONE, ROLE_SOURCE, ROLE_TARGET}:
            raise ValueError("Invalid all-task chunk role.")


@dataclass(frozen=True)
class AllTaskTokenMetadata:
    family: torch.Tensor
    role: torch.Tensor
    sample_id: torch.Tensor
    local_index: torch.Tensor
    valid: torch.Tensor
    position_ids: torch.Tensor

    def validate(self, name: str) -> None:
        shape = self.family.shape
        if self.family.ndim != 2 or self.role.shape != shape or self.sample_id.shape != shape:
            raise ValueError(f"{name} metadata shapes do not agree.")
        if self.local_index.shape != shape or self.valid.shape != shape:
            raise ValueError(f"{name} metadata local/valid shape mismatch.")
        if self.position_ids.shape != (3, *shape):
            raise ValueError(f"{name} position IDs have the wrong shape.")
        valid_matches_family = self.valid.eq(
            self.family.ne(FAMILY_CODE["padding"])
        ).all()
        if valid_matches_family.device.type == "cuda":
            torch._assert_async(
                valid_matches_family,
                f"{name} valid mask is not derived from padding family.",
            )
        elif not bool(valid_matches_family):
            raise ValueError(f"{name} valid mask is not derived from padding family.")


@dataclass(frozen=True)
class AllTaskIndexMap:
    query: AllTaskTokenMetadata
    key_value: AllTaskTokenMetadata
    query_slices: Mapping[str, Tuple[int, int]]
    key_value_slices: Mapping[str, Tuple[int, int]]
    query_length: int
    key_value_length: int

    def validate(self) -> None:
        self.query.validate("query")
        self.key_value.validate("key_value")
        for slices, length in (
            (self.query_slices, self.query_length),
            (self.key_value_slices, self.key_value_length),
        ):
            spans = sorted(
                (int(start), int(end), name)
                for name, (start, end) in slices.items()
            )
            if spans and (spans[0][0] < 0 or spans[-1][1] > int(length)):
                raise ValueError("All-task index slice exceeds axis length.")
            for left, right in zip(spans, spans[1:]):
                if left[1] > right[0]:
                    raise ValueError(f"Overlapping all-task slices: {left}, {right}.")

    def dense_allow_mask(self) -> torch.Tensor:
        """Small dense oracle used only by tests and planner probes."""

        query = self.query
        key = self.key_value
        same = query.sample_id.unsqueeze(-1).eq(key.sample_id.unsqueeze(-2))
        valid = query.valid.unsqueeze(-1) & key.valid.unsqueeze(-2)
        query_family = query.family.unsqueeze(-1)
        key_family = key.family.unsqueeze(-2)
        query_role = query.role.unsqueeze(-1)
        key_role = key.role.unsqueeze(-2)
        allowed = (
            query_family.eq(FAMILY_CODE["und"])
            & key_family.eq(FAMILY_CODE["und"])
            & key.local_index.unsqueeze(-2).le(query.local_index.unsqueeze(-1))
        )
        for family in ALL_TASK_FAMILIES:
            code = FAMILY_CODE[family]
            condition = key_family.eq(condition_family_code(family))
            native = key_family.eq(code)
            source = query_role.eq(ROLE_SOURCE) & key_role.eq(ROLE_SOURCE) & native
            target = query_role.eq(ROLE_TARGET) & native & key_role.ge(ROLE_SOURCE)
            source_boundary = (
                query_role.eq(ROLE_SOURCE)
                & condition
                & key_role.eq(ROLE_SOURCE)
            )
            target_condition = query_role.eq(ROLE_TARGET) & condition
            allowed = allowed | (
                query_family.eq(code)
                & (
                    source
                    | target
                    | source_boundary
                    | target_condition
                )
            )
        padding = query_family.eq(FAMILY_CODE["padding"]) & key_family.eq(
            FAMILY_CODE["padding"]
        )
        return (valid & same & allowed) | padding


def _axis_metadata(
    axes: Sequence[AllTaskTokenAxis],
    padded: int,
) -> Tuple[AllTaskTokenMetadata, Dict[str, Tuple[int, int]]]:
    if not axes:
        raise ValueError("At least one all-task axis is required.")
    for axis in axes:
        axis.validate()
    device = axes[0].sample_id.device
    family_rows: List[torch.Tensor] = []
    role_rows: List[torch.Tensor] = []
    sample_rows: List[torch.Tensor] = []
    local_rows: List[torch.Tensor] = []
    position_rows: List[torch.Tensor] = []
    slices: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for axis in axes:
        count = int(axis.sample_id.shape[1])
        if axis.name in slices:
            raise ValueError(f"Duplicate all-task axis name {axis.name}.")
        slices[axis.name] = (offset, offset + count)
        family_rows.append(
            torch.full(
                (1, count),
                FAMILY_CODE[axis.family],
                device=device,
                dtype=torch.long,
            )
        )
        role_rows.append(
            torch.full((1, count), int(axis.role), device=device, dtype=torch.long)
        )
        sample_rows.append(axis.sample_id.to(device=device, dtype=torch.long))
        local_rows.append(axis.local_index.to(device=device, dtype=torch.long))
        position_rows.append(axis.position_ids.to(device=device, dtype=torch.long))
        offset += count
    if int(padded) < offset:
        raise ValueError(f"All-task active tokens {offset} exceed padded length {padded}.")
    if int(padded) > offset:
        count = int(padded) - offset
        slices["padding"] = (offset, int(padded))
        family_rows.append(
            torch.full(
                (1, count),
                FAMILY_CODE["padding"],
                device=device,
                dtype=torch.long,
            )
        )
        role_rows.append(torch.zeros((1, count), device=device, dtype=torch.long))
        sample_rows.append(
            torch.full((1, count), -1, device=device, dtype=torch.long)
        )
        local_rows.append(
            torch.arange(count, device=device, dtype=torch.long).view(1, -1)
        )
        position_rows.append(
            torch.zeros((3, 1, count), device=device, dtype=torch.long)
        )
    family = torch.cat(family_rows, dim=1)
    return (
        AllTaskTokenMetadata(
            family=family,
            role=torch.cat(role_rows, dim=1),
            sample_id=torch.cat(sample_rows, dim=1),
            local_index=torch.cat(local_rows, dim=1),
            valid=family.ne(FAMILY_CODE["padding"]),
            position_ids=torch.cat(position_rows, dim=2),
        ),
        slices,
    )


def build_all_task_index_map(
    *,
    query_axes: Sequence[AllTaskTokenAxis],
    key_value_axes: Sequence[AllTaskTokenAxis],
    max_query_tokens: int,
    max_key_value_tokens: int,
) -> AllTaskIndexMap:
    query, query_slices = _axis_metadata(query_axes, int(max_query_tokens))
    key_value, key_value_slices = _axis_metadata(
        key_value_axes,
        int(max_key_value_tokens),
    )
    result = AllTaskIndexMap(
        query=query,
        key_value=key_value,
        query_slices=query_slices,
        key_value_slices=key_value_slices,
        query_length=int(max_query_tokens),
        key_value_length=int(max_key_value_tokens),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class AllTaskFamilyChunk:
    name: str
    family: str
    role: str
    sample_id: int
    value: Union[torch.Tensor, sp.SparseTensor]
    timestep: torch.Tensor
    concat_shape: Optional[sp.SparseTensor] = None
    position_offset: int = 0
    chunk_id: str = ""
    position_ids: Optional[torch.Tensor] = None
    branch: str = "gen"
    task_kind: str = "generation"


@dataclass(frozen=True)
class AllTaskPackedBatch:
    input_ids: torch.Tensor
    und_sample_ids: torch.Tensor
    und_position_ids: torch.Tensor
    logical_sample_ids: Tuple[str, ...]
    family_chunks: Mapping[str, Tuple[AllTaskFamilyChunk, ...]]
    max_unified_tokens: int
    max_key_value_tokens: int
    und_position_offsets: Tuple[int, ...] = ()
    cpu_value_checks_passed: bool = False
    condition_modes: Tuple[str, ...] = ()
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    image_sample_ids: Optional[torch.Tensor] = None
    visual_pos_mask: Optional[torch.Tensor] = None
    visual_token_indices: Optional[torch.Tensor] = None
    und_token_kinds: Optional[torch.Tensor] = None
    point_embeds: Optional[torch.Tensor] = None
    point_clouds: Optional[torch.Tensor] = None
    point_cloud_sample_ids: Optional[torch.Tensor] = None
    point_pos_mask: Optional[torch.Tensor] = None
    point_token_indices: Optional[torch.Tensor] = None
    condition_pos_mask: Optional[torch.Tensor] = None
    condition_token_indices: Optional[torch.Tensor] = None
    und_boundary_family: Optional[torch.Tensor] = None
    und_boundary_role: Optional[torch.Tensor] = None
    virtual_position_plan_passed: bool = False
    virtual_sample_lengths: Tuple[int, ...] = ()
    assistant_pos_mask: Optional[torch.Tensor] = None
    qa_query_pos_mask: Optional[torch.Tensor] = None
    qa_query_token_indices: Optional[torch.Tensor] = None
    qa_label_ids: Optional[torch.Tensor] = None
    qa_label_slices: Tuple[Tuple[int, int], ...] = ()
    record_ids: Tuple[str, ...] = ()
    pack_id: Optional[int] = None
    branch_by_sample: Tuple[str, ...] = ()

    def validate(self) -> None:
        if self.input_ids.shape[0] != 1 or self.und_sample_ids.shape != self.input_ids.shape:
            raise ValueError(
                "All-task physical batch must be [1,T] with matching sample IDs."
            )
        if self.und_position_ids.shape != (3, 1, self.input_ids.shape[1]):
            raise ValueError("All-task UND position IDs must be [3,1,T].")
        if not self.logical_sample_ids:
            raise ValueError("All-task cart must contain at least one logical bundle.")
        if self.branch_by_sample and len(self.branch_by_sample) != len(self.logical_sample_ids):
            raise ValueError("branch_by_sample must align with logical_sample_ids.")
        if any(str(value) != "gen" for value in self.branch_by_sample):
            raise ValueError("branch_by_sample values must be `gen`.")
        if self.und_position_offsets and len(self.und_position_offsets) != len(
            self.logical_sample_ids
        ):
            raise ValueError("UND position offsets must align with logical samples.")
        if self.virtual_sample_lengths and len(self.virtual_sample_lengths) != len(
            self.logical_sample_ids
        ):
            raise ValueError("Virtual sample lengths must align with logical samples.")
        if (self.und_boundary_family is None) != (self.und_boundary_role is None):
            raise ValueError("Boundary family and role metadata must be provided together.")
        if self.und_boundary_family is not None:
            if (
                self.und_boundary_family.shape != self.input_ids.shape
                or self.und_boundary_role is None
                or self.und_boundary_role.shape != self.input_ids.shape
            ):
                raise ValueError("Boundary metadata must match compact input_ids.")
            if self.und_boundary_family.dtype != torch.long:
                raise ValueError("Boundary family metadata must use torch.long.")
            if self.und_boundary_role.dtype != torch.long:
                raise ValueError("Boundary role metadata must use torch.long.")
            if bool((self.und_boundary_family < 0).any()) or bool(
                (self.und_boundary_family > FAMILY_CODE["material"]).any()
            ):
                raise ValueError("Boundary family metadata is outside the V2 ABI.")
            if bool((self.und_boundary_role < ROLE_NONE).any()) or bool(
                (self.und_boundary_role > ROLE_TARGET).any()
            ):
                raise ValueError("Boundary role metadata is outside the V2 ABI.")
        for family, chunks in self.family_chunks.items():
            if family not in ALL_TASK_FAMILIES:
                raise ValueError(f"Unknown all-task family {family}.")
            for chunk in chunks:
                if chunk.branch != "gen":
                    raise ValueError(f"Invalid branch on chunk {chunk.name!r}.")
                if chunk.task_kind and not (
                    chunk.task_kind == "generation"
                ):
                    raise ValueError(f"Invalid task_kind on chunk {chunk.name!r}.")
        if self.max_unified_tokens % 128 or self.max_key_value_tokens % 128:
            raise ValueError("All-task token budgets must be multiples of 128.")
        if self.point_pos_mask is not None:
            self._validate_point_inputs()
        elif self.point_embeds is not None or self.point_clouds is not None:
            raise ValueError("Point values require a point placeholder mask.")
        elif self.point_token_indices is not None:
            raise ValueError("Point token indices require a point placeholder mask.")
        self._validate_static_indices()
        if self.qa_query_pos_mask is not None and self.qa_label_ids is None:
            raise ValueError("QA query positions require labels.")
        if (
            self.qa_label_ids is not None
            and not self.cpu_value_checks_passed
            and int(self.qa_label_ids.numel())
            != int(
                self.qa_query_pos_mask.sum()
                if self.qa_query_pos_mask is not None
                else 0
            )
        ):
            raise ValueError("QA labels do not match query positions.")
        if self.qa_label_slices:
            cursor = 0
            for start, end in self.qa_label_slices:
                if int(start) != cursor or int(end) <= int(start):
                    raise ValueError("QA label slices must be positive and contiguous.")
                cursor = int(end)
            if self.qa_label_ids is None or cursor != int(self.qa_label_ids.numel()):
                raise ValueError("QA label slices do not cover all QA labels.")
        if (
            not self.cpu_value_checks_passed
            and self.assistant_pos_mask is not None
            and self.condition_pos_mask is not None
            and bool((self.assistant_pos_mask & self.condition_pos_mask).any())
        ):
            raise ValueError(
                "Assistant answer tokens must never enter the condition view."
            )

    def _validate_point_inputs(self) -> None:
        assert self.point_pos_mask is not None
        if self.point_pos_mask.shape != self.input_ids.shape:
            raise ValueError("Point mask must match input_ids.")
        if (self.point_embeds is None) == (self.point_clouds is None):
            raise ValueError(
                "Provide exactly one of frozen point embeddings or raw point clouds."
            )
        if self.point_embeds is not None:
            if self.point_embeds.shape[0] != int(self.point_pos_mask.sum()):
                raise ValueError(
                    "Point embedding count does not match point placeholder count."
                )
            if self.point_embeds.ndim != 2 or self.point_embeds.shape[-1] != 384:
                raise ValueError(
                    "All-task point embeddings must be frozen PointBERT tokens with width 384."
                )
        if self.point_clouds is not None:
            if self.point_clouds.ndim != 3 or tuple(self.point_clouds.shape[1:]) != (
                8192,
                6,
            ):
                raise ValueError("Raw mesh inputs must have shape [P,8192,6].")
            if (
                not self.cpu_value_checks_passed
                and int(self.point_pos_mask.sum())
                != int(self.point_clouds.shape[0]) * 513
            ):
                raise ValueError(
                    "Each raw point cloud must own exactly 513 point placeholders."
                )
            if (
                self.point_cloud_sample_ids is None
                or self.point_cloud_sample_ids.shape != (self.point_clouds.shape[0],)
            ):
                raise ValueError(
                    "Raw point clouds require one logical sample ID each."
                )
        if self.point_token_indices is not None:
            self._validate_index(
                self.point_token_indices,
                self.point_pos_mask,
                "Point",
            )

    def _validate_static_indices(self) -> None:
        if self.visual_token_indices is not None:
            if self.visual_pos_mask is None:
                raise ValueError(
                    "Visual token indices require a visual placeholder mask."
                )
            self._validate_index(
                self.visual_token_indices,
                self.visual_pos_mask,
                "Visual",
            )
        if self.condition_pos_mask is not None:
            if self.condition_pos_mask.shape != self.input_ids.shape:
                raise ValueError("Condition mask must match input_ids.")
            if self.condition_pos_mask.dtype != torch.bool:
                raise ValueError("Condition mask must be boolean.")
        if self.condition_token_indices is not None:
            if self.condition_token_indices.ndim != 1:
                raise ValueError("Condition token indices must be one-dimensional.")
            if self.condition_token_indices.dtype != torch.long:
                raise ValueError("Condition token indices must use torch.long.")
            if (
                not self.cpu_value_checks_passed
                and not torch.equal(
                    self.condition_token_indices,
                    self._condition_indices_from_masks(),
                )
            ):
                raise ValueError(
                    "Condition token indices do not match the condition mask."
                )
        if (
            self.assistant_pos_mask is not None
            and self.assistant_pos_mask.shape != self.input_ids.shape
        ):
            raise ValueError("Assistant mask must match input_ids.")
        if (
            self.qa_query_pos_mask is not None
            and self.qa_query_pos_mask.shape != self.input_ids.shape
        ):
            raise ValueError("QA query mask must match input_ids.")
        if self.qa_query_token_indices is not None:
            if self.qa_query_token_indices.ndim != 1:
                raise ValueError("QA query token indices must be one-dimensional.")
            if self.qa_query_token_indices.dtype != torch.long:
                raise ValueError("QA query token indices must use torch.long.")
            if (
                not self.cpu_value_checks_passed
                and self.qa_query_pos_mask is not None
                and not torch.equal(
                    self.qa_query_token_indices,
                    self.qa_query_pos_mask[0].nonzero(as_tuple=False).flatten(),
                )
            ):
                raise ValueError(
                    "QA query token indices do not match the QA query mask."
                )

    def _validate_index(
        self,
        indices: torch.Tensor,
        mask: torch.Tensor,
        label: str,
    ) -> None:
        if indices.ndim != 1:
            raise ValueError(f"{label} token indices must be one-dimensional.")
        if indices.dtype != torch.long:
            raise ValueError(f"{label} token indices must use torch.long.")
        if (
            not self.cpu_value_checks_passed
            and not torch.equal(
                indices,
                mask[0].nonzero(as_tuple=False).flatten(),
            )
        ):
            raise ValueError(f"{label} token indices do not match the {label.lower()} mask.")

    def _condition_indices_from_masks(self) -> torch.Tensor:
        if self.condition_pos_mask is not None:
            mask = self.condition_pos_mask[0]
        elif self.assistant_pos_mask is not None:
            mask = ~self.assistant_pos_mask[0]
        else:
            mask = torch.ones_like(self.input_ids[0], dtype=torch.bool)
        return mask.nonzero(as_tuple=False).flatten()


@dataclass
class AllTaskOutput:
    family_velocities: Dict[str, Union[torch.Tensor, sp.SparseTensor]]
    und_hidden: torch.Tensor
    qa_logits: Optional[torch.Tensor]
    trace: Dict[str, Any]
    ddp_anchor: Optional[torch.Tensor] = None


def _run_pair_relation(
    query: Tuple[int, int, int, int, int],
    key: Tuple[int, int, int, int, int],
) -> int:
    query_family, query_role, query_sample, query_min, query_max = query
    key_family, key_role, key_sample, key_min, key_max = key
    if (
        query_family == FAMILY_CODE["padding"]
        or key_family == FAMILY_CODE["padding"]
    ):
        return (
            2
            if query_family == key_family == FAMILY_CODE["padding"]
            else 0
        )
    if query_sample != key_sample:
        return 0
    if query_family == FAMILY_CODE["und"]:
        if key_family == FAMILY_CODE["und"] and key_max <= query_min:
            return 2
        return (
            1
            if key_family == FAMILY_CODE["und"] and key_min <= query_max
            else 0
        )
    if query_family in tuple(FAMILY_CODE[name] for name in ALL_TASK_FAMILIES):
        query_name = next(
            name
            for name in ALL_TASK_FAMILIES
            if FAMILY_CODE[name] == query_family
        )
        if key_family == condition_family_code(query_name):
            if query_role == ROLE_SOURCE:
                return 2 if key_role == ROLE_SOURCE else 0
            return 2 if query_role == ROLE_TARGET else 0
    if key_family != query_family:
        return 0
    if query_role == ROLE_SOURCE:
        return 2 if key_role == ROLE_SOURCE else 0
    if query_role == ROLE_TARGET:
        return 2 if key_role in {ROLE_SOURCE, ROLE_TARGET} else 0
    return 0


def create_all_task_block_mask(
    index_map: AllTaskIndexMap,
    *,
    block_size: int = 128,
):
    """Build a direct rectangular Flex BlockMask for one all-task slot."""

    require_flex_attention()
    from torch.nn.attention.flex_attention import BlockMask

    index_map.validate()

    def runs(
        metadata: AllTaskTokenMetadata,
    ) -> List[List[Tuple[int, int, int, int, int]]]:
        values = torch.stack(
            (
                metadata.family[0],
                metadata.role[0],
                metadata.sample_id[0],
                metadata.local_index[0],
            )
        ).detach().cpu().tolist()
        family_values, role_values, sample_values, local_values = values
        output = []
        for start in range(0, len(family_values), int(block_size)):
            end = min(start + int(block_size), len(family_values))
            rows = []
            for index in range(start, end):
                current = (
                    int(family_values[index]),
                    int(role_values[index]),
                    int(sample_values[index]),
                    int(local_values[index]),
                    int(local_values[index]),
                )
                if rows and rows[-1][0:3] == current[0:3]:
                    rows[-1] = (*rows[-1][0:4], current[4])
                else:
                    rows.append(current)
            output.append(rows)
        return output

    query_runs = runs(index_map.query)
    key_runs = runs(index_map.key_value)
    key_blocks: Dict[Tuple[int, int], List[int]] = {}
    for block, row in enumerate(key_runs):
        for family, _role, sample, _low, _high in row:
            key_blocks.setdefault((family, sample), []).append(block)
    partial: List[List[int]] = []
    full: List[List[int]] = []
    candidates = 0
    for query_row in query_runs:
        candidate = set()
        for family, _role, sample, _low, _high in query_row:
            if family == FAMILY_CODE["und"]:
                candidate.update(key_blocks.get((family, sample), []))
            elif family in (
                FAMILY_CODE["ss"],
                FAMILY_CODE["shape"],
                FAMILY_CODE["material"],
            ):
                candidate.update(
                    key_blocks.get(
                        (
                            condition_family_code(
                                next(
                                    name
                                    for name in ALL_TASK_FAMILIES
                                    if FAMILY_CODE[name] == family
                                )
                            ),
                            sample,
                        ),
                        [],
                    )
                )
                candidate.update(key_blocks.get((family, sample), []))
        candidates += len(candidate)
        partial_row: List[int] = []
        full_row: List[int] = []
        for block in sorted(candidate):
            relations = [
                _run_pair_relation(query_run, key_run)
                for query_run in query_row
                for key_run in key_runs[block]
            ]
            if relations and all(value == 2 for value in relations):
                full_row.append(block)
            elif any(value > 0 for value in relations):
                partial_row.append(block)
        partial.append(partial_row)
        full.append(full_row)

    device = index_map.query.family.device
    block_count = len(key_runs)
    kv_indices = torch.zeros(
        (1, 1, len(query_runs), block_count),
        device=device,
        dtype=torch.int32,
    )
    full_indices = torch.zeros_like(kv_indices)
    kv_counts = torch.zeros(
        (1, 1, len(query_runs)),
        device=device,
        dtype=torch.int32,
    )
    full_counts = torch.zeros_like(kv_counts)
    for index, row in enumerate(partial):
        if row:
            kv_indices[0, 0, index, : len(row)] = torch.tensor(
                row,
                device=device,
                dtype=torch.int32,
            )
            kv_counts[0, 0, index] = len(row)
    for index, row in enumerate(full):
        if row:
            full_indices[0, 0, index, : len(row)] = torch.tensor(
                row,
                device=device,
                dtype=torch.int32,
            )
            full_counts[0, 0, index] = len(row)

    query_metadata = index_map.query
    key_metadata = index_map.key_value

    def mask_mod(batch, head, query_index, key_index):
        del head
        query_family = query_metadata.family[batch, query_index]
        query_role = query_metadata.role[batch, query_index]
        key_family = key_metadata.family[batch, key_index]
        key_role = key_metadata.role[batch, key_index]
        same = query_metadata.sample_id[batch, query_index].eq(
            key_metadata.sample_id[batch, key_index]
        )
        valid = query_metadata.valid[batch, query_index] & key_metadata.valid[
            batch,
            key_index,
        ]
        causal = key_metadata.local_index[batch, key_index].le(
            query_metadata.local_index[batch, query_index]
        )
        allowed = (
            query_family.eq(FAMILY_CODE["und"])
            & key_family.eq(FAMILY_CODE["und"])
            & causal
        )
        for family in ALL_TASK_FAMILIES:
            code = FAMILY_CODE[family]
            source = (
                query_role.eq(ROLE_SOURCE)
                & key_role.eq(ROLE_SOURCE)
                & key_family.eq(code)
            )
            target = (
                query_role.eq(ROLE_TARGET)
                & key_family.eq(code)
                & key_role.ge(ROLE_SOURCE)
            )
            condition = query_role.eq(ROLE_TARGET) & key_family.eq(
                condition_family_code(family)
            )
            source_boundary = (
                query_role.eq(ROLE_SOURCE)
                & key_family.eq(condition_family_code(family))
                & key_role.eq(ROLE_SOURCE)
            )
            allowed = allowed | (
                query_family.eq(code)
                & (source | target | condition | source_boundary)
            )
        padding = query_family.eq(FAMILY_CODE["padding"]) & key_family.eq(
            FAMILY_CODE["padding"]
        )
        return (valid & same & allowed) | padding

    mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_counts,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_counts,
        full_kv_indices=full_indices,
        BLOCK_SIZE=int(block_size),
        mask_mod=mask_mod,
        seq_lengths=(index_map.query_length, index_map.key_value_length),
    )
    mask.eva01_stats = DirectBlockMaskStats(
        query_tokens=index_map.query_length,
        key_value_tokens=index_map.key_value_length,
        query_blocks=len(query_runs),
        key_value_blocks=len(key_runs),
        full_blocks=sum(len(row) for row in full),
        partial_blocks=sum(len(row) for row in partial),
        candidate_blocks=candidates,
        mask_batch=1,
        mask_heads=1,
        block_size=int(block_size),
    )
    return mask


__all__ = [
    "FAMILY_CODE",
    "ROLE_CODE",
    "ROLE_NONE",
    "ROLE_SOURCE",
    "ROLE_TARGET",
    "AllTaskFamilyChunk",
    "AllTaskIndexMap",
    "AllTaskOutput",
    "AllTaskPackedBatch",
    "AllTaskTokenAxis",
    "AllTaskTokenMetadata",
    "build_all_task_index_map",
    "create_all_task_block_mask",
    "condition_family_code",
]
