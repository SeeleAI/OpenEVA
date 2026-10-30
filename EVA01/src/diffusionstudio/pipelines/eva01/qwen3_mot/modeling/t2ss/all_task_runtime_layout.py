"""Physical carrier layout helpers for EVA01 Unified All-Task V2."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import torch

from ...trellis2.modules import sparse as sp
from .adapters import CARRIER_HEAD_DIM, CARRIER_HEADS, linear_with_lora
from .all_task_adapters import ALL_TASK_FAMILIES
from .all_task_layout import (
    FAMILY_CODE,
    ROLE_CODE,
    ROLE_NONE,
    ROLE_SOURCE,
    ROLE_TARGET,
    AllTaskFamilyChunk,
    AllTaskIndexMap,
    AllTaskPackedBatch,
    AllTaskTokenAxis,
    build_all_task_index_map,
)
from .runtime_types import FamilyRuntimeState


class AllTaskRuntimeLayoutMixin:
    """Build semantic metadata and numerically stable physical Flex axes."""

    @staticmethod
    def _axis_for_state(
        chunk: AllTaskFamilyChunk,
        state: FamilyRuntimeState,
    ) -> AllTaskTokenAxis:
        return AllTaskTokenAxis(
            name=chunk.name,
            family=chunk.family,
            role=ROLE_CODE[chunk.role],
            sample_id=state.sample_ids,
            local_index=state.local_indices,
            position_ids=state.position_ids,
        )

    @staticmethod
    def _und_axis(batch: AllTaskPackedBatch) -> AllTaskTokenAxis:
        return AllTaskTokenAxis(
            "und",
            "und",
            ROLE_NONE,
            batch.und_sample_ids,
            batch.und_position_ids[0],
            batch.und_position_ids,
        )

    @staticmethod
    def _und_axis_indices(
        batch: AllTaskPackedBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partition compact UND rows without changing their logical order.

        Boundary tokens are logically interleaved in the virtual sequence, but
        a trailing physical slice prevents them from shifting inherited
        Current-B SS rows.  BlockMask metadata and explicit positions continue
        to own all logical causality and Running-Base semantics.
        """

        rows = torch.arange(
            batch.input_ids.shape[1],
            device=batch.input_ids.device,
            dtype=torch.long,
        )
        if batch.und_boundary_family is None:
            return rows, rows.new_empty((0,))
        boundary = batch.und_boundary_family[0].to(
            device=rows.device,
            dtype=torch.long,
        ).ne(0)
        return (
            (~boundary).nonzero(as_tuple=False).flatten(),
            boundary.nonzero(as_tuple=False).flatten(),
        )

    @classmethod
    def _und_axes(
        cls,
        batch: AllTaskPackedBatch,
    ) -> Tuple[AllTaskTokenAxis, ...]:
        common, boundary = cls._und_axis_indices(batch)

        def axis(name: str, indices: torch.Tensor) -> AllTaskTokenAxis:
            return AllTaskTokenAxis(
                name,
                "und",
                ROLE_NONE,
                batch.und_sample_ids.index_select(1, indices),
                batch.und_position_ids[0].index_select(1, indices),
                batch.und_position_ids.index_select(2, indices),
            )

        result = [axis("und", common)]
        if int(boundary.numel()):
            result.append(axis("und_boundary", boundary))
        return tuple(result)

    @classmethod
    def _condition_axis_indices(
        cls,
        family: str,
        role: int,
        batch: AllTaskPackedBatch,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        condition = cls._condition_mask(batch)[0]
        if batch.und_boundary_family is None:
            boundary_family = torch.zeros_like(batch.input_ids[0])
            boundary_role = torch.zeros_like(batch.input_ids[0])
        else:
            boundary_family = batch.und_boundary_family[0].to(
                device=batch.input_ids.device,
                dtype=torch.long,
            )
            assert batch.und_boundary_role is not None
            boundary_role = batch.und_boundary_role[0].to(
                device=batch.input_ids.device,
                dtype=torch.long,
            )
        selected = condition & (
            boundary_family.eq(0)
            | boundary_family.eq(FAMILY_CODE[str(family)])
        )
        selected &= boundary_role.eq(int(role))
        if sample_ids is not None:
            selected &= cls._sample_id_mask(batch, sample_ids)
        return selected.nonzero(as_tuple=False).flatten()

    @staticmethod
    def _sample_id_mask(
        batch: AllTaskPackedBatch,
        sample_ids: Sequence[int],
    ) -> torch.Tensor:
        rows = batch.und_sample_ids[0].to(
            device=batch.input_ids.device,
            dtype=torch.long,
        )
        unique = tuple(dict.fromkeys(int(value) for value in sample_ids))
        selected = torch.zeros_like(rows, dtype=torch.bool)
        for sample_id in unique:
            selected |= rows.eq(int(sample_id))
        return selected

    @classmethod
    def _condition_indices_for_samples(
        cls,
        batch: AllTaskPackedBatch,
        sample_ids: Sequence[int],
    ) -> torch.Tensor:
        selected = cls._condition_mask(batch)[0] & cls._sample_id_mask(
            batch,
            sample_ids,
        )
        return selected.nonzero(as_tuple=False).flatten()

    @classmethod
    def _condition_axes(
        cls,
        family: str,
        batch: AllTaskPackedBatch,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[AllTaskTokenAxis, ...]:
        axes = []
        role_names = {
            ROLE_NONE: "common",
            ROLE_SOURCE: "source",
            ROLE_TARGET: "target",
        }
        for role, role_name in role_names.items():
            indices = cls._condition_axis_indices(
                family,
                role,
                batch,
                sample_ids,
            )
            if not int(indices.numel()):
                continue
            axes.append(
                AllTaskTokenAxis(
                    f"condition_{family}_{role_name}",
                    f"condition_{family}",
                    role,
                    batch.und_sample_ids.index_select(1, indices),
                    batch.und_position_ids[0].index_select(1, indices),
                    batch.und_position_ids.index_select(2, indices),
                )
            )
        return tuple(axes)

    @staticmethod
    def _condition_axis_family(name: str) -> str:
        parts = str(name).split("_")
        if len(parts) != 3 or parts[0] != "condition":
            raise ValueError(f"Invalid condition axis name {name!r}.")
        family = parts[1]
        if family not in ALL_TASK_FAMILIES:
            raise ValueError(f"Invalid condition family in axis {name!r}.")
        return family

    @staticmethod
    def _condition_axis_role(name: str) -> int:
        role_name = str(name).rsplit("_", 1)[-1]
        role = {
            "common": ROLE_NONE,
            "source": ROLE_SOURCE,
            "target": ROLE_TARGET,
        }.get(role_name)
        if role is None:
            raise ValueError(f"Invalid condition role in axis {name!r}.")
        return role

    @staticmethod
    def _legacy_condition_axis(
        family: str,
        batch: AllTaskPackedBatch,
    ) -> AllTaskTokenAxis:
        indices = AllTaskRuntimeLayoutMixin._condition_indices(batch)
        return AllTaskTokenAxis(
            f"condition_{family}",
            f"condition_{family}",
            ROLE_NONE,
            batch.und_sample_ids.index_select(1, indices),
            batch.und_position_ids[0].index_select(1, indices),
            batch.und_position_ids.index_select(2, indices),
        )

    def _slot_map(
        self,
        batch: AllTaskPackedBatch,
        states: Mapping[
            str,
            List[Tuple[AllTaskFamilyChunk, FamilyRuntimeState]],
        ],
        slot: int,
    ) -> AllTaskIndexMap:
        query_axes: List[AllTaskTokenAxis] = []
        und_axes = self._und_axes(batch) if slot < 28 else ()
        if und_axes:
            query_axes.append(und_axes[0])
        active_families = tuple(
            family
            for family in ALL_TASK_FAMILIES
            if any(
                slot < len(state.base.blocks)
                for _chunk, state in states[family]
            )
        )
        active_sample_ids = {
            family: tuple(
                dict.fromkeys(
                    int(chunk.sample_id)
                    for chunk, state in states[family]
                    if slot < len(state.base.blocks)
                )
            )
            for family in active_families
        }
        condition_axes = tuple(
            axis
            for family in active_families
            for axis in self._condition_axes(
                family,
                batch,
                active_sample_ids[family],
            )
        )
        common_conditions = tuple(
            axis for axis in condition_axes if int(axis.role) == ROLE_NONE
        )
        boundary_conditions = tuple(
            axis for axis in condition_axes if int(axis.role) != ROLE_NONE
        )
        key_axes: List[AllTaskTokenAxis] = []
        if und_axes:
            key_axes.append(und_axes[0])
        key_axes.extend(common_conditions)
        for family in ALL_TASK_FAMILIES:
            for chunk, state in states[family]:
                if slot < len(state.base.blocks):
                    axis = self._axis_for_state(chunk, state)
                    query_axes.append(axis)
                    key_axes.append(axis)
        if len(und_axes) > 1:
            query_axes.append(und_axes[1])
            key_axes.append(und_axes[1])
        key_axes.extend(boundary_conditions)
        return build_all_task_index_map(
            query_axes=query_axes,
            key_value_axes=key_axes,
            max_query_tokens=int(batch.max_unified_tokens),
            max_key_value_tokens=int(batch.max_key_value_tokens),
        )

    def _grouped_sparse_native_self_pre_mot(
        self,
        active: Sequence[Tuple[AllTaskFamilyChunk, FamilyRuntimeState]],
        *,
        block_index: int,
    ) -> None:
        if len(active) < 2:
            raise ValueError("Grouped sparse native attention requires multiple chunks.")
        block = active[0][1].base.blocks[int(block_index)]
        branches = []
        gates = []
        for _chunk, state in active:
            current = state.base.blocks[int(block_index)]
            if current is not block or not isinstance(state.hidden, sp.SparseTensor):
                raise RuntimeError("Grouped sparse chunks must share one family block.")
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self._native_modulation(current, state.timestep_embedding)
            )
            branches.append(
                state.hidden.replace(
                    current.norm1(state.hidden.feats) * (1 + scale_msa)
                    + shift_msa
                )
            )
            gates.append(gate_msa)
            state.mlp_modulation = (shift_mlp, scale_mlp, gate_mlp)
        attended = block.self_attn.grouped_self_forward(branches)
        if len(attended) != len(active):
            raise RuntimeError("Grouped sparse attention output count drifted.")
        for (_chunk, state), branch, gate in zip(active, attended, gates):
            state.hidden = state.hidden.replace(
                state.hidden.feats + branch.feats * gate
            )

    def _apply_grouped_sparse_family_attention_output(
        self,
        active: Sequence[Tuple[AllTaskFamilyChunk, FamilyRuntimeState]],
        *,
        block_index: int,
        carrier_outputs: Sequence[torch.Tensor],
    ) -> None:
        if len(active) < 2 or len(active) != len(carrier_outputs):
            raise ValueError("Grouped sparse output requires aligned multiple chunks.")
        first_state = active[0][1]
        block = first_state.base.blocks[int(block_index)]
        adapters = self.adapters.family_slot(first_state.name, block_index)
        lengths = tuple(
            int(state.hidden.feats.shape[0]) for _chunk, state in active
        )
        for (_chunk, state), carrier, length in zip(
            active,
            carrier_outputs,
            lengths,
        ):
            if (
                state.base.blocks[int(block_index)] is not block
                or not isinstance(state.hidden, sp.SparseTensor)
                or int(carrier.shape[1]) != length
            ):
                raise RuntimeError("Grouped sparse output family ABI drifted.")
        combined_carrier = torch.cat(tuple(carrier_outputs), dim=1)
        native = adapters.o_bridge(combined_carrier)
        native_features = native.squeeze(0).reshape(
            int(native.shape[1]),
            adapters.spec.hidden_size,
        )
        combined_branch = torch.cat(
            tuple(
                torch.full(
                    (length,),
                    False,
                    device=native_features.device,
                    dtype=torch.bool,
                )
                for (chunk, _state), length in zip(active, lengths)
            ),
            dim=0,
        )
        update = linear_with_lora(
            native_features,
            block.cross_attn.to_out,
            adapters.o,
            branch_mask=combined_branch,
        )
        branches = []
        gates = []
        cursor = 0
        for (_chunk, state), length in zip(active, lengths):
            end = cursor + length
            state.hidden = state.hidden.replace(
                state.hidden.feats + update[cursor:end]
            )
            if state.mlp_modulation is None:
                raise RuntimeError("Sparse family MLP modulation is missing.")
            shift_mlp, scale_mlp, gate_mlp = state.mlp_modulation
            branches.append(
                block.norm3(state.hidden.feats) * (1 + scale_mlp) + shift_mlp
            )
            gates.append(gate_mlp)
            cursor = end
        if cursor != int(update.shape[0]):
            raise RuntimeError("Grouped sparse O projection split did not cover rows.")
        combined_branch = sp.VarLenTensor.from_tensor_list(branches)
        mlp_rows = block.mlp(combined_branch).to_tensor_list()
        for (_chunk, state), branch, gate in zip(active, mlp_rows, gates):
            state.hidden = state.hidden.replace(state.hidden.feats + branch * gate)

    @staticmethod
    def _pad(
        values: Sequence[torch.Tensor],
        length: int,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if not values:
            raise ValueError("Cannot build an empty carrier axis.")
        active = torch.cat(tuple(values), dim=1)
        if active.shape[1] > int(length):
            raise ValueError("Carrier active length exceeds cart budget.")
        if active.shape[1] == int(length):
            return active
        zeros = torch.zeros(
            (1, int(length) - active.shape[1], CARRIER_HEADS, CARRIER_HEAD_DIM),
            device=active.device,
            dtype=dtype or active.dtype,
        )
        return torch.cat((active, zeros), dim=1)


__all__ = ["AllTaskRuntimeLayoutMixin"]
