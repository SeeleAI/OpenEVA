"""Whole-slot activation checkpoint helpers for Unified All-Task V2."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ...trellis2.modules import sparse as sp
from .all_task_adapters import ALL_TASK_FAMILIES
from .all_task_layout import AllTaskFamilyChunk, AllTaskIndexMap, AllTaskPackedBatch
from .runtime_types import FamilyRuntimeState


class AllTaskSlotCheckpointMixin:
    """Keep the tensor-only checkpoint ABI separate from the slot executor."""

    @staticmethod
    def _runtime_state_tensor(state: FamilyRuntimeState) -> torch.Tensor:
        if isinstance(state.hidden, sp.SparseTensor):
            return state.hidden.feats
        return state.hidden

    @staticmethod
    def _record_runtime_state_on_stream(
        state: FamilyRuntimeState,
        stream: torch.cuda.Stream,
    ) -> None:
        hidden = state.hidden
        if isinstance(hidden, sp.SparseTensor):
            hidden.feats.record_stream(stream)
            hidden.coords.record_stream(stream)
        else:
            hidden.record_stream(stream)

    @staticmethod
    def _replace_runtime_state_tensor(
        state: FamilyRuntimeState,
        value: torch.Tensor,
    ) -> None:
        if isinstance(state.hidden, sp.SparseTensor):
            state.hidden = state.hidden.replace(value)
        else:
            state.hidden = value

    def _run_all_slot_checkpointed(
        self,
        batch: AllTaskPackedBatch,
        states: Dict[
            str,
            List[Tuple[AllTaskFamilyChunk, FamilyRuntimeState]],
        ],
        und_hidden: torch.Tensor,
        slot: int,
        index_map: AllTaskIndexMap,
        block_mask: Any,
        mrope_factors: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Recompute one complete unified slot during backward.

        Sparse coordinates stay owned by their original SparseTensor. Only
        features cross the tensor-only checkpoint ABI. The recompute executes
        the same parallel native branches and one direct FlexAttention call.
        """

        active_states = [
            state
            for family in ALL_TASK_FAMILIES
            for _chunk, state in states[family]
            if slot < len(state.base.blocks)
        ]
        trace_holder: Dict[str, Dict[str, Any]] = {}

        def run_slot(
            current_und: torch.Tensor,
            *current_family_hidden: torch.Tensor,
        ) -> Tuple[torch.Tensor, ...]:
            if len(current_family_hidden) != len(active_states):
                raise RuntimeError("Activation-checkpoint state arity drifted.")
            for state, value in zip(active_states, current_family_hidden):
                self._replace_runtime_state_tensor(state, value)
            # Recompute runs under autograd's non-reentrant checkpoint engine.
            # The forward path may use side streams, but mutating the shared
            # FamilyRuntimeState objects from those streams during recompute
            # races the tensor-only checkpoint ABI (and produced a real CUDA
            # Indexing.cu device-side assert at 30 slots).  Keep recompute
            # deterministic/serial; the normal forward still retains the
            # parallel native branches used by the performance profile.
            recompute_parallel = bool(
                getattr(self, "checkpoint_recompute_parallel_native", True)
            )
            next_und, trace = self._run_all_slot(
                batch,
                states,
                current_und,
                slot,
                index_map,
                block_mask,
                mrope_factors,
                parallel_native_branches=recompute_parallel,
            )
            trace_holder["trace"] = trace
            return (
                next_und,
                *(self._runtime_state_tensor(state) for state in active_states),
            )

        outputs = activation_checkpoint(
            run_slot,
            und_hidden,
            *(self._runtime_state_tensor(state) for state in active_states),
            use_reentrant=False,
            preserve_rng_state=True,
        )
        if not isinstance(outputs, tuple) or len(outputs) != len(active_states) + 1:
            raise RuntimeError("Activation-checkpoint output ABI drifted.")
        next_und = outputs[0]
        for state, value in zip(active_states, outputs[1:]):
            self._replace_runtime_state_tensor(state, value)
        trace = trace_holder.get("trace")
        if trace is None:
            raise RuntimeError("Activation-checkpoint slot trace is missing.")
        result = dict(trace)
        result["activation_checkpointed"] = True
        return next_und, result

    def activation_checkpoint_slot_indices(self, slot_count: int) -> Tuple[int, ...]:
        """Return a deterministic layer-balanced checkpoint schedule."""

        total = int(slot_count)
        count = min(int(self.activation_checkpoint_slots), total)
        if count <= 0:
            return ()
        return tuple((index * total) // count for index in range(count))


__all__ = ["AllTaskSlotCheckpointMixin"]
