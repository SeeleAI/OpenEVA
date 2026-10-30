"""Fail-closed direct-BlockMask FlexAttention for EVA01 Unified MoT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .index_map import (
    CONDITION_CODE_BY_QUERY_FAMILY,
    FAMILY_MATERIAL,
    FAMILY_PADDING,
    FAMILY_SHAPE,
    FAMILY_SS,
    FAMILY_UND,
    UnifiedIndexMap,
)

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        flex_attention as torch_flex_attention,
    )

    _FLEX_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as error:  # pragma: no cover - exercised by fail-closed import tests
    BlockMask = Any  # type: ignore[assignment,misc]
    torch_flex_attention = None  # type: ignore[assignment]
    _FLEX_IMPORT_ERROR = error


@dataclass(frozen=True)
class DirectBlockMaskStats:
    """Compact construction receipt attached to every formal BlockMask."""

    query_tokens: int
    key_value_tokens: int
    query_blocks: int
    key_value_blocks: int
    full_blocks: int
    partial_blocks: int
    candidate_blocks: int
    mask_batch: int
    mask_heads: int
    block_size: int
    dense_truth_materialized: bool = False


@dataclass(frozen=True)
class _Run:
    family: int
    sample_id: int
    start: int
    end: int
    min_local: int
    max_local: int


def require_flex_attention() -> None:
    """Fail immediately when the required FlexAttention API is unavailable."""

    if _FLEX_IMPORT_ERROR is not None or torch_flex_attention is None:
        raise RuntimeError(
            "EVA01 requires torch FlexAttention and does not permit an SDPA/eager fallback."
        ) from _FLEX_IMPORT_ERROR


def _block_runs(index_map: UnifiedIndexMap, *, query_axis: bool, block_size: int) -> List[List[_Run]]:
    metadata = index_map.query if query_axis else index_map.key_value
    if metadata.family.shape[0] != 1:
        raise ValueError("Formal EVA01 direct BlockMask requires physical batch size one.")
    families = metadata.family[0].detach().to(device="cpu", dtype=torch.long).tolist()
    samples = metadata.sample_id[0].detach().to(device="cpu", dtype=torch.long).tolist()
    locals_ = metadata.local_index[0].detach().to(device="cpu", dtype=torch.long).tolist()
    blocks: List[List[_Run]] = []
    for block_start in range(0, len(families), int(block_size)):
        block_end = min(block_start + int(block_size), len(families))
        runs: List[_Run] = []
        run_start = block_start
        while run_start < block_end:
            family = int(families[run_start])
            sample = int(samples[run_start])
            run_end = run_start + 1
            while (
                run_end < block_end
                and int(families[run_end]) == family
                and int(samples[run_end]) == sample
            ):
                run_end += 1
            local_values = locals_[run_start:run_end]
            runs.append(
                _Run(
                    family=family,
                    sample_id=sample,
                    start=run_start,
                    end=run_end,
                    min_local=min(int(value) for value in local_values),
                    max_local=max(int(value) for value in local_values),
                )
            )
            run_start = run_end
        blocks.append(runs)
    return blocks


def _visible_native_families(query_family: int) -> Tuple[int, ...]:
    if query_family == FAMILY_SS:
        return (FAMILY_SS,)
    if query_family == FAMILY_SHAPE:
        return (FAMILY_SS, FAMILY_SHAPE)
    if query_family == FAMILY_MATERIAL:
        return (FAMILY_SS, FAMILY_SHAPE, FAMILY_MATERIAL)
    return ()


def _visible_key_families(query_family: int, *, variant: str) -> Tuple[int, ...]:
    if query_family == FAMILY_PADDING:
        return (FAMILY_PADDING,)
    if query_family == FAMILY_UND:
        return (FAMILY_UND,)
    native = _visible_native_families(query_family)
    if not native:
        return ()
    condition = FAMILY_UND if variant == "A" else CONDITION_CODE_BY_QUERY_FAMILY[query_family]
    return (condition, *native)


def _run_pair_relation(query: _Run, key_value: _Run, *, variant: str) -> int:
    """Return zero for none, one for partial, or two for fully visible."""

    if query.family == FAMILY_PADDING or key_value.family == FAMILY_PADDING:
        return 2 if query.family == key_value.family == FAMILY_PADDING else 0
    if query.sample_id != key_value.sample_id:
        return 0
    if key_value.family not in _visible_key_families(query.family, variant=variant):
        return 0
    if query.family == FAMILY_UND:
        if key_value.max_local <= query.min_local:
            return 2
        if key_value.min_local <= query.max_local:
            return 1
        return 0
    return 2


def _classify_tile(query_runs: Sequence[_Run], key_value_runs: Sequence[_Run], *, variant: str) -> int:
    relations = [
        _run_pair_relation(query, key_value, variant=variant)
        for query in query_runs
        for key_value in key_value_runs
    ]
    if relations and all(relation == 2 for relation in relations):
        return 2
    if any(relation > 0 for relation in relations):
        return 1
    return 0


def _candidate_blocks(
    query_runs: Sequence[_Run],
    *,
    variant: str,
    key_blocks_by_family_sample: Dict[Tuple[int, int], Tuple[int, ...]],
) -> Tuple[int, ...]:
    candidates: set[int] = set()
    for query in query_runs:
        for family in _visible_key_families(query.family, variant=variant):
            candidates.update(key_blocks_by_family_sample.get((family, query.sample_id), ()))
    return tuple(sorted(candidates))


def _build_mask_mod(index_map: UnifiedIndexMap) -> Callable[..., torch.Tensor]:
    query = index_map.query
    key_value = index_map.key_value

    def mask_mod(
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_value_index: torch.Tensor,
    ) -> torch.Tensor:
        del head_index
        q_family = query.family[batch_index, query_index]
        kv_family = key_value.family[batch_index, key_value_index]
        q_sample = query.sample_id[batch_index, query_index]
        kv_sample = key_value.sample_id[batch_index, key_value_index]
        same_sample = q_sample.eq(kv_sample)
        valid = query.valid[batch_index, query_index] & key_value.valid[
            batch_index, key_value_index
        ]
        und = (
            q_family.eq(FAMILY_UND)
            & kv_family.eq(FAMILY_UND)
            & key_value.local_index[batch_index, key_value_index].le(
                query.local_index[batch_index, query_index]
            )
        )
        allowed = und
        for query_family, visible_native in (
            (FAMILY_SS, (FAMILY_SS,)),
            (FAMILY_SHAPE, (FAMILY_SS, FAMILY_SHAPE)),
            (FAMILY_MATERIAL, (FAMILY_SS, FAMILY_SHAPE, FAMILY_MATERIAL)),
        ):
            condition = (
                FAMILY_UND
                if index_map.variant == "A"
                else CONDITION_CODE_BY_QUERY_FAMILY[query_family]
            )
            visible = kv_family.eq(condition)
            for native_family in visible_native:
                visible = visible | kv_family.eq(native_family)
            allowed = allowed | (q_family.eq(query_family) & visible)
        padding = q_family.eq(FAMILY_PADDING) & kv_family.eq(FAMILY_PADDING)
        return (valid & same_sample & allowed) | padding

    return mask_mod


def create_unified_block_mask(
    index_map: UnifiedIndexMap,
    *,
    block_size: int = 128,
    compile_mask: bool = False,
    num_heads: int = 1,
) -> BlockMask:
    """Build BlockMask directly from sparse KV block metadata without a T-by-T tensor."""

    require_flex_attention()
    index_map.validate()
    if int(num_heads) != 1:
        raise ValueError("EVA01 token-level BlockMask must use H=1 and broadcast to carrier heads.")
    if compile_mask:
        raise ValueError("Direct BlockMask construction has no create_block_mask compile path.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    query_runs = _block_runs(index_map, query_axis=True, block_size=int(block_size))
    key_value_runs = _block_runs(index_map, query_axis=False, block_size=int(block_size))
    key_blocks_by_family_sample: Dict[Tuple[int, int], set[int]] = {}
    for block_index, runs in enumerate(key_value_runs):
        for run in runs:
            key_blocks_by_family_sample.setdefault((run.family, run.sample_id), set()).add(block_index)
    frozen_key_blocks = {
        key: tuple(sorted(values)) for key, values in key_blocks_by_family_sample.items()
    }

    partial_rows: List[List[int]] = []
    full_rows: List[List[int]] = []
    candidate_count = 0
    for runs in query_runs:
        candidates = _candidate_blocks(
            runs,
            variant=index_map.variant,
            key_blocks_by_family_sample=frozen_key_blocks,
        )
        candidate_count += len(candidates)
        partial: List[int] = []
        full: List[int] = []
        for key_block in candidates:
            relation = _classify_tile(
                runs,
                key_value_runs[key_block],
                variant=index_map.variant,
            )
            if relation == 2:
                full.append(key_block)
            elif relation == 1:
                partial.append(key_block)
        partial_rows.append(partial)
        full_rows.append(full)

    device = index_map.query.family.device
    query_blocks = len(query_runs)
    # PyTorch 2.6's ordered-to-dense transpose uses the final dimension as
    # the KV block domain, so it must cover every legal block index even when
    # each row contains only a small visible subset.
    key_value_blocks = len(key_value_runs)
    kv_num_blocks = torch.tensor(
        [[len(row) for row in partial_rows]],
        device=device,
        dtype=torch.int32,
    ).unsqueeze(0)
    full_kv_num_blocks = torch.tensor(
        [[len(row) for row in full_rows]],
        device=device,
        dtype=torch.int32,
    ).unsqueeze(0)
    kv_indices = torch.zeros(
        (1, 1, query_blocks, key_value_blocks),
        device=device,
        dtype=torch.int32,
    )
    full_kv_indices = torch.zeros(
        (1, 1, query_blocks, key_value_blocks),
        device=device,
        dtype=torch.int32,
    )
    for query_block, row in enumerate(partial_rows):
        if row:
            kv_indices[0, 0, query_block, : len(row)] = torch.tensor(
                row,
                device=device,
                dtype=torch.int32,
            )
    for query_block, row in enumerate(full_rows):
        if row:
            full_kv_indices[0, 0, query_block, : len(row)] = torch.tensor(
                row,
                device=device,
                dtype=torch.int32,
            )
    try:
        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=int(block_size),
            mask_mod=_build_mask_mod(index_map),
            seq_lengths=(index_map.query_length, index_map.key_value_length),
        )
    except BaseException as error:
        raise RuntimeError(
            "EVA01 direct BlockMask construction failed; attention fallback is forbidden."
        ) from error
    stats = DirectBlockMaskStats(
        query_tokens=index_map.query_length,
        key_value_tokens=index_map.key_value_length,
        query_blocks=len(query_runs),
        key_value_blocks=len(key_value_runs),
        full_blocks=sum(len(row) for row in full_rows),
        partial_blocks=sum(len(row) for row in partial_rows),
        candidate_blocks=candidate_count,
        mask_batch=1,
        mask_heads=1,
        block_size=int(block_size),
    )
    setattr(block_mask, "eva01_stats", stats)
    return block_mask


class FlexAttentionOnly(nn.Module):
    """One strict FlexAttention event with no alternate backend."""

    backend_name = "flex_attention"
    # PyTorch 2.6 selects an eight-warp H20 default for head_dim=128 and then
    # removes that choice because of pytorch/pytorch#141603.  max_autotune is
    # therefore still required to expose a legal four-warp backward choice.
    # Leaving the backward geometry unconstrained is unsafe for rectangular
    # Variant-B Q/KV: one PyTorch 2.6 Triton candidate can poison the parent
    # CUDA context.  Pin the verified backward tiles and benchmark all Flex
    # candidates in Inductor's isolated autotune subprocess.  The selected op
    # remains FlexAttention and execution still fails closed.
    safe_kernel_options = {
        "BLOCK_M1": 32,
        "BLOCK_N1": 64,
        "BLOCK_M2": 64,
        "BLOCK_N2": 32,
    }

    def __init__(
        self,
        *,
        head_dim: int = 128,
        compile_kernel: bool = True,
    ) -> None:
        super().__init__()
        require_flex_attention()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        self.head_dim = int(head_dim)
        self.scale = self.head_dim**-0.5
        self.compile_kernel = bool(compile_kernel)
        self.kernel_options = dict(self.safe_kernel_options)
        self.call_count = 0
        self._training_kernel: Callable[..., torch.Tensor]
        self._inference_kernel: Callable[..., torch.Tensor]
        if self.compile_kernel:
            try:
                # Both compiled callables lower the same FlexAttention op.
                # PyTorch 2.6 needs max-autotune to find a legal H20 backward
                # kernel for this head/block geometry, while max-autotune can
                # benchmark an unsafe forward candidate when an evaluator
                # encounters several sequence shapes.  Keep the training and
                # no-grad compile profiles separate without mutating global
                # Inductor configuration or introducing another backend.
                self._training_kernel = torch.compile(
                    torch_flex_attention,
                    dynamic=False,
                    options={"max_autotune": True, "autotune_in_subproc": True},
                )
                self._inference_kernel = torch.compile(
                    torch_flex_attention,
                    dynamic=False,
                    options={"max_autotune": False},
                )
            except BaseException as error:
                raise RuntimeError(
                    "EVA01 FlexAttention compile setup failed; attention fallback is forbidden."
                ) from error
        else:
            self._training_kernel = torch_flex_attention
            self._inference_kernel = torch_flex_attention

    def reset_call_count(self) -> None:
        self.call_count = 0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("FlexAttention Q/K/V must have shape [B,H,T,D].")
        if query.shape[:2] != key.shape[:2] or key.shape != value.shape:
            raise ValueError(
                f"Invalid rectangular Q/K/V shapes: Q={tuple(query.shape)}, "
                f"K={tuple(key.shape)}, V={tuple(value.shape)}."
            )
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError(f"Unified carrier head_dim must be {self.head_dim}.")
        if not isinstance(block_mask, BlockMask):
            raise TypeError("EVA01 Unified MoT requires a torch BlockMask instance.")
        needs_backward = torch.is_grad_enabled() and (
            query.requires_grad or key.requires_grad or value.requires_grad
        )
        kernel = self._training_kernel if needs_backward else self._inference_kernel
        try:
            output = kernel(
                query,
                key,
                value,
                block_mask=block_mask,
                scale=self.scale,
                kernel_options=self.kernel_options,
            )
        except BaseException as error:
            raise RuntimeError(
                "EVA01 FlexAttention execution failed; SDPA/eager fallback is forbidden."
            ) from error
        self.call_count += 1
        if not isinstance(output, torch.Tensor):
            raise TypeError("FlexAttention returned an unexpected non-tensor output.")
        return output


__all__ = [
    "BlockMask",
    "DirectBlockMaskStats",
    "FlexAttentionOnly",
    "create_unified_block_mask",
    "require_flex_attention",
]
