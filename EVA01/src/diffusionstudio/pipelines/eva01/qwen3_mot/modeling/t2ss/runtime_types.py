"""Shared task metadata and runtime value types for EVA01 Unified MoT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ...trellis2.modules import sparse as sp
from .index_map import TokenAxis


EXPECTED_FROZEN_PARAMETERS: Mapping[str, int] = {
    "qwen": 2_127_532_032,
    "ss": 975_440_648,
    "shape": 1_292_253_728,
    "material": 1_292_302_880,
}
EXPECTED_TRAINABLE_PARAMETERS: Mapping[Tuple[str, str], int] = {
    ("t2ss", "A"): 16_642_048,
    ("t2ss", "B"): 82_795_264,
    ("t2shape", "A"): 30_558_720,
    ("t2shape", "B"): 101_736_960,
    ("t2sm", "A"): 44_344_320,
    ("t2sm", "B"): 120_449_280,
    # Unified All-Task V2 restores Current-B's slot-canonical condition path
    # and adds family-native SS/Shape/Material KV views.
    ("all_task", "B"): 127_552_768,
}
VISUAL_DETAIL_SCHEMA_VERSION = 2
RAW_DETAIL_PROJECTION = "linear_pca_3072x2048"
EXPECTED_HAAR_GATED_DETAIL_DELTA = 6_827_649


def expected_haar_gated_detail_delta(
    detail_gate_hidden: int = 128,
) -> int:
    gate_hidden = int(detail_gate_hidden)
    if gate_hidden <= 0:
        raise ValueError("Detail gate hidden width must be positive.")
    raw_adapter = 3072 * 2048
    gate = (2048 + 2048 + 64 + 28 + 1) * gate_hidden + 1
    return raw_adapter + gate


def expected_trainable_parameters(
    task_scope: str,
    variant: str,
    visual_detail_mode: str,
    detail_gate_hidden: int = 128,
) -> int:
    key = (str(task_scope).strip().lower(), str(variant).strip().upper())
    base = EXPECTED_TRAINABLE_PARAMETERS[key]
    mode = str(visual_detail_mode).strip().lower()
    if mode == "none":
        return base
    if mode == "haar_gated" and key[1] == "B":
        return base + expected_haar_gated_detail_delta(
            detail_gate_hidden,
        )
    raise ValueError(f"Unsupported adapter/detail schema: {key}, {mode}.")
TASK_FAMILIES: Mapping[str, Tuple[str, ...]] = {
    "t2ss": ("ss",),
    "t2shape": ("ss", "shape"),
    "t2sm": ("ss", "shape", "material"),
    "all_task": ("ss", "shape", "material"),
}
TASK_TARGET_FAMILY: Mapping[str, str] = {
    "t2ss": "ss",
    "t2shape": "shape",
    "t2sm": "material",
    "all_task": "all",
}
TASK_SLOT_COUNT: Mapping[str, int] = {
    "t2ss": 28,
    "t2shape": 30,
    "t2sm": 30,
    "all_task": 30,
}
CONDITION_MODES = ("text", "image", "both")
UND_TOKEN_TEXT = 0
UND_TOKEN_IMAGE = 1
UND_TOKEN_SPECIAL = 2


@dataclass(frozen=True)
class PackedGenerationBatch:
    """Physical-batch-one input with P tightly packed logical samples."""

    input_ids: torch.Tensor
    und_sample_ids: torch.Tensor
    und_position_ids: torch.Tensor
    logical_sample_ids: Tuple[str, ...]
    family_inputs: Mapping[str, Any]
    max_unified_tokens: int
    max_condition_tokens: int
    record_ids: Tuple[str, ...] = ()
    occurrences: Tuple[int, ...] = ()
    pack_id: Optional[int] = None
    global_step_pack_ids: Tuple[int, ...] = ()
    global_step_logical_count: Optional[int] = None
    condition_modes: Tuple[str, ...] = ()
    condition_drop_mask: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    image_sample_ids: Optional[torch.Tensor] = None
    visual_pos_mask: Optional[torch.Tensor] = None
    visual_token_indices: Optional[torch.Tensor] = None
    und_token_kinds: Optional[torch.Tensor] = None
    selected_view_ids: Tuple[int, ...] = ()
    selected_view_paths: Tuple[str, ...] = ()
    deepstack_stage_count: int = 3

    @property
    def logical_sample_count(self) -> int:
        return len(self.logical_sample_ids)

    def validate(self, *, task: str, block_size: int) -> None:
        normalized_task = str(task).strip().lower()
        if normalized_task not in TASK_FAMILIES:
            raise ValueError(f"Unsupported generation task `{task}`.")
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("Packed generation input_ids must have physical shape [1,T_und].")
        if self.und_sample_ids.shape != self.input_ids.shape:
            raise ValueError("und_sample_ids must match packed input_ids.")
        if self.und_position_ids.shape != (3, 1, self.input_ids.shape[1]):
            raise ValueError("und_position_ids must have shape [3,1,T_und].")
        if self.logical_sample_count <= 0:
            raise ValueError("Packed generation requires at least one logical sample.")
        expected_ids = torch.arange(
            self.logical_sample_count,
            device=self.und_sample_ids.device,
            dtype=torch.long,
        )
        observed_ids = torch.unique(self.und_sample_ids[0], sorted=True)
        if not torch.equal(observed_ids, expected_ids):
            raise ValueError(
                f"Packed UND sample IDs must be contiguous 0..P-1, got {observed_ids.tolist()}."
            )
        previous_end = 0
        for sample_index in range(self.logical_sample_count):
            indices = self.und_sample_ids[0].eq(sample_index).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                raise ValueError(f"Logical sample {sample_index} has no UND tokens.")
            if int(indices[0]) != previous_end or not torch.equal(
                indices,
                torch.arange(indices[0], indices[-1] + 1, device=indices.device),
            ):
                raise ValueError("Each logical caption must occupy one tight contiguous segment.")
            previous_end = int(indices[-1]) + 1
        if previous_end != self.input_ids.shape[1]:
            raise ValueError("Packed UND captions contain an internal gap or padding token.")
        if int(self.max_unified_tokens) <= 0 or int(self.max_unified_tokens) % int(block_size):
            raise ValueError("max_unified_tokens must be a positive block-size multiple.")
        if int(self.max_condition_tokens) < int(self.input_ids.shape[1]):
            raise ValueError("max_condition_tokens is smaller than the packed conditional UND length.")
        missing = [name for name in TASK_FAMILIES[normalized_task] if name not in self.family_inputs]
        if missing:
            raise ValueError(f"Packed {normalized_task} input is missing families {missing}.")
        if self.record_ids and len(self.record_ids) != self.logical_sample_count:
            raise ValueError("record_ids must contain one value per logical sample.")
        if self.occurrences and len(self.occurrences) != self.logical_sample_count:
            raise ValueError("occurrences must contain one value per logical sample.")
        modes = self.condition_modes or ("text",) * self.logical_sample_count
        if len(modes) != self.logical_sample_count:
            raise ValueError("condition_modes must contain one value per logical sample.")
        unknown_modes = sorted(set(modes).difference(CONDITION_MODES))
        if unknown_modes:
            raise ValueError(f"Unknown condition modes: {unknown_modes}.")
        if self.condition_drop_mask is None:
            dropped = torch.zeros(
                self.logical_sample_count,
                device=self.input_ids.device,
                dtype=torch.bool,
            )
        else:
            if self.condition_drop_mask.shape != (self.logical_sample_count,):
                raise ValueError("condition_drop_mask must contain one value per logical sample.")
            dropped = self.condition_drop_mask.to(dtype=torch.bool)
        if self.und_token_kinds is not None:
            if self.und_token_kinds.shape != self.input_ids.shape:
                raise ValueError("und_token_kinds must match packed input_ids.")
            observed_kinds = set(int(value) for value in torch.unique(self.und_token_kinds).tolist())
            if not observed_kinds.issubset(
                {UND_TOKEN_TEXT, UND_TOKEN_IMAGE, UND_TOKEN_SPECIAL}
            ):
                raise ValueError(f"Unknown UND token kinds: {sorted(observed_kinds)}.")

        image_fields = (
            self.pixel_values,
            self.image_grid_thw,
            self.image_sample_ids,
            self.visual_pos_mask,
        )
        has_images = any(value is not None for value in image_fields)
        if has_images and not all(value is not None for value in image_fields):
            raise ValueError("All packed image fields must be present together.")
        image_mode_count = sum(mode in {"image", "both"} for mode in modes)
        effective_image_ids = tuple(
            index
            for index, mode in enumerate(modes)
            if mode in {"image", "both"} and not bool(dropped[index])
        )
        if has_images:
            assert self.image_grid_thw is not None
            assert self.image_sample_ids is not None
            assert self.visual_pos_mask is not None
            if self.image_grid_thw.ndim != 2 or self.image_grid_thw.shape[1] != 3:
                raise ValueError("image_grid_thw must have shape [N_images,3].")
            if self.image_sample_ids.shape != (self.image_grid_thw.shape[0],):
                raise ValueError("image_sample_ids must contain one ID per packed image.")
            if self.visual_pos_mask.shape != self.input_ids.shape or self.visual_pos_mask.dtype != torch.bool:
                raise ValueError("visual_pos_mask must be boolean and match packed input_ids.")
            if self.visual_token_indices is not None:
                if self.visual_token_indices.ndim != 1:
                    raise ValueError("visual_token_indices must be one-dimensional.")
                if self.visual_token_indices.dtype != torch.long:
                    raise ValueError("visual_token_indices must use torch.long.")
            if int(self.image_grid_thw.shape[0]) != len(effective_image_ids):
                raise ValueError("Every non-dropped image/both sample must provide exactly one image.")
            expected_visual = int(
                (self.image_grid_thw.prod(dim=-1) // 4).sum().detach().cpu()
            )
            observed_visual = (
                int(self.visual_token_indices.numel())
                if self.visual_token_indices is not None
                else int(self.visual_pos_mask.sum().detach().cpu())
            )
            if observed_visual != expected_visual:
                raise ValueError(
                    "Visual placeholder count does not match merged Qwen image tokens."
                )
            observed_image_ids = tuple(
                int(value) for value in self.image_sample_ids.detach().cpu().tolist()
            )
            if observed_image_ids != effective_image_ids:
                raise ValueError("Packed images must follow logical sample order exactly.")
        elif effective_image_ids:
            raise ValueError("Non-dropped image/both modes require packed image tensors.")
        if self.selected_view_ids and len(self.selected_view_ids) != image_mode_count:
            raise ValueError("selected_view_ids must contain one value per packed image.")
        if self.selected_view_paths and len(self.selected_view_paths) != image_mode_count:
            raise ValueError("selected_view_paths must contain one value per packed image.")
        if not 0 <= int(self.deepstack_stage_count) <= 3:
            raise ValueError("deepstack_stage_count must be in [0,3].")


@dataclass(frozen=True)
class UnifiedMoTSlotTrace:
    slot_index: int
    backend: str
    flex_call_ordinal: int
    query_length: int
    key_value_length: int
    query_families: Tuple[str, ...]
    key_value_families: Tuple[str, ...]
    family_query_tokens: Dict[str, int]
    family_key_value_tokens: Dict[str, int]
    exhausted_read_only_families: Tuple[str, ...]
    block_mask: Dict[str, Any]
    detail_gate: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedMoTForwardTrace:
    task: str
    variant: str
    slot_count: int
    flex_call_count: int
    backend: str
    logical_sample_count: int
    max_unified_tokens: int
    max_key_value_tokens: int
    active_query_tokens: int
    query_padding_tokens: int
    ss_ape_injections: int
    slots: Tuple[UnifiedMoTSlotTrace, ...]
    condition_mode_counts: Dict[str, int] = field(default_factory=dict)
    visual_token_count: int = 0
    vision_patch_token_count: int = 0
    deepstack_injections: int = 0
    visual_detail_mode: str = "none"
    visual_detail_tap: Optional[str] = None
    raw_detail_token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedGenerationOutput:
    """Task-family velocity plus exact logical/attention bookkeeping."""

    velocity: Union[torch.Tensor, sp.SparseTensor]
    logical_sample_ids: Tuple[str, ...]
    logical_sample_map: torch.Tensor
    und_hidden: torch.Tensor
    trace: Optional[UnifiedMoTForwardTrace] = None
    padding_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class FamilyRuntimeState:
    name: str
    base: nn.Module
    hidden: Union[torch.Tensor, sp.SparseTensor]
    timestep_embedding: torch.Tensor
    sample_ids: torch.Tensor
    local_indices: torch.Tensor
    position_ids: torch.Tensor
    role: str
    mlp_modulation: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    @property
    def token_count(self) -> int:
        if isinstance(self.hidden, sp.SparseTensor):
            return int(self.hidden.feats.shape[0])
        return int(self.hidden.shape[0] * self.hidden.shape[1])

    def axis(self) -> TokenAxis:
        return TokenAxis(
            self.name,
            self.sample_ids,
            self.local_indices,
            self.position_ids,
        )


def apply_trainable_linear(module: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        return module(hidden_states)
    input_dtype = hidden_states.dtype
    return module(hidden_states.to(dtype=module.weight.dtype)).to(dtype=input_dtype)


def expand_qwen_grouped_kv(states: torch.Tensor, *, groups: int) -> torch.Tensor:
    if states.ndim != 4:
        raise ValueError("Qwen KV states must have shape [B,T,H_kv,D].")
    return states.repeat_interleave(int(groups), dim=2)


def family_features(hidden: Union[torch.Tensor, sp.SparseTensor]) -> torch.Tensor:
    if isinstance(hidden, sp.SparseTensor):
        return hidden.feats
    return hidden


def family_carrier_view(states: torch.Tensor) -> torch.Tensor:
    if states.ndim == 4:
        return states.reshape(1, states.shape[0] * states.shape[1], states.shape[2], states.shape[3])
    if states.ndim == 3:
        return states.unsqueeze(0)
    raise ValueError(f"Unexpected family carrier shape {tuple(states.shape)}.")


def _layout_equal(left: Sequence[slice], right: Sequence[slice]) -> bool:
    return len(left) == len(right) and all(
        (a.start, a.stop, a.step) == (b.start, b.stop, b.step) for a, b in zip(left, right)
    )


def validate_sparse_alignment(
    noisy: sp.SparseTensor,
    clean: sp.SparseTensor,
    *,
    noisy_name: str,
    clean_name: str,
) -> None:
    if not isinstance(noisy, sp.SparseTensor) or not isinstance(clean, sp.SparseTensor):
        raise TypeError(f"{noisy_name} and {clean_name} must both be native SparseTensor values.")
    if noisy.feats.shape[0] != clean.feats.shape[0]:
        raise ValueError(f"{noisy_name} and {clean_name} token counts differ.")
    if not torch.equal(noisy.coords, clean.coords):
        raise ValueError(f"{noisy_name} and {clean_name} coords differ.")
    if not _layout_equal(noisy.layout, clean.layout):
        raise ValueError(f"{noisy_name} and {clean_name} layouts differ.")


__all__ = [
    "EXPECTED_FROZEN_PARAMETERS",
    "EXPECTED_HAAR_GATED_DETAIL_DELTA",
    "RAW_DETAIL_PROJECTION",
    "VISUAL_DETAIL_SCHEMA_VERSION",
    "expected_haar_gated_detail_delta",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "FamilyRuntimeState",
    "PackedGenerationBatch",
    "TASK_FAMILIES",
    "TASK_SLOT_COUNT",
    "TASK_TARGET_FAMILY",
    "UnifiedGenerationOutput",
    "UnifiedMoTForwardTrace",
    "UnifiedMoTSlotTrace",
    "apply_trainable_linear",
    "expand_qwen_grouped_kv",
    "expected_trainable_parameters",
    "family_carrier_view",
    "family_features",
    "validate_sparse_alignment",
]
