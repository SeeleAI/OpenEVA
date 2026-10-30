"""Shared trainable adapters for EVA01 Unified Generation MoT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detail_condition import (
    DepthTimeTokenGate,
    RawDetailAdapter,
    VISUAL_DETAIL_MODES,
    VISUAL_DETAIL_TAPS,
)
from .runtime_types import RAW_DETAIL_PROJECTION, VISUAL_DETAIL_SCHEMA_VERSION


CARRIER_HEADS = 16
CARRIER_HEAD_DIM = 128
CANONICAL_CONDITION_DIM = 1024
NUM_QWEN_VISION_BLOCKS = 24


@dataclass(frozen=True)
class GenerationFamilySpec:
    """Native projection ABI for one pretrained GEN family."""

    name: str
    hidden_size: int
    num_heads: int
    head_dim: int
    condition_dim: int
    num_blocks: int


GENERATION_FAMILY_SPECS: Mapping[str, GenerationFamilySpec] = {
    "ss": GenerationFamilySpec("ss", 1280, 16, 80, 768, 28),
    "shape": GenerationFamilySpec("shape", 1536, 12, 128, 1024, 30),
    "material": GenerationFamilySpec("material", 1536, 12, 128, 1024, 30),
}


class LoRADelta(nn.Module):
    """A base-free LoRA delta used with an externally owned frozen projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("LoRA dimensions and rank must be positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5, generator=generator)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.in_features:
            raise ValueError(
                f"LoRA input width must be {self.in_features}, got {hidden_states.shape[-1]}."
            )
        input_dtype = hidden_states.dtype
        if torch.is_autocast_enabled():
            output = self.lora_b(self.lora_a(hidden_states))
        else:
            compute_states = hidden_states.to(dtype=self.lora_a.weight.dtype)
            output = self.lora_b(self.lora_a(compute_states)).to(dtype=input_dtype)
        return output * self.scaling


class GenerationRouteCompatibleLoRA(nn.Module):
    """Generation-only LoRA with the trained routed GEMM shape.

    The final Unified All-Task checkpoint selected generation rows through a
    two-bank routed projection.  The inactive bank contributed zeros, but its
    rank was still present in both GEMMs.  Keeping that compute shape avoids
    reduced-precision drift while storing only the released generation bank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("LoRA dimensions and rank must be positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5, generator=generator)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.in_features:
            raise ValueError(
                f"LoRA input width must be {self.in_features}, got {hidden_states.shape[-1]}."
            )
        prefix = tuple(int(value) for value in hidden_states.shape[:-1])
        rows = 1
        for value in prefix:
            rows *= value
        flat = hidden_states.reshape(rows, self.in_features)
        input_dtype = flat.dtype
        compute = (
            flat
            if torch.is_autocast_enabled()
            else flat.to(dtype=self.lora_a.weight.dtype)
        )
        padded_a = torch.cat(
            (self.lora_a.weight, torch.zeros_like(self.lora_a.weight)),
            dim=0,
        )
        low_rank = F.linear(compute, padded_a)
        generation_low, inactive_low = low_rank.split((self.rank, self.rank), dim=-1)
        selected_low = torch.cat(
            (generation_low * 1.0, inactive_low * 0.0),
            dim=-1,
        )
        padded_b = torch.cat(
            (
                self.lora_b.weight * self.scaling,
                torch.zeros_like(self.lora_b.weight),
            ),
            dim=1,
        )
        output = F.linear(selected_low, padded_b)
        if not torch.is_autocast_enabled():
            output = output.to(dtype=input_dtype)
        return output.reshape(*prefix, self.out_features)


def lora_delta_forward(
    delta: Optional[nn.Module],
    hidden_states: torch.Tensor,
    *,
    branch_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Apply one generation LoRA delta."""

    if delta is None:
        return None
    return delta(hidden_states)


class FactorizedHeadDimBridge(nn.Module):
    """Learnable head and head-dimension mapping without padding or truncation."""

    def __init__(
        self,
        in_heads: int,
        in_head_dim: int,
        out_heads: int,
        out_head_dim: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_heads", in_heads),
            ("in_head_dim", in_head_dim),
            ("out_heads", out_heads),
            ("out_head_dim", out_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        self.in_heads = int(in_heads)
        self.in_head_dim = int(in_head_dim)
        self.out_heads = int(out_heads)
        self.out_head_dim = int(out_head_dim)
        self.head_mixer = nn.Parameter(torch.empty(self.out_heads, self.in_heads))
        self.dim_mixer = nn.Parameter(torch.empty(self.out_head_dim, self.in_head_dim))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        nn.init.orthogonal_(self.head_mixer, generator=generator)
        nn.init.orthogonal_(self.dim_mixer, generator=generator)

    def copy_from_(self, other: "FactorizedHeadDimBridge", *, transpose: bool = False) -> None:
        """Copy only initialization while preserving independent trainable parameters."""

        with torch.no_grad():
            source_head = other.head_mixer.T if transpose else other.head_mixer
            source_dim = other.dim_mixer.T if transpose else other.dim_mixer
            if self.head_mixer.shape != source_head.shape or self.dim_mixer.shape != source_dim.shape:
                raise ValueError("Bridge shapes are incompatible for the requested initialization copy.")
            self.head_mixer.copy_(source_head)
            self.dim_mixer.copy_(source_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4:
            raise ValueError(f"Bridge input must have shape [B,T,H,D], got {tuple(states.shape)}.")
        if states.shape[-2:] != (self.in_heads, self.in_head_dim):
            raise ValueError(
                "Bridge input head layout mismatch: "
                f"expected {(self.in_heads, self.in_head_dim)}, got {tuple(states.shape[-2:])}."
            )
        input_dtype = states.dtype
        if torch.is_autocast_enabled():
            return torch.einsum("bthd,oh,ed->btoe", states, self.head_mixer, self.dim_mixer)
        output = torch.einsum(
            "bthd,oh,ed->btoe",
            states.to(dtype=self.head_mixer.dtype),
            self.head_mixer,
            self.dim_mixer,
        )
        return output.to(dtype=input_dtype)


class QwenSlotLoRA(nn.Module):
    """Reachable Qwen Q/K/V/O deltas for one native decoder layer."""

    def __init__(
        self,
        *,
        include_q: bool = True,
        include_k: bool = True,
        include_v: bool = True,
        include_o: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        base_seed = None if seed is None else int(seed)
        self.q = (
            LoRADelta(2048, 2048, rank=16, alpha=16, seed=base_seed)
            if include_q
            else None
        )
        self.k = (
            LoRADelta(
                2048,
                1024,
                rank=16,
                alpha=16,
                seed=None if base_seed is None else base_seed + 1,
            )
            if include_k
            else None
        )
        self.v = (
            LoRADelta(
                2048,
                1024,
                rank=16,
                alpha=16,
                seed=None if base_seed is None else base_seed + 2,
            )
            if include_v
            else None
        )
        self.o = (
            LoRADelta(
                2048,
                2048,
                rank=16,
                alpha=16,
                seed=None if base_seed is None else base_seed + 3,
            )
            if include_o
            else None
        )


class QwenVisionSlotLoRA(nn.Module):
    """Independent rank-16 Q/K/V/O deltas for one frozen Qwen vision block."""

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__()
        base_seed = None if seed is None else int(seed)
        self.q = LoRADelta(1024, 1024, rank=16, alpha=16, seed=base_seed)
        self.k = LoRADelta(
            1024,
            1024,
            rank=16,
            alpha=16,
            seed=None if base_seed is None else base_seed + 1,
        )
        self.v = LoRADelta(
            1024,
            1024,
            rank=16,
            alpha=16,
            seed=None if base_seed is None else base_seed + 2,
        )
        self.o = LoRADelta(
            1024,
            1024,
            rank=16,
            alpha=16,
            seed=None if base_seed is None else base_seed + 3,
        )


class SharedCanonicalConditionAdapter(nn.Module):
    """The only Qwen-hidden condition adapter owner across depth, family, and modality."""

    NUM_DEPTHS = 28

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        self.depth_projections = nn.ModuleList(
            [nn.Linear(2048, CANONICAL_CONDITION_DIM, bias=True) for _ in range(self.NUM_DEPTHS)]
        )
        self.ss_width_bridge = nn.Linear(
            CANONICAL_CONDITION_DIM,
            GENERATION_FAMILY_SPECS["ss"].condition_dim,
            bias=True,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        for projection in self.depth_projections:
            nn.init.xavier_uniform_(projection.weight, generator=generator)
            nn.init.zeros_(projection.bias)
        nn.init.orthogonal_(self.ss_width_bridge.weight, generator=generator)
        nn.init.zeros_(self.ss_width_bridge.bias)

    def canonical(self, slot_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        depth = min(max(int(slot_index), 0), self.NUM_DEPTHS - 1)
        projection = self.depth_projections[depth]
        input_dtype = hidden_states.dtype
        if torch.is_autocast_enabled():
            return projection(hidden_states)
        return projection(hidden_states.to(dtype=projection.weight.dtype)).to(dtype=input_dtype)

    def for_family(self, family_name: str, canonical_states: torch.Tensor) -> torch.Tensor:
        normalized = str(family_name).strip().lower()
        if normalized == "ss":
            input_dtype = canonical_states.dtype
            if torch.is_autocast_enabled():
                return self.ss_width_bridge(canonical_states)
            return self.ss_width_bridge(
                canonical_states.to(dtype=self.ss_width_bridge.weight.dtype)
            ).to(dtype=input_dtype)
        if normalized in {"shape", "material"}:
            return canonical_states
        raise KeyError(f"Unknown generation family `{family_name}`.")


class GenerationFamilySlotAdapters(nn.Module):
    """MoT-facing Q/K/V/O LoRA and carrier bridges for one GEN block."""

    def __init__(
        self,
        spec: GenerationFamilySpec,
        *,
        seed: int,
        dual_lora: bool = False,
        routed_lora: bool = False,
        generation_only: bool = False,
        gen_rank: int = 64,
    ) -> None:
        super().__init__()
        self.spec = spec
        hidden = int(spec.hidden_size)
        if dual_lora or routed_lora or not generation_only:
            raise ValueError("The public adapter registry supports generation_v1 only.")
        delta = GenerationRouteCompatibleLoRA
        kwargs = {"rank": int(gen_rank), "alpha": int(gen_rank)}
        self.q = delta(hidden, hidden, seed=seed + 10, **kwargs)
        self.k = (
            LoRADelta(hidden, hidden, rank=32, alpha=32, seed=seed + 11)
            if routed_lora or generation_only
            else delta(hidden, hidden, seed=seed + 11, **kwargs)
        )
        self.v = (
            LoRADelta(hidden, hidden, rank=32, alpha=32, seed=seed + 12)
            if routed_lora or generation_only
            else delta(hidden, hidden, seed=seed + 12, **kwargs)
        )
        self.o = delta(hidden, hidden, seed=seed + 13, **kwargs)
        self.q_bridge = FactorizedHeadDimBridge(
            spec.num_heads,
            spec.head_dim,
            CARRIER_HEADS,
            CARRIER_HEAD_DIM,
            seed=seed,
        )
        self.k_bridge = FactorizedHeadDimBridge(
            spec.num_heads,
            spec.head_dim,
            CARRIER_HEADS,
            CARRIER_HEAD_DIM,
            seed=seed + 1,
        )
        self.v_bridge = FactorizedHeadDimBridge(
            spec.num_heads,
            spec.head_dim,
            CARRIER_HEADS,
            CARRIER_HEAD_DIM,
            seed=seed + 2,
        )
        self.o_bridge = FactorizedHeadDimBridge(
            CARRIER_HEADS,
            CARRIER_HEAD_DIM,
            spec.num_heads,
            spec.head_dim,
            seed=seed + 3,
        )
        self.k_bridge.copy_from_(self.q_bridge)
        self.o_bridge.copy_from_(self.v_bridge, transpose=True)


class FamilyConditionSlotAdapters(nn.Module):
    """Family-native K/V projection after the one shared canonical condition adapter."""

    def __init__(
        self,
        spec: GenerationFamilySpec,
        *,
        seed: int,
        family_adapters: GenerationFamilySlotAdapters,
        dual_lora: bool = False,
        routed_lora: bool = False,
        generation_only: bool = False,
        gen_rank: int = 64,
    ) -> None:
        super().__init__()
        self.spec = spec
        if dual_lora or routed_lora or not generation_only:
            raise ValueError("The public adapter registry supports generation_v1 only.")
        if generation_only:
            self.cross_kv = LoRADelta(
                spec.condition_dim,
                2 * spec.hidden_size,
                rank=int(gen_rank),
                alpha=int(gen_rank),
                seed=seed + 10,
            )
        else:
            self.cross_kv = LoRADelta(
                spec.condition_dim,
                2 * spec.hidden_size,
                rank=32,
                alpha=32,
                seed=seed + 10,
            )
        self.k_bridge = FactorizedHeadDimBridge(
            spec.num_heads,
            spec.head_dim,
            CARRIER_HEADS,
            CARRIER_HEAD_DIM,
            seed=seed + 1,
        )
        self.v_bridge = FactorizedHeadDimBridge(
            spec.num_heads,
            spec.head_dim,
            CARRIER_HEADS,
            CARRIER_HEAD_DIM,
            seed=seed + 2,
        )
        self.k_bridge.copy_from_(family_adapters.q_bridge)
        self.v_bridge.copy_from_(family_adapters.v_bridge)


# Compatibility names for existing t2ss imports. They are aliases of the shared
# family implementation and do not create a task-specific parameter bank.
SSSlotAdapters = GenerationFamilySlotAdapters
SSConditionSlotAdapters = FamilyConditionSlotAdapters
T2SSSlotAdapters = GenerationFamilySlotAdapters


class EVA01AdapterRegistry(nn.Module):
    """Single shared adapter owner for every active generation family."""

    NUM_QWEN_LAYERS = 28

    def __init__(
        self,
        *,
        variant: str,
        active_families: Sequence[str] = ("ss",),
        task_scope: str = "t2ss",
        seed: int = 20260727,
        visual_detail_mode: str = "none",
        visual_detail_tap: str = "block5",
        raw_detail_projection: str = RAW_DETAIL_PROJECTION,
        raw_detail_init_sha256: Optional[str] = None,
        detail_gate_hidden: int = 128,
        detail_gate_max: float = 0.25,
    ) -> None:
        super().__init__()
        normalized_variant = str(variant).strip().upper()
        if normalized_variant not in {"A", "B"}:
            raise ValueError("condition variant must be `A` or `B`.")
        normalized_families = tuple(dict.fromkeys(str(name).strip().lower() for name in active_families))
        unknown = sorted(set(normalized_families).difference(GENERATION_FAMILY_SPECS))
        if unknown:
            raise ValueError(f"Unknown generation families: {unknown}.")
        if not normalized_families or normalized_families[0] != "ss":
            raise ValueError("The shared family schema must begin with `ss`.")
        normalized_scope = str(task_scope).strip().lower()
        expected_families = {
            "t2ss": ("ss",),
            "t2shape": ("ss", "shape"),
            "t2sm": ("ss", "shape", "material"),
            "all": ("ss", "shape", "material"),
        }
        if normalized_scope not in expected_families:
            raise ValueError(f"Unsupported adapter task_scope `{task_scope}`.")
        if normalized_families != expected_families[normalized_scope]:
            raise ValueError(
                f"task_scope {normalized_scope} requires families {expected_families[normalized_scope]}, "
                f"got {normalized_families}."
            )
        self.variant = normalized_variant
        self.active_families = normalized_families
        self.task_scope = normalized_scope
        self.seed = int(seed)
        normalized_detail_mode = str(visual_detail_mode).strip().lower()
        normalized_detail_tap = str(visual_detail_tap).strip().lower()
        if normalized_detail_mode not in VISUAL_DETAIL_MODES:
            raise ValueError(f"Unknown visual_detail_mode `{visual_detail_mode}`.")
        if normalized_detail_tap not in VISUAL_DETAIL_TAPS:
            raise ValueError(f"Unknown visual_detail_tap `{visual_detail_tap}`.")
        if self.variant != "B" and normalized_detail_mode != "none":
            raise ValueError("Raw visual detail is defined only for the shared Variant-B condition view.")
        self.visual_detail_mode = normalized_detail_mode
        self.visual_detail_tap = normalized_detail_tap
        self.raw_detail_projection = (
            str(raw_detail_projection)
            if normalized_detail_mode == "haar_gated"
            else "none"
        )
        self.raw_detail_init_sha256 = (
            None
            if normalized_detail_mode == "none"
            else str(raw_detail_init_sha256 or "")
        )
        if normalized_detail_mode == "haar_gated":
            if self.raw_detail_projection != RAW_DETAIL_PROJECTION:
                raise ValueError("Detail-B-v2 requires the direct PCA projection schema.")
            if len(self.raw_detail_init_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.raw_detail_init_sha256.lower()
            ):
                raise ValueError("Detail-B-v2 requires one valid PCA initialization SHA256.")
        self.detail_gate_hidden = int(detail_gate_hidden)
        self.detail_gate_max = float(detail_gate_max)

        qwen_slots = []
        for slot_index in range(self.NUM_QWEN_LAYERS):
            if slot_index != self.NUM_QWEN_LAYERS - 1 or self.task_scope != "t2ss":
                include = (True, True, True, True)
            elif self.variant == "A":
                # Slot 27 K/V is consumed by the final SS block; Q/O is not.
                include = (False, True, True, False)
            else:
                # Variant B's final SS condition view reads the hidden entering
                # slot 27, so no slot-27 Q/K/V/O delta is reachable in t2ss.
                include = (False, False, False, False)
            qwen_slots.append(
                QwenSlotLoRA(
                    include_q=include[0],
                    include_k=include[1],
                    include_v=include[2],
                    include_o=include[3],
                    seed=self.seed + slot_index * 10,
                )
            )
        self.qwen = nn.ModuleList(qwen_slots)
        if self.variant == "B":
            self.vision = nn.ModuleList(
                [
                    QwenVisionSlotLoRA(seed=self.seed + 1_000_000 + block_index * 10)
                    for block_index in range(NUM_QWEN_VISION_BLOCKS)
                ]
            )
            self.canonical_condition = SharedCanonicalConditionAdapter(
                seed=self.seed + 9_000_000
            )
            if self.visual_detail_mode == "haar_gated":
                self.raw_detail_adapter = RawDetailAdapter(
                    seed=self.seed + 10_000_000,
                )
                self.detail_gate = DepthTimeTokenGate(
                    hidden_dim=self.detail_gate_hidden,
                    g_max=self.detail_gate_max,
                    seed=self.seed + 11_000_000,
                )
            else:
                self.raw_detail_adapter = None
                self.detail_gate = None
        else:
            self.vision = nn.ModuleList()
            self.canonical_condition = None
            self.raw_detail_adapter = None
            self.detail_gate = None

        family_banks: Dict[str, nn.ModuleList] = {}
        condition_banks: Dict[str, nn.ModuleList] = {}
        for family_offset, family_name in enumerate(self.active_families):
            spec = GENERATION_FAMILY_SPECS[family_name]
            slots = nn.ModuleList(
                [
                    GenerationFamilySlotAdapters(
                        spec,
                        seed=self.seed + family_offset * 100_000 + slot_index * 1000,
                    )
                    for slot_index in range(spec.num_blocks)
                ]
            )
            family_banks[family_name] = slots
            if self.variant == "B":
                condition_banks[family_name] = nn.ModuleList(
                    [
                        FamilyConditionSlotAdapters(
                            spec,
                            seed=self.seed + family_offset * 100_000 + slot_index * 1000 + 100,
                            family_adapters=slots[slot_index],
                        )
                        for slot_index in range(spec.num_blocks)
                    ]
                )
        self.families = nn.ModuleDict(family_banks)
        self.conditions = nn.ModuleDict(condition_banks)

    def qwen_slot(self, slot_index: int) -> QwenSlotLoRA:
        return self.qwen[int(slot_index)]

    def family_slot(self, family_name: str, slot_index: int) -> GenerationFamilySlotAdapters:
        return self.families[str(family_name)][int(slot_index)]

    def condition_slot(self, family_name: str, slot_index: int) -> FamilyConditionSlotAdapters:
        if self.variant != "B":
            raise RuntimeError("Variant A does not own condition-view adapters.")
        return self.conditions[str(family_name)][int(slot_index)]

    def vision_slot(self, block_index: int) -> QwenVisionSlotLoRA:
        if self.variant != "B":
            raise RuntimeError("Historical Variant A does not own multimodal vision LoRA.")
        return self.vision[int(block_index)]

    def canonical_condition_states(
        self,
        slot_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant != "B" or self.canonical_condition is None:
            raise RuntimeError("Only Variant B owns the canonical condition adapter.")
        return self.canonical_condition.canonical(slot_index, hidden_states)

    def family_condition_states(
        self,
        family_name: str,
        canonical_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant != "B" or self.canonical_condition is None:
            raise RuntimeError("Only Variant B owns the canonical condition adapter.")
        return self.canonical_condition.for_family(family_name, canonical_states)

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self.state_dict())

    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Strict adapter load produced incompatible keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )

    def named_parameter_groups(self) -> Dict[str, Tuple[str, ...]]:
        groups: Dict[str, list[str]] = {
            "und_lora": [],
            "vision_lora": [],
            "ss_mot_lora": [],
            "shape_mot_lora": [],
            "material_mot_lora": [],
            "carrier_bridge": [],
            "condition_adapter": [],
            "condition_kv_lora": [],
            "condition_bridge": [],
            "raw_detail_adapter": [],
            "detail_gate": [],
        }
        groups["und_lora"].extend(name for name, _ in self.qwen.named_parameters())
        groups["und_lora"] = [f"qwen.{name}" for name in groups["und_lora"]]
        groups["vision_lora"].extend(
            f"vision.{name}" for name, _ in self.vision.named_parameters()
        )
        if self.canonical_condition is not None:
            groups["condition_adapter"].extend(
                f"canonical_condition.{name}"
                for name, _ in self.canonical_condition.named_parameters()
            )
        if self.raw_detail_adapter is not None:
            groups["raw_detail_adapter"].extend(
                f"raw_detail_adapter.{name}"
                for name, _ in self.raw_detail_adapter.named_parameters()
            )
        if self.detail_gate is not None:
            groups["detail_gate"].extend(
                f"detail_gate.{name}" for name, _ in self.detail_gate.named_parameters()
            )
        for family_name, bank in self.families.items():
            for name, _ in bank.named_parameters():
                full_name = f"families.{family_name}.{name}"
                target = "carrier_bridge" if "bridge" in name else f"{family_name}_mot_lora"
                groups[target].append(full_name)
        for family_name, bank in self.conditions.items():
            for name, _ in bank.named_parameters():
                full_name = f"conditions.{family_name}.{name}"
                if "cross_kv" in name:
                    target = "condition_kv_lora"
                else:
                    target = "condition_bridge"
                groups[target].append(full_name)
        return {name: tuple(values) for name, values in groups.items() if values}

    def iter_group_parameters(self, group_name: str) -> Iterator[nn.Parameter]:
        group_names = self.named_parameter_groups()
        if group_name not in group_names:
            raise KeyError(f"Unknown or inactive adapter group `{group_name}`.")
        parameters = dict(self.named_parameters())
        for name in group_names[group_name]:
            yield parameters[name]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def schema_inventory(self) -> Dict[str, object]:
        state = self.adapter_state_dict()
        return {
            "variant": self.variant,
            "task_scope": self.task_scope,
            "visual_detail_mode": self.visual_detail_mode,
            "visual_detail_tap": self.visual_detail_tap,
            "visual_detail_schema_version": (
                VISUAL_DETAIL_SCHEMA_VERSION if self.visual_detail_mode == "haar_gated" else 0
            ),
            "raw_detail_projection": self.raw_detail_projection,
            "raw_detail_init_sha256": self.raw_detail_init_sha256,
            "detail_gate_hidden": self.detail_gate_hidden,
            "detail_gate_max": self.detail_gate_max,
            "adapter_numel": sum(value.numel() for value in state.values()),
            "adapter_key_count": len(state),
            # Checkpoint metadata is JSON.  Returning a tuple here makes the
            # freshly constructed runtime schema compare unequal to the same
            # schema after a JSON save/load round trip (tuple versus list),
            # which prevents every strict adapter-only resume.
            "adapter_keys": sorted(state),
        }


class _MergedQwenSlot(nn.Module):
    """Projection markers for a release model whose Qwen LoRA is already folded."""

    def __init__(self) -> None:
        super().__init__()
        self.q = None
        self.k = None
        self.v = None
        self.o = None


class _MergedFamilySlot(nn.Module):
    """Non-foldable carrier bridges retained after native GEN LoRA merge."""

    def __init__(self, spec: GenerationFamilySpec, *, seed: int) -> None:
        super().__init__()
        self.spec = spec
        self.q = None
        self.k = None
        self.v = None
        self.o = None
        self.q_bridge = FactorizedHeadDimBridge(
            spec.num_heads, spec.head_dim, CARRIER_HEADS, CARRIER_HEAD_DIM, seed=seed
        )
        self.k_bridge = FactorizedHeadDimBridge(
            spec.num_heads, spec.head_dim, CARRIER_HEADS, CARRIER_HEAD_DIM, seed=seed + 1
        )
        self.v_bridge = FactorizedHeadDimBridge(
            spec.num_heads, spec.head_dim, CARRIER_HEADS, CARRIER_HEAD_DIM, seed=seed + 2
        )
        self.o_bridge = FactorizedHeadDimBridge(
            CARRIER_HEADS, CARRIER_HEAD_DIM, spec.num_heads, spec.head_dim, seed=seed + 3
        )


class _MergedConditionSlot(nn.Module):
    """Variant-B ordinary condition layers after cross-KV LoRA is folded."""

    def __init__(self, spec: GenerationFamilySpec, *, seed: int) -> None:
        super().__init__()
        self.spec = spec
        self.cross_kv = None
        self.k_bridge = FactorizedHeadDimBridge(
            spec.num_heads, spec.head_dim, CARRIER_HEADS, CARRIER_HEAD_DIM, seed=seed + 1
        )
        self.v_bridge = FactorizedHeadDimBridge(
            spec.num_heads, spec.head_dim, CARRIER_HEADS, CARRIER_HEAD_DIM, seed=seed + 2
        )


class MergedEVA01AdapterRegistry(nn.Module):
    """Release-time ordinary bridges/condition layers with no LoRA module or key."""

    NUM_QWEN_LAYERS = 28

    def __init__(
        self,
        *,
        variant: str,
        active_families: Sequence[str],
        task_scope: str,
        seed: int = 20260727,
        visual_detail_mode: str = "none",
        visual_detail_tap: str = "block5",
        raw_detail_projection: str = RAW_DETAIL_PROJECTION,
        raw_detail_init_sha256: Optional[str] = None,
        detail_gate_hidden: int = 128,
        detail_gate_max: float = 0.25,
    ) -> None:
        super().__init__()
        normalized_variant = str(variant).strip().upper()
        normalized_families = tuple(str(value).strip().lower() for value in active_families)
        normalized_scope = str(task_scope).strip().lower()
        expected = {
            "t2ss": ("ss",),
            "t2shape": ("ss", "shape"),
            "t2sm": ("ss", "shape", "material"),
            "all": ("ss", "shape", "material"),
        }
        if normalized_variant not in {"A", "B"}:
            raise ValueError("Merged variant must be A or B.")
        if expected.get(normalized_scope) != normalized_families:
            raise ValueError("Merged registry task scope/family schema mismatch.")
        self.variant = normalized_variant
        self.active_families = normalized_families
        self.task_scope = normalized_scope
        self.seed = int(seed)
        self.visual_detail_mode = str(visual_detail_mode).strip().lower()
        self.visual_detail_tap = str(visual_detail_tap).strip().lower()
        self.raw_detail_projection = (
            str(raw_detail_projection) if self.visual_detail_mode == "haar_gated" else "none"
        )
        self.raw_detail_init_sha256 = (
            None if self.visual_detail_mode == "none" else str(raw_detail_init_sha256 or "")
        )
        self.detail_gate_hidden = int(detail_gate_hidden)
        self.detail_gate_max = float(detail_gate_max)
        if self.visual_detail_mode not in VISUAL_DETAIL_MODES:
            raise ValueError("Merged visual-detail mode is invalid.")
        if self.visual_detail_tap not in VISUAL_DETAIL_TAPS:
            raise ValueError("Merged visual-detail tap is invalid.")
        if self.variant != "B" and self.visual_detail_mode != "none":
            raise ValueError("Merged raw visual detail is Variant-B only.")
        if self.visual_detail_mode == "haar_gated":
            if self.raw_detail_projection != RAW_DETAIL_PROJECTION:
                raise ValueError("Merged Detail-B-v2 projection schema is invalid.")
            if len(self.raw_detail_init_sha256) != 64:
                raise ValueError("Merged Detail-B-v2 requires its PCA initialization SHA256.")
        self.qwen = nn.ModuleList([_MergedQwenSlot() for _ in range(self.NUM_QWEN_LAYERS)])
        self.vision = nn.ModuleList()
        self.canonical_condition = (
            SharedCanonicalConditionAdapter(seed=self.seed + 9_000_000)
            if self.variant == "B"
            else None
        )
        if self.visual_detail_mode == "haar_gated":
            self.raw_detail_adapter = RawDetailAdapter(
                seed=self.seed + 10_000_000,
            )
            self.detail_gate = DepthTimeTokenGate(
                hidden_dim=self.detail_gate_hidden,
                g_max=self.detail_gate_max,
                seed=self.seed + 11_000_000,
            )
        else:
            self.raw_detail_adapter = None
            self.detail_gate = None
        self.families = nn.ModuleDict(
            {
                family_name: nn.ModuleList(
                    [
                        _MergedFamilySlot(
                            GENERATION_FAMILY_SPECS[family_name],
                            seed=self.seed + family_offset * 100_000 + slot_index * 1000,
                        )
                        for slot_index in range(GENERATION_FAMILY_SPECS[family_name].num_blocks)
                    ]
                )
                for family_offset, family_name in enumerate(self.active_families)
            }
        )
        self.conditions = nn.ModuleDict()
        if self.variant == "B":
            self.conditions = nn.ModuleDict(
                {
                    family_name: nn.ModuleList(
                        [
                            _MergedConditionSlot(
                                GENERATION_FAMILY_SPECS[family_name],
                                seed=self.seed + family_offset * 100_000 + slot_index * 1000 + 100,
                            )
                            for slot_index in range(
                                GENERATION_FAMILY_SPECS[family_name].num_blocks
                            )
                        ]
                    )
                    for family_offset, family_name in enumerate(self.active_families)
                }
            )

    def qwen_slot(self, slot_index: int) -> _MergedQwenSlot:
        return self.qwen[int(slot_index)]

    def family_slot(self, family_name: str, slot_index: int) -> _MergedFamilySlot:
        return self.families[str(family_name)][int(slot_index)]

    def condition_slot(self, family_name: str, slot_index: int) -> _MergedConditionSlot:
        if self.variant != "B":
            raise RuntimeError("Variant A has no merged condition view.")
        return self.conditions[str(family_name)][int(slot_index)]

    def canonical_condition_states(
        self,
        slot_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant != "B" or self.canonical_condition is None:
            raise RuntimeError("Only merged Variant B owns the canonical condition adapter.")
        return self.canonical_condition.canonical(slot_index, hidden_states)

    def family_condition_states(
        self,
        family_name: str,
        canonical_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant != "B" or self.canonical_condition is None:
            raise RuntimeError("Only merged Variant B owns the canonical condition adapter.")
        return self.canonical_condition.for_family(family_name, canonical_states)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def linear_with_lora(
    hidden_states: torch.Tensor,
    base_projection: nn.Linear,
    delta: Optional[LoRADelta],
    *,
    branch_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a frozen base projection plus an optional reachable LoRA delta."""

    base_output = base_projection(hidden_states)
    delta_output = lora_delta_forward(
        delta,
        hidden_states,
        branch_mask=branch_mask,
    )
    if delta_output is None:
        return base_output
    return base_output + delta_output.to(dtype=base_output.dtype)


def sliced_linear_with_lora(
    hidden_states: torch.Tensor,
    *,
    fused_projection: nn.Linear,
    slice_index: int,
    slice_count: int,
    delta: Optional[LoRADelta],
    branch_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply one frozen fused-projection slice plus an optional LoRA delta."""

    if not 0 <= slice_index < slice_count:
        raise ValueError("slice_index is outside the fused projection partition.")
    if fused_projection.out_features % slice_count != 0:
        raise ValueError("Fused projection output width is not divisible by slice_count.")
    width = fused_projection.out_features // slice_count
    start = slice_index * width
    end = start + width
    bias = None if fused_projection.bias is None else fused_projection.bias[start:end]
    base_output = F.linear(hidden_states, fused_projection.weight[start:end], bias)
    delta_output = lora_delta_forward(
        delta,
        hidden_states,
        branch_mask=branch_mask,
    )
    if delta_output is None:
        return base_output
    return base_output + delta_output.to(dtype=base_output.dtype)


__all__ = [
    "CARRIER_HEAD_DIM",
    "CARRIER_HEADS",
    "EVA01AdapterRegistry",
    "FactorizedHeadDimBridge",
    "FamilyConditionSlotAdapters",
    "GENERATION_FAMILY_SPECS",
    "GenerationRouteCompatibleLoRA",
    "GenerationFamilySlotAdapters",
    "GenerationFamilySpec",
    "LoRADelta",
    "lora_delta_forward",
    "MergedEVA01AdapterRegistry",
    "QwenSlotLoRA",
    "SSConditionSlotAdapters",
    "SSSlotAdapters",
    "T2SSSlotAdapters",
    "linear_with_lora",
    "sliced_linear_with_lora",
]
