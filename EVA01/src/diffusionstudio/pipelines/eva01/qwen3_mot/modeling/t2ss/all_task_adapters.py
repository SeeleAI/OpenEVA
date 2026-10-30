"""Trainable owner registry for EVA01 Unified All-Task V2."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import (
    CANONICAL_CONDITION_DIM,
    CARRIER_HEAD_DIM,
    CARRIER_HEADS,
    GENERATION_FAMILY_SPECS,
    FactorizedHeadDimBridge,
    GenerationRouteCompatibleLoRA,
    GenerationFamilySlotAdapters,
    LoRADelta,
    QwenSlotLoRA,
    QwenVisionSlotLoRA,
)
from .all_task_tokens import (
    SPECIAL_EMBEDDING_TOKEN_IDS,
    select_special_embedding_rows,
)


ALL_TASK_SCHEMA = "eva01_unified_all_task_v2"
ALL_TASK_VARIANT = "B"
ALL_TASK_FAMILIES = ("ss", "shape", "material")
ALL_TASK_TRAINABLE_NUMEL = 127_552_768
ALL_TASK_TRAINABLE_KEYS = 2_065
ALL_TASK_FROZEN_NUMEL = 5_709_388_488
ALL_TASK_TOTAL_NUMEL = 5_836_941_256
ALL_TASK_INHERITED_FREEZE_STEPS = 5_000
GENERATION_BRANCH_SCHEMA = "generation_v1"
ROUTED_TRAINABLE_NUMEL = 209_243_392
ROUTED_TRAINABLE_KEYS = 2_593


def carrier_rms(states: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    """Parameter-free per-head RMS used by every non-Qwen carrier Q/K."""

    return states * torch.rsqrt(
        states.float().square().mean(dim=-1, keepdim=True) + eps
    ).to(dtype=states.dtype)


class SlotCanonicalConditionAdapter(nn.Module):
    """Thirty independent 2048->1024 slot projections and three family readers.

    Slots 0..27 and the SS reader are migrated bit-for-bit from Current-B.
    Slots 28/29 and the Shape/Material readers are new trainable owners.
    """

    NUM_DEPTHS = 30

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        self.depth_projections = nn.ModuleList(
            [
                nn.Linear(2048, CANONICAL_CONDITION_DIM, bias=True)
                for _ in range(self.NUM_DEPTHS)
            ]
        )
        self.ss_width_bridge = nn.Linear(
            CANONICAL_CONDITION_DIM,
            GENERATION_FAMILY_SPECS["ss"].condition_dim,
            bias=True,
        )
        self.shape_width_bridge = nn.Linear(
            CANONICAL_CONDITION_DIM,
            GENERATION_FAMILY_SPECS["shape"].condition_dim,
            bias=True,
        )
        self.material_width_bridge = nn.Linear(
            CANONICAL_CONDITION_DIM,
            GENERATION_FAMILY_SPECS["material"].condition_dim,
            bias=True,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        for projection in self.depth_projections:
            nn.init.xavier_uniform_(projection.weight, generator=generator)
            nn.init.zeros_(projection.bias)
        nn.init.orthogonal_(self.ss_width_bridge.weight, generator=generator)
        nn.init.zeros_(self.ss_width_bridge.bias)
        for reader in (self.shape_width_bridge, self.material_width_bridge):
            nn.init.eye_(reader.weight)
            nn.init.zeros_(reader.bias)

    def canonical(self, slot_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        depth = int(slot_index)
        if not 0 <= depth < self.NUM_DEPTHS:
            raise IndexError(f"Condition slot {depth} is outside [0,{self.NUM_DEPTHS}).")
        projection = self.depth_projections[depth]
        input_dtype = hidden_states.dtype
        if torch.is_autocast_enabled():
            return projection(hidden_states)
        return projection(hidden_states.to(dtype=projection.weight.dtype)).to(
            dtype=input_dtype
        )

    def for_family(
        self,
        family_name: str,
        canonical_states: torch.Tensor,
    ) -> torch.Tensor:
        reader = getattr(self, f"{str(family_name)}_width_bridge")
        input_dtype = canonical_states.dtype
        if torch.is_autocast_enabled():
            return reader(canonical_states)
        return reader(canonical_states.to(dtype=reader.weight.dtype)).to(
            dtype=input_dtype
        )


class ResidualHeadPadBridge(nn.Module):
    """Fixed 12<->16 head padding plus one zero-init trainable residual.

    Head width stays 128.  The frozen path is exact at step zero: native-to-
    carrier keeps heads 0..11 and appends four zeros; carrier-to-native keeps
    heads 0..11.  Each Q/K/V/O owner has its own [16,12] residual matrix.
    """

    def __init__(self, *, to_carrier: bool) -> None:
        super().__init__()
        self.to_carrier = bool(to_carrier)
        self.head_residual = nn.Parameter(torch.zeros(CARRIER_HEADS, 12))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4 or int(states.shape[-1]) != CARRIER_HEAD_DIM:
            raise ValueError("Residual head bridge expects [B,T,H,128].")
        compute = states if torch.is_autocast_enabled() else states.to(
            dtype=self.head_residual.dtype
        )
        if self.to_carrier:
            if int(states.shape[-2]) != 12:
                raise ValueError("Native-to-carrier bridge expects twelve heads.")
            fixed = F.pad(compute, (0, 0, 0, CARRIER_HEADS - 12))
            residual = torch.einsum(
                "bthd,oh->btod",
                compute,
                self.head_residual,
            )
        else:
            if int(states.shape[-2]) != CARRIER_HEADS:
                raise ValueError("Carrier-to-native bridge expects sixteen heads.")
            fixed = compute[:, :, :12, :]
            residual = torch.einsum(
                "btod,oh->bthd",
                compute,
                self.head_residual,
            )
        return (fixed + residual).to(dtype=states.dtype)


class SparseFamilySlotAdapters(nn.Module):
    """Shape/Material Q/K/V/O LoRA with zero-preserving head bridges."""

    def __init__(
        self,
        family: str,
        *,
        seed: int,
        dual_lora: bool = False,
        routed_lora: bool = False,
        generation_only: bool = False,
        gen_rank: int = 64,
    ) -> None:
        super().__init__()
        spec = GENERATION_FAMILY_SPECS[str(family)]
        if spec.num_heads != 12 or spec.head_dim != CARRIER_HEAD_DIM:
            raise ValueError("Sparse family bridge ABI must be 12 heads x 128.")
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
        self.q_bridge = ResidualHeadPadBridge(to_carrier=True)
        self.k_bridge = ResidualHeadPadBridge(to_carrier=True)
        self.v_bridge = ResidualHeadPadBridge(to_carrier=True)
        self.o_bridge = ResidualHeadPadBridge(to_carrier=False)


class FamilyConditionSlotAdapters(nn.Module):
    """Family-native pretrained cross-KV reader plus trainable delta/bridge."""

    def __init__(
        self,
        family: str,
        *,
        seed: int,
        dual_lora: bool = False,
        routed_lora: bool = False,
        generation_only: bool = False,
        gen_rank: int = 64,
    ) -> None:
        super().__init__()
        spec = GENERATION_FAMILY_SPECS[str(family)]
        self.spec = spec
        if dual_lora or routed_lora or not generation_only:
            raise ValueError("The public adapter registry supports generation_v1 only.")
        if generation_only:
            self.cross_kv = GenerationRouteCompatibleLoRA(
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
        if family == "ss":
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
        else:
            self.k_bridge = ResidualHeadPadBridge(to_carrier=True)
            self.v_bridge = ResidualHeadPadBridge(to_carrier=True)


class OpenEVAConnector(nn.Module):
    """The OpenEVA 384->1024->2048->2048 connector ABI."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        if point_features.shape[-1] != 384:
            raise ValueError("OpenEVA connector expects 384-wide PointBERT features.")
        return self.layers(point_features)


class AllTaskAdapterRegistry(nn.Module):
    """Exactly the trainable owner set for Unified All-Task V2."""

    def __init__(
        self,
        *,
        seed: int = 20260820,
        branch_schema: str = "shared_v2",
        gen_rank: int = 64,
    ) -> None:
        super().__init__()
        self.variant = ALL_TASK_VARIANT
        self.task_scope = "all_task"
        self.active_families = ALL_TASK_FAMILIES
        self.seed = int(seed)
        self.branch_schema = str(branch_schema).strip().lower()
        if self.branch_schema != GENERATION_BRANCH_SCHEMA:
            raise ValueError(f"Unknown all-task branch schema `{branch_schema}`.")
        self.dual_lora = False
        self.routed_lora = False
        self.generation_only = self.branch_schema == GENERATION_BRANCH_SCHEMA
        self.gen_rank = int(gen_rank)
        if self.gen_rank <= 0:
            raise ValueError("Generation LoRA rank must be positive.")
        self.visual_detail_mode = "none"
        self.visual_detail_tap = "block5"
        self.raw_detail_projection = "none"
        self.raw_detail_init_sha256 = None
        self.detail_gate_hidden = 0
        self.detail_gate_max = 0.0
        self.raw_detail_adapter = None
        self.detail_gate = None
        self.qwen = nn.ModuleList(
            [QwenSlotLoRA(seed=self.seed + 10 * index) for index in range(28)]
        )
        self.vision = nn.ModuleList(
            [
                QwenVisionSlotLoRA(seed=self.seed + 1_000_000 + 10 * index)
                for index in range(24)
            ]
        )
        self.connector = OpenEVAConnector()
        self.special_token_embeddings = nn.Embedding(
            len(SPECIAL_EMBEDDING_TOKEN_IDS),
            2048,
        )
        nn.init.zeros_(self.special_token_embeddings.weight)
        self._special_token_embeddings_initialized = False
        self.canonical_condition = SlotCanonicalConditionAdapter(
            seed=self.seed + 9_000_000
        )

        family_banks: Dict[str, nn.ModuleList] = {}
        condition_banks: Dict[str, nn.ModuleList] = {}
        for family_offset, family_name in enumerate(ALL_TASK_FAMILIES):
            spec = GENERATION_FAMILY_SPECS[family_name]
            family_banks[family_name] = nn.ModuleList(
                [
                    (
                        GenerationFamilySlotAdapters(
                            spec,
                            seed=self.seed
                            + family_offset * 100_000
                            + 1_000 * index,
                            dual_lora=self.dual_lora,
                            routed_lora=self.routed_lora,
                            generation_only=self.generation_only,
                            gen_rank=self.gen_rank,
                        )
                        if family_name == "ss"
                        else SparseFamilySlotAdapters(
                            family_name,
                            seed=self.seed
                            + family_offset * 100_000
                            + 1_000 * index,
                            dual_lora=self.dual_lora,
                            routed_lora=self.routed_lora,
                            generation_only=self.generation_only,
                            gen_rank=self.gen_rank,
                        )
                    )
                    for index in range(spec.num_blocks)
                ]
            )
            condition_banks[family_name] = nn.ModuleList(
                [
                    FamilyConditionSlotAdapters(
                        family_name,
                        seed=self.seed
                        + family_offset * 100_000
                        + 1_000 * index
                        + 100,
                        dual_lora=self.dual_lora,
                        routed_lora=self.routed_lora,
                        generation_only=self.generation_only,
                        gen_rank=self.gen_rank,
                    )
                    for index in range(spec.num_blocks)
                ]
            )
        self.families = nn.ModuleDict(family_banks)
        self.conditions = nn.ModuleDict(condition_banks)
        if self.branch_schema == "shared_v2" and self.trainable_parameter_count != ALL_TASK_TRAINABLE_NUMEL:
            raise RuntimeError(
                "Unified All-Task adapter inventory drifted: "
                f"expected {ALL_TASK_TRAINABLE_NUMEL}, got {self.trainable_parameter_count}."
            )
        if self.branch_schema == "shared_v2" and len(tuple(self.state_dict())) != ALL_TASK_TRAINABLE_KEYS:
            raise RuntimeError(
                "Unified All-Task adapter key inventory drifted: "
                f"expected {ALL_TASK_TRAINABLE_KEYS}, got {len(tuple(self.state_dict()))}."
            )

    def qwen_slot(self, index: int) -> QwenSlotLoRA:
        return self.qwen[int(index)]

    def vision_slot(self, index: int) -> QwenVisionSlotLoRA:
        return self.vision[int(index)]

    def family_slot(self, family: str, index: int) -> nn.Module:
        return self.families[str(family)][int(index)]

    def condition_slot(self, family: str, index: int) -> FamilyConditionSlotAdapters:
        return self.conditions[str(family)][int(index)]

    def canonical_condition_states(
        self,
        slot_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.canonical_condition.canonical(slot_index, hidden_states)

    def family_condition_states(
        self,
        family: str,
        canonical_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.canonical_condition.for_family(family, canonical_states)

    @torch.no_grad()
    def initialize_special_token_embeddings(
        self,
        base_embedding_weight: torch.Tensor,
    ) -> Dict[str, Any]:
        """Copy the eight original Qwen rows into trainable adapter storage."""

        rows = select_special_embedding_rows(base_embedding_weight)
        return self.initialize_special_token_embedding_rows(rows)

    @torch.no_grad()
    def initialize_special_token_embedding_rows(
        self,
        base_rows: torch.Tensor,
    ) -> Dict[str, Any]:
        """Initialize from an already sliced set of eight Qwen rows."""

        if tuple(base_rows.shape) != (8, 2048):
            raise ValueError("Special-token base rows must have shape [8,2048].")
        rows = base_rows.detach().to(
            device=self.special_token_embeddings.weight.device,
            dtype=self.special_token_embeddings.weight.dtype,
        )
        self.special_token_embeddings.weight.copy_(rows)
        max_abs = float(
            (self.special_token_embeddings.weight.detach() - rows).abs().max()
        )
        self._special_token_embeddings_initialized = max_abs == 0.0
        if not self._special_token_embeddings_initialized:
            raise RuntimeError("Special-token embedding initialization was not exact.")
        return {
            "schema_version": 1,
            "passed": True,
            "token_ids": list(SPECIAL_EMBEDDING_TOKEN_IDS),
            "shape": list(self.special_token_embeddings.weight.shape),
            "max_abs_error": max_abs,
            "base_embedding_modified": False,
        }

    @property
    def special_token_embeddings_initialized(self) -> bool:
        return bool(self._special_token_embeddings_initialized)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def named_parameter_groups(self) -> Dict[str, Tuple[str, ...]]:
        """Return disjoint groups matching the two-phase V2 optimizer recipe."""

        inherited: List[str] = []
        new: List[str] = []
        connector = [
            f"connector.{name}" for name, _ in self.connector.named_parameters()
        ]
        # Current-B's historical ``t2ss`` adapter intentionally stopped at
        # Qwen layer 26 (its legacy registry omitted the final language
        # layer).  Unified All-Task has an explicit slot 27, but that slot is
        # freshly initialized with a zero LoRA-B.  Keep it in the ``new``
        # group so its B matrix can update during the bootstrap phase; if it
        # were classified as inherited/frozen, its A matrix would have an
        # exactly-zero gradient forever and the slot would be dead.
        for slot, module in enumerate(self.qwen):
            target = new if int(slot) == 27 else inherited
            target.extend(
                f"qwen.{slot}.{name}" for name, _ in module.named_parameters()
            )
        inherited.extend(
            f"vision.{name}" for name, _ in self.vision.named_parameters()
        )
        for depth, projection in enumerate(
            self.canonical_condition.depth_projections
        ):
            target = inherited if depth < 28 else new
            target.extend(
                f"canonical_condition.depth_projections.{depth}.{name}"
                for name, _ in projection.named_parameters()
            )
        inherited.extend(
            f"canonical_condition.ss_width_bridge.{name}"
            for name, _ in self.canonical_condition.ss_width_bridge.named_parameters()
        )
        for family in ("shape", "material"):
            reader = getattr(self.canonical_condition, f"{family}_width_bridge")
            new.extend(
                f"canonical_condition.{family}_width_bridge.{name}"
                for name, _ in reader.named_parameters()
            )
        new.extend(
            f"special_token_embeddings.{name}"
            for name, _ in self.special_token_embeddings.named_parameters()
        )
        for family, bank in self.families.items():
            target = inherited if family == "ss" else new
            for name, _ in bank.named_parameters():
                full = f"families.{family}.{name}"
                target.append(full)
        for family, bank in self.conditions.items():
            target = inherited if family == "ss" else new
            for name, _ in bank.named_parameters():
                full = f"conditions.{family}.{name}"
                target.append(full)
        groups = {
            "inherited": tuple(inherited),
            "new": tuple(new),
            "connector": tuple(connector),
        }
        flattened = tuple(name for names in groups.values() for name in names)
        if len(flattened) != len(set(flattened)):
            raise RuntimeError("V2 optimizer parameter groups overlap.")
        if set(flattened) != set(dict(self.named_parameters())):
            raise RuntimeError("V2 optimizer parameter groups do not cover the registry.")
        return groups

    def iter_group_parameters(self, group: str) -> Iterable[nn.Parameter]:
        names = self.named_parameter_groups().get(str(group))
        if names is None:
            raise KeyError(group)
        parameters = dict(self.named_parameters())
        return (parameters[name] for name in names)

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu() for name, value in self.state_dict().items()
        }

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        result = self.load_state_dict(dict(state), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"All-task adapter load mismatch: {result}.")
        self._special_token_embeddings_initialized = True

    def schema_inventory(self) -> Dict[str, Any]:
        state = self.state_dict()
        return {
            "schema_version": 2,
            "architecture": ALL_TASK_SCHEMA,
            "variant": ALL_TASK_VARIANT,
            "task_scope": self.task_scope,
            "active_families": list(ALL_TASK_FAMILIES),
            "condition_owner": "slot_independent_canonical_1024_family_native_kv",
            "condition_depth_count": 30,
            "condition_view_count_generation": 3,
            "special_token_embedding_shape": [8, 2048],
            "special_token_embedding_token_ids": list(
                SPECIAL_EMBEDDING_TOKEN_IDS
            ),
            "special_token_embeddings_initialized": bool(
                self._special_token_embeddings_initialized
            ),
            "adapter_numel": sum(value.numel() for value in state.values()),
            "adapter_key_count": len(state),
            "adapter_keys": sorted(state),
            "branch_schema": self.branch_schema,
            "dual_lora": bool(self.dual_lora),
            "routed_lora": bool(self.routed_lora),
            "generation_only": bool(self.generation_only),
            "branch_owner": "generation_only",
            "gen_rank": int(self.gen_rank),
            "inherited_freeze_steps": ALL_TASK_INHERITED_FREEZE_STEPS,
            "route_semantics": (
                "selected_branch_only_no_detach_no_sum"
                if self.routed_lora
                else ("opposite_branch_detached" if self.dual_lora else "single_shared")
            ),
            "forbidden_owners": [
                "condition_carrier",
                "openeva_language_lora",
                "raw_detail_adapter",
                "detail_gate",
            ],
        }


__all__ = [
    "ALL_TASK_FAMILIES",
    "ALL_TASK_FROZEN_NUMEL",
    "ALL_TASK_INHERITED_FREEZE_STEPS",
    "ALL_TASK_SCHEMA",
    "ALL_TASK_TOTAL_NUMEL",
    "ALL_TASK_TRAINABLE_KEYS",
    "ALL_TASK_TRAINABLE_NUMEL",
    "ALL_TASK_VARIANT",
    "GENERATION_BRANCH_SCHEMA",
    "AllTaskAdapterRegistry",
    "FamilyConditionSlotAdapters",
    "OpenEVAConnector",
    "ResidualHeadPadBridge",
    "SlotCanonicalConditionAdapter",
    "SparseFamilySlotAdapters",
    "carrier_rms",
]
