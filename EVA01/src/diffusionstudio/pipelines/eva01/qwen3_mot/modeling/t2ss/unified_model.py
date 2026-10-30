"""Shared full-forward EVA01 Unified MoT executor for SS, Shape, and Material."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....qwen3_vl.modeling_qwen3_vl import rotate_half
from ...trellis2.modules import sparse as sp
from ...trellis2.modules.utils import manual_cast
from .adapters import (
    CARRIER_HEAD_DIM,
    CARRIER_HEADS,
    EVA01AdapterRegistry,
    QwenSlotLoRA,
    linear_with_lora,
    sliced_linear_with_lora,
)
from .base_loader import (
    load_qwen_base,
    load_ss_xl_base,
)
from .detail_condition import (
    DETAIL_INTERVENTIONS,
    HaarDetailSource,
    VISUAL_DETAIL_TAPS,
    grouped_haar_2x2,
)
from .flex_only import DirectBlockMaskStats, FlexAttentionOnly, create_unified_block_mask
from .index_map import TokenAxis, UnifiedIndexMap, build_generation_index_map
from ..attention.rope import build_dense_raster_position_ids


from .runtime_types import (
    EXPECTED_FROZEN_PARAMETERS,
    RAW_DETAIL_PROJECTION,
    FamilyRuntimeState as _FamilyRuntimeState,
    PackedGenerationBatch,
    TASK_FAMILIES,
    TASK_SLOT_COUNT,
    TASK_TARGET_FAMILY,
    UnifiedGenerationOutput,
    UnifiedMoTForwardTrace,
    UnifiedMoTSlotTrace,
    expand_qwen_grouped_kv as _expand_qwen_grouped_kv,
    expected_trainable_parameters,
    family_carrier_view as _family_carrier_view,
    family_features as _family_features,
    validate_sparse_alignment as _validate_sparse_alignment,
)
from .visual_detail_runtime import VisualDetailRuntimeMixin


def _apply_cached_interleaved_mrope_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        (query * query_cos) + (rotate_half(query) * query_sin),
        (key * key_cos) + (rotate_half(key) * key_sin),
    )


class EVA01UnifiedGenerationMoT(VisualDetailRuntimeMixin, nn.Module):
    """One shared-family executor; task selects activity and mask, never a parameter bank."""

    NUM_QWEN_LAYERS = 28
    UND_HIDDEN = 2048
    CARRIER_HEADS = CARRIER_HEADS
    CARRIER_HEAD_DIM = CARRIER_HEAD_DIM
    SS_TOKENS = 16**3

    def set_sampled_qk_probe(
        self,
        probe: Optional[Callable[..., None]],
    ) -> None:
        """Install an eval-only sampled-Q/K observer without changing the model graph.

        The observer receives already-RoPE'd unified Q/K immediately before the
        one formal FlexAttention call.  It must only inspect detached tensors;
        the production path never installs an observer and therefore pays no
        probe cost.
        """

        object.__setattr__(self, "_eva01_sampled_qk_probe", probe)

    def set_visual_detail_probe(self, probe: Optional[Callable[..., None]]) -> None:
        """Install an eval/probe-only observer for detached token-level gate values."""

        object.__setattr__(self, "_eva01_visual_detail_probe", probe)

    def __init__(
        self,
        *,
        qwen: nn.Module,
        ss_xl: nn.Module,
        variant: str,
        task_scope: str = "t2ss",
        shape_model: Optional[nn.Module] = None,
        material_model: Optional[nn.Module] = None,
        adapters: Optional[EVA01AdapterRegistry] = None,
        adapter_seed: int = 20260727,
        compile_flex_attention: bool = True,
        block_size: int = 128,
        visual_detail_mode: str = "none",
        visual_detail_tap: str = "block5",
        raw_detail_projection: str = RAW_DETAIL_PROJECTION,
        raw_detail_init_sha256: Optional[str] = None,
        detail_gate_hidden: int = 128,
        detail_gate_max: float = 0.25,
    ) -> None:
        super().__init__()
        normalized_variant = str(variant).strip().upper()
        normalized_scope = str(task_scope).strip().lower()
        if normalized_variant not in {"A", "B"}:
            raise ValueError("condition variant must be `A` or `B`.")
        if normalized_scope not in TASK_FAMILIES:
            raise ValueError(f"Unsupported task_scope `{task_scope}`.")
        self.variant = normalized_variant
        self.task_scope = normalized_scope
        self.active_families = TASK_FAMILIES[normalized_scope]
        self.qwen = qwen
        self.ss_xl = ss_xl
        self.shape_model = shape_model
        self.material_model = material_model
        self.adapters = adapters or EVA01AdapterRegistry(
            variant=self.variant,
            active_families=self.active_families,
            task_scope=self.task_scope,
            seed=int(adapter_seed),
            visual_detail_mode=visual_detail_mode,
            visual_detail_tap=visual_detail_tap,
            raw_detail_projection=raw_detail_projection,
            raw_detail_init_sha256=raw_detail_init_sha256,
            detail_gate_hidden=detail_gate_hidden,
            detail_gate_max=detail_gate_max,
        )
        if (
            self.adapters.variant != self.variant
            or self.adapters.task_scope != self.task_scope
            or tuple(self.adapters.active_families) != tuple(self.active_families)
        ):
            raise ValueError("Adapter registry does not match model variant/task scope.")
        if (
            self.adapters.visual_detail_mode != str(visual_detail_mode).strip().lower()
            or self.adapters.visual_detail_tap != str(visual_detail_tap).strip().lower()
        ):
            raise ValueError("Adapter registry does not match the requested visual-detail schema.")
        self.visual_detail_mode = self.adapters.visual_detail_mode
        self.visual_detail_tap = self.adapters.visual_detail_tap
        self.flex_attention = FlexAttentionOnly(
            head_dim=self.CARRIER_HEAD_DIM,
            compile_kernel=bool(compile_flex_attention),
        )
        self.block_size = int(block_size)
        self._validate_base_architecture()
        self.freeze_base_()
        self._bind_vision_lora()

    def _bind_vision_lora(self) -> None:
        """Attach non-owning LoRA references while the registry remains the sole parameter owner."""

        visual_blocks = self.qwen.model.visual.blocks
        if len(visual_blocks) != 24:
            raise ValueError("Qwen3VL vision base must expose exactly 24 blocks.")
        for block_index, block in enumerate(visual_blocks):
            attention = block.attn
            if self.variant == "B" and len(getattr(self.adapters, "vision", ())) == 24:
                slot = self.adapters.vision_slot(block_index)
                object.__setattr__(
                    attention,
                    "_eva01_qkv_lora_refs",
                    (weakref.ref(slot.q), weakref.ref(slot.k), weakref.ref(slot.v)),
                )
                object.__setattr__(attention, "_eva01_o_lora_ref", weakref.ref(slot.o))
            else:
                object.__setattr__(attention, "_eva01_qkv_lora_refs", None)
                object.__setattr__(attention, "_eva01_o_lora_ref", None)

    @classmethod
    def from_pretrained(
        cls,
        *,
        variant: str,
        device: torch.device,
        task_scope: str = "t2ss",
        qwen_root: Path,
        ss_xl_stem: Path,
        shape_stem: Optional[Path] = None,
        material_stem: Optional[Path] = None,
        adapter_seed: int = 20260727,
        compile_flex_attention: bool = True,
        block_size: int = 128,
        visual_detail_mode: str = "none",
        visual_detail_tap: str = "block5",
        raw_detail_projection: str = RAW_DETAIL_PROJECTION,
        raw_detail_init_sha256: Optional[str] = None,
        detail_gate_hidden: int = 128,
        detail_gate_max: float = 0.25,
    ) -> Tuple["EVA01UnifiedGenerationMoT", Dict[str, Dict[str, Any]]]:
        normalized_scope = str(task_scope).strip().lower()
        qwen, qwen_receipt = load_qwen_base(qwen_root, device=device)
        ss_xl, ss_receipt = load_ss_xl_base(ss_xl_stem, device=device)
        shape_model = None
        material_model = None
        load_receipts: Dict[str, Dict[str, Any]] = {
            "qwen": qwen_receipt.to_dict(),
            "ss_xl": ss_receipt.to_dict(),
        }
        if normalized_scope in {"t2shape", "t2sm"}:
            if shape_stem is None:
                raise ValueError("shape_stem is required for Shape generation.")

            shape_model, shape_receipt = load_slat_base(
                shape_stem,
                family="shape",
                device=device,
            )
            load_receipts["shape"] = shape_receipt.to_dict()
        if normalized_scope == "t2sm":
            if material_stem is None:
                raise ValueError("material_stem is required for Material generation.")

            material_model, material_receipt = load_slat_base(
                material_stem,
                family="material",
                device=device,
            )
            load_receipts["material"] = material_receipt.to_dict()
        adapters = EVA01AdapterRegistry(
            variant=variant,
            active_families=TASK_FAMILIES[normalized_scope],
            task_scope=normalized_scope,
            seed=adapter_seed,
            visual_detail_mode=visual_detail_mode,
            visual_detail_tap=visual_detail_tap,
            raw_detail_projection=raw_detail_projection,
            raw_detail_init_sha256=raw_detail_init_sha256,
            detail_gate_hidden=detail_gate_hidden,
            detail_gate_max=detail_gate_max,
        ).to(device=device)
        model = cls(
            qwen=qwen,
            ss_xl=ss_xl,
            shape_model=shape_model,
            material_model=material_model,
            variant=variant,
            task_scope=normalized_scope,
            adapters=adapters,
            adapter_seed=adapter_seed,
            compile_flex_attention=compile_flex_attention,
            block_size=block_size,
            visual_detail_mode=visual_detail_mode,
            visual_detail_tap=visual_detail_tap,
            raw_detail_projection=raw_detail_projection,
            raw_detail_init_sha256=raw_detail_init_sha256,
            detail_gate_hidden=detail_gate_hidden,
            detail_gate_max=detail_gate_max,
        )
        load_receipts["parameters"] = model.parameter_inventory()
        return model, load_receipts

    def _validate_base_architecture(self) -> None:
        language_model = self.qwen.model.language_model
        if len(language_model.layers) != self.NUM_QWEN_LAYERS:
            raise ValueError("Qwen base must expose exactly 28 text decoder layers.")
        text_config = self.qwen.config.text_config
        actual_qwen = (
            int(text_config.hidden_size),
            int(text_config.num_attention_heads),
            int(text_config.num_key_value_heads),
            int(text_config.head_dim),
        )
        if actual_qwen != (2048, 16, 8, 128):
            raise ValueError(f"Unexpected Qwen text ABI: {actual_qwen}.")
        if len(self.ss_xl.blocks) != 28:
            raise ValueError("TRELLIS1 SS XL must expose exactly 28 blocks.")
        actual_ss = (
            int(self.ss_xl.model_channels),
            int(self.ss_xl.num_heads),
            int(self.ss_xl.model_channels // self.ss_xl.num_heads),
            int(self.ss_xl.resolution),
            str(self.ss_xl.pe_mode),
        )
        if actual_ss != (1280, 16, 80, 16, "ape") or bool(self.ss_xl.share_mod):
            raise ValueError(f"Unexpected TRELLIS1 SS ABI: {actual_ss}.")
        for family_name, model in (
            ("shape", self.shape_model),
            ("material", self.material_model),
        ):
            if family_name not in self.active_families:
                if model is not None:
                    raise ValueError(f"Inactive family {family_name} must not instantiate a frozen base.")
                continue
            if model is None:
                raise ValueError(f"task_scope {self.task_scope} requires the {family_name} base.")
            expected_in = 32 if family_name == "shape" else 64
            actual = (
                int(model.model_channels),
                int(model.num_heads),
                int(model.model_channels // model.num_heads),
                int(model.num_blocks),
                int(model.in_channels),
                int(model.out_channels),
                int(model.cond_channels),
                str(model.pe_mode),
                bool(model.share_mod),
            )
            if actual != (1536, 12, 128, 30, expected_in, 32, 1024, "rope", True):
                raise ValueError(f"Unexpected {family_name} ABI: {actual}.")

    def freeze_base_(self) -> None:
        for base in (self.qwen, self.ss_xl, self.shape_model, self.material_model):
            if base is None:
                continue
            for parameter in base.parameters():
                parameter.requires_grad_(False)
        for parameter in self.adapters.parameters():
            parameter.requires_grad_(True)

    def parameter_inventory(self) -> Dict[str, Any]:
        frozen_by_family = {
            "qwen": sum(parameter.numel() for parameter in self.qwen.parameters()),
            "ss": sum(parameter.numel() for parameter in self.ss_xl.parameters()),
        }
        if self.shape_model is not None:
            frozen_by_family["shape"] = sum(
                parameter.numel() for parameter in self.shape_model.parameters()
            )
        if self.material_model is not None:
            frozen_by_family["material"] = sum(
                parameter.numel() for parameter in self.material_model.parameters()
            )
        for name, observed in frozen_by_family.items():
            expected = EXPECTED_FROZEN_PARAMETERS[name]
            if observed != expected:
                raise RuntimeError(f"Frozen {name} parameter count must be {expected}, got {observed}.")
        trainable = self.adapters.trainable_parameter_count
        expected_trainable = expected_trainable_parameters(
            self.task_scope,
            self.variant,
            self.visual_detail_mode,
            self.adapters.detail_gate_hidden,
        )
        if trainable != expected_trainable:
            raise RuntimeError(
                f"{self.task_scope} variant {self.variant} trainable count must be "
                f"{expected_trainable}, got {trainable}."
            )
        frozen_total = sum(frozen_by_family.values())
        named = dict(self.adapters.named_parameters())
        return {
            "task_scope": self.task_scope,
            "variant": self.variant,
            "visual_detail_mode": self.visual_detail_mode,
            "visual_detail_tap": self.visual_detail_tap,
            "frozen_by_family": frozen_by_family,
            "frozen_total": frozen_total,
            "new_total": trainable,
            "trainable_total": trainable,
            "model_total": frozen_total + trainable,
            "parameter_groups": {
                name: sum(named[parameter_name].numel() for parameter_name in names)
                for name, names in self.adapters.named_parameter_groups().items()
            },
            "adapter_schema": self.adapters.schema_inventory(),
        }

    @staticmethod
    def _apply_interleaved_mrope(
        *,
        rotary_embedding: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_cos, query_sin = rotary_embedding(query, query_position_ids)
        key_cos, key_sin = rotary_embedding(key, key_position_ids)
        query_cos = query_cos.unsqueeze(1)
        query_sin = query_sin.unsqueeze(1)
        key_cos = key_cos.unsqueeze(1)
        key_sin = key_sin.unsqueeze(1)
        return (
            (query * query_cos) + (rotate_half(query) * query_sin),
            (key * key_cos) + (rotate_half(key) * key_sin),
        )

    @staticmethod
    def _interleaved_mrope_factors(
        *,
        rotary_embedding: nn.Module,
        reference: torch.Tensor,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_cos, query_sin = rotary_embedding(
            reference,
            query_position_ids,
        )
        key_cos, key_sin = rotary_embedding(
            reference,
            key_position_ids,
        )
        return (
            query_cos.unsqueeze(1),
            query_sin.unsqueeze(1),
            key_cos.unsqueeze(1),
            key_sin.unsqueeze(1),
        )

    @staticmethod
    def _apply_cached_interleaved_mrope(
        query: torch.Tensor,
        key: torch.Tensor,
        factors: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _apply_cached_interleaved_mrope_impl(
            query,
            key,
            *factors,
        )

    def _project_qwen_qkv(
        self,
        *,
        layer: nn.Module,
        adapters: QwenSlotLoRA,
        normalized_hidden: torch.Tensor,
        include_query: bool = True,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        attention = layer.self_attn
        batch_size, token_count, _ = normalized_hidden.shape
        query = None
        if include_query:
            query = linear_with_lora(normalized_hidden, attention.q_proj, adapters.q)
            query = attention.q_norm(query.view(batch_size, token_count, 16, 128))
        key = linear_with_lora(normalized_hidden, attention.k_proj, adapters.k)
        value = linear_with_lora(normalized_hidden, attention.v_proj, adapters.v)
        key = attention.k_norm(key.view(batch_size, token_count, 8, 128))
        value = value.view(batch_size, token_count, 8, 128)
        return query, _expand_qwen_grouped_kv(key, groups=2), _expand_qwen_grouped_kv(
            value,
            groups=2,
        )

    def _prepare_ss_state(
        self,
        *,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        batch: PackedGenerationBatch,
        role: str,
    ) -> _FamilyRuntimeState:
        expected = (batch.logical_sample_count, 8, 16, 16, 16)
        if tuple(latent.shape) != expected:
            raise ValueError(f"SS latent must have shape {expected}, got {tuple(latent.shape)}.")
        hidden = latent.view(latent.shape[0], latent.shape[1], -1).permute(0, 2, 1).contiguous()
        hidden = self.ss_xl.input_layer(hidden)
        hidden = hidden + self.ss_xl.pos_emb[None]
        timestep_embedding = self.ss_xl.t_embedder(timesteps.to(dtype=torch.float32) * 1000.0)
        hidden = manual_cast(hidden, self.ss_xl.dtype)
        timestep_embedding = manual_cast(timestep_embedding, self.ss_xl.dtype)
        sample_ids = torch.arange(
            batch.logical_sample_count,
            device=latent.device,
            dtype=torch.long,
        ).repeat_interleave(self.SS_TOKENS).view(1, -1)
        local = torch.arange(self.SS_TOKENS, device=latent.device, dtype=torch.long).repeat(
            batch.logical_sample_count
        ).view(1, -1)
        position_rows = []
        for sample_index in range(batch.logical_sample_count):
            text_positions = batch.und_position_ids[
                :, 0, batch.und_sample_ids[0].eq(sample_index)
            ]
            offset = int(text_positions.max().item()) + 1
            position_rows.append(
                build_dense_raster_position_ids((16, 16, 16), offset, device=latent.device)
            )
        positions = torch.cat(position_rows, dim=1).unsqueeze(1)
        return _FamilyRuntimeState(
            name="ss",
            base=self.ss_xl,
            hidden=hidden,
            timestep_embedding=timestep_embedding,
            sample_ids=sample_ids,
            local_indices=local,
            position_ids=positions,
            role=role,
        )

    def _prepare_sparse_state(
        self,
        *,
        family_name: str,
        latent: sp.SparseTensor,
        timesteps: torch.Tensor,
        batch: PackedGenerationBatch,
        role: str,
        concat_cond: Optional[sp.SparseTensor] = None,
    ) -> _FamilyRuntimeState:
        if not isinstance(latent, sp.SparseTensor):
            raise TypeError(f"{family_name} must be a native sp.SparseTensor.")
        if len(latent.layout) != batch.logical_sample_count:
            raise ValueError(f"{family_name} sparse layout does not contain P logical samples.")
        base = self.shape_model if family_name == "shape" else self.material_model
        if base is None:
            raise RuntimeError(f"The {family_name} frozen base is not loaded.")
        input_value = latent
        if concat_cond is not None:
            _validate_sparse_alignment(
                latent,
                concat_cond,
                noisy_name="noisy material",
                clean_name="clean shape concat",
            )
            input_value = sp.sparse_cat([latent, concat_cond], dim=-1)
        hidden = base.input_layer(input_value)
        hidden = manual_cast(hidden, base.dtype)
        timestep_embedding = base.t_embedder(timesteps.to(dtype=torch.float32) * 1000.0)
        if base.share_mod:
            timestep_embedding = base.adaLN_modulation(timestep_embedding)
        timestep_embedding = manual_cast(timestep_embedding, base.dtype)
        if base.pe_mode == "ape":
            hidden = hidden + manual_cast(base.pos_embedder(hidden.coords[:, 1:]), base.dtype)
        coords = hidden.coords
        sample_ids = coords[:, 0].to(dtype=torch.long).view(1, -1)
        local_rows = []
        position_rows = []
        for sample_index, layout in enumerate(hidden.layout):
            token_count = int(layout.stop - layout.start)
            local_rows.append(torch.arange(token_count, device=coords.device, dtype=torch.long))
            text_positions = batch.und_position_ids[
                :, 0, batch.und_sample_ids[0].eq(sample_index)
            ]
            offset = int(text_positions.max().item()) + 1
            xyz = coords[layout, 1:4].to(dtype=torch.long).T
            position_rows.append(xyz + offset)
        return _FamilyRuntimeState(
            name=family_name,
            base=base,
            hidden=hidden,
            timestep_embedding=timestep_embedding,
            sample_ids=sample_ids,
            local_indices=torch.cat(local_rows).view(1, -1),
            position_ids=torch.cat(position_rows, dim=1).unsqueeze(1),
            role=role,
        )

    @staticmethod
    def _native_modulation(
        block: nn.Module,
        timestep_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(block, "share_mod", False):
            return (block.modulation + timestep_embedding).type(timestep_embedding.dtype).chunk(6, dim=1)
        return block.adaLN_modulation(timestep_embedding).chunk(6, dim=1)

    def _family_native_self_pre_mot(
        self,
        state: _FamilyRuntimeState,
        *,
        block_index: int,
    ) -> None:
        block = state.base.blocks[int(block_index)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._native_modulation(
            block,
            state.timestep_embedding,
        )
        if isinstance(state.hidden, sp.SparseTensor):
            # Every all-task sparse chunk is one native sequence.  Applying
            # [1,C] modulation directly to [N,C] features is exactly the same
            # broadcast as SparseTensor.__elemwise__, but avoids constructing
            # a GPU batch-broadcast index and its synchronizing IndexBackward.
            branch = state.hidden.replace(
                block.norm1(state.hidden.feats) * (1 + scale_msa) + shift_msa
            )
            branch = block.self_attn(branch)
            state.hidden = state.hidden.replace(
                state.hidden.feats + branch.feats * gate_msa
            )
        else:
            branch = block.norm1(state.hidden)
            branch = branch * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            phases = getattr(state.base, "rope_phases", None)
            branch = block.self_attn(branch, phases=phases)
            state.hidden = state.hidden + branch * gate_msa.unsqueeze(1)
        state.mlp_modulation = (shift_mlp, scale_mlp, gate_mlp)

    def _family_mot_input(
        self,
        state: _FamilyRuntimeState,
        *,
        block_index: int,
    ) -> Union[torch.Tensor, sp.SparseTensor]:
        block = state.base.blocks[int(block_index)]
        if isinstance(state.hidden, sp.SparseTensor):
            return state.hidden.replace(block.norm2(state.hidden.feats))
        return block.norm2(state.hidden)

    def _project_family_qkv(
        self,
        state: _FamilyRuntimeState,
        *,
        block_index: int,
        mot_input: Union[torch.Tensor, sp.SparseTensor],
        include_query: bool = True,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        block = state.base.blocks[int(block_index)]
        adapters = self.adapters.family_slot(state.name, block_index)
        features = _family_features(mot_input)
        query = None
        if include_query:
            query = linear_with_lora(features, block.cross_attn.to_q, adapters.q)
            query = query.view(*query.shape[:-1], adapters.spec.num_heads, adapters.spec.head_dim)
            if getattr(block.cross_attn, "qk_rms_norm", False):
                query = block.cross_attn.q_rms_norm(query)
            query = adapters.q_bridge(_family_carrier_view(query))
        key = sliced_linear_with_lora(
            features,
            fused_projection=block.self_attn.to_qkv,
            slice_index=1,
            slice_count=3,
            delta=adapters.k,
        )
        value = sliced_linear_with_lora(
            features,
            fused_projection=block.self_attn.to_qkv,
            slice_index=2,
            slice_count=3,
            delta=adapters.v,
        )
        key = key.view(*key.shape[:-1], adapters.spec.num_heads, adapters.spec.head_dim)
        value = value.view(*value.shape[:-1], adapters.spec.num_heads, adapters.spec.head_dim)
        if getattr(block.self_attn, "qk_rms_norm", False):
            key = block.self_attn.k_rms_norm(key)
        return query, adapters.k_bridge(_family_carrier_view(key)), adapters.v_bridge(
            _family_carrier_view(value)
        )

    def _project_condition_view(
        self,
        *,
        family_name: str,
        block_index: int,
        canonical_condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_base = {
            "ss": self.ss_xl,
            "shape": self.shape_model,
            "material": self.material_model,
        }[family_name]
        if state_base is None:
            raise RuntimeError(f"Condition view requested for unloaded family {family_name}.")
        block = state_base.blocks[int(block_index)]
        adapters = self.adapters.condition_slot(family_name, block_index)
        context = self.adapters.family_condition_states(family_name, canonical_condition)
        key_value = linear_with_lora(context, block.cross_attn.to_kv, adapters.cross_kv)
        token_count = int(key_value.shape[1])
        key, value = key_value.view(
            1,
            token_count,
            2,
            adapters.spec.num_heads,
            adapters.spec.head_dim,
        ).unbind(dim=2)
        if getattr(block.cross_attn, "qk_rms_norm", False):
            key = block.cross_attn.k_rms_norm(key)
        return adapters.k_bridge(key), adapters.v_bridge(value)

    def _apply_family_attention_output(
        self,
        state: _FamilyRuntimeState,
        *,
        block_index: int,
        carrier_output: torch.Tensor,
        branch_mask: Optional[torch.Tensor] = None,
    ) -> None:
        block = state.base.blocks[int(block_index)]
        adapters = self.adapters.family_slot(state.name, block_index)
        native = adapters.o_bridge(carrier_output)
        if isinstance(state.hidden, sp.SparseTensor):
            native_features = native.squeeze(0).reshape(native.shape[1], adapters.spec.hidden_size)
            update = linear_with_lora(
                native_features,
                block.cross_attn.to_out,
                adapters.o,
                branch_mask=branch_mask,
            )
            state.hidden = state.hidden.replace(state.hidden.feats + update)
            if state.mlp_modulation is None:
                raise RuntimeError("Sparse family MLP modulation is missing.")
            shift_mlp, scale_mlp, gate_mlp = state.mlp_modulation
            branch = state.hidden.replace(
                block.norm3(state.hidden.feats) * (1 + scale_mlp) + shift_mlp
            )
            branch = block.mlp(branch)
            state.hidden = state.hidden.replace(
                state.hidden.feats + branch.feats * gate_mlp
            )
        else:
            native_features = native.reshape(
                state.hidden.shape[0],
                state.hidden.shape[1],
                adapters.spec.hidden_size,
            )
            update = linear_with_lora(
                native_features,
                block.cross_attn.to_out,
                adapters.o,
                branch_mask=branch_mask,
            )
            state.hidden = state.hidden + update
            if state.mlp_modulation is None:
                raise RuntimeError("Dense family MLP modulation is missing.")
            shift_mlp, scale_mlp, gate_mlp = state.mlp_modulation
            branch = block.norm3(state.hidden)
            branch = branch * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            branch = block.mlp(branch)
            state.hidden = state.hidden + branch * gate_mlp.unsqueeze(1)

    @staticmethod
    def _axis_from_und(batch: PackedGenerationBatch) -> TokenAxis:
        return TokenAxis(
            "und",
            batch.und_sample_ids,
            batch.und_position_ids[0],
            batch.und_position_ids,
        )

    @staticmethod
    def _condition_axis(family_name: str, batch: PackedGenerationBatch) -> TokenAxis:
        return TokenAxis(
            f"condition_{family_name}",
            batch.und_sample_ids,
            batch.und_position_ids[0],
            batch.und_position_ids,
        )

    def _build_slot_index_map(
        self,
        *,
        task: str,
        batch: PackedGenerationBatch,
        states: Mapping[str, _FamilyRuntimeState],
        query_families: Sequence[str],
    ) -> UnifiedIndexMap:
        query_axes: List[TokenAxis] = []
        if "und" in query_families:
            query_axes.append(self._axis_from_und(batch))
        query_axes.extend(states[name].axis() for name in query_families if name != "und")
        key_axes: List[TokenAxis] = []
        if self.variant == "A" or "und" in query_families:
            key_axes.append(self._axis_from_und(batch))
        if self.variant == "B":
            key_axes.extend(
                self._condition_axis(name, batch)
                for name in query_families
                if name in TASK_FAMILIES[task]
            )
        key_axes.extend(states[name].axis() for name in TASK_FAMILIES[task])
        max_kv = int(batch.max_unified_tokens)
        if self.variant == "B":
            max_kv += len(TASK_FAMILIES[task]) * int(batch.max_condition_tokens)
        return build_generation_index_map(
            task=task,
            variant=self.variant,
            query_axes=query_axes,
            key_value_axes=key_axes,
            max_query_tokens=int(batch.max_unified_tokens),
            max_key_value_tokens=max_kv,
        )

    @staticmethod
    def _pad_carrier(
        values: Sequence[torch.Tensor],
        *,
        target_tokens: int,
    ) -> torch.Tensor:
        if not values:
            raise ValueError("At least one carrier segment is required.")
        active = torch.cat(tuple(values), dim=1)
        if active.shape[0] != 1 or active.shape[2:] != (CARRIER_HEADS, CARRIER_HEAD_DIM):
            raise ValueError(f"Invalid unified carrier segment shape {tuple(active.shape)}.")
        if active.shape[1] > int(target_tokens):
            raise ValueError(
                f"Active carrier tokens {active.shape[1]} exceed fixed target {target_tokens}."
            )
        if active.shape[1] == int(target_tokens):
            return active
        padding = torch.zeros(
            (1, int(target_tokens) - active.shape[1], CARRIER_HEADS, CARRIER_HEAD_DIM),
            device=active.device,
            dtype=active.dtype,
        )
        return torch.cat((active, padding), dim=1)

    def _run_slot(
        self,
        *,
        task: str,
        slot_index: int,
        batch: PackedGenerationBatch,
        und_hidden: torch.Tensor,
        states: Dict[str, _FamilyRuntimeState],
        index_map: UnifiedIndexMap,
        block_mask: Any,
        raw_detail_source: Optional[HaarDetailSource],
        raw_detail_states: Optional[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, UnifiedMoTSlotTrace, Optional[torch.Tensor]]:
        q_by_name: Dict[str, torch.Tensor] = {}
        k_by_name: Dict[str, torch.Tensor] = {}
        v_by_name: Dict[str, torch.Tensor] = {}
        exhausted: List[str] = []
        query_families = tuple(name for name in index_map.query_slices if name != "padding")

        if "und" in query_families:
            layer = self.qwen.model.language_model.layers[slot_index]
            und_residual = und_hidden
            normalized = layer.input_layernorm(und_hidden)
            q_und, k_und, v_und = self._project_qwen_qkv(
                layer=layer,
                adapters=self.adapters.qwen_slot(slot_index),
                normalized_hidden=normalized,
            )
            if q_und is None:
                raise RuntimeError("Progressing UND slot did not create Q.")
            q_by_name["und"] = q_und
            k_by_name["und"] = k_und
            v_by_name["und"] = v_und
        else:
            exhausted.append("und")
            if self.variant == "A":
                layer = self.qwen.model.language_model.layers[-1]
                normalized = layer.input_layernorm(und_hidden)
                _q_und, k_und, v_und = self._project_qwen_qkv(
                    layer=layer,
                    adapters=self.adapters.qwen_slot(self.NUM_QWEN_LAYERS - 1),
                    normalized_hidden=normalized,
                    include_query=False,
                )
                k_by_name["und"] = k_und
                v_by_name["und"] = v_und

        for family_name in TASK_FAMILIES[task]:
            state = states[family_name]
            progressing = family_name in query_families
            if progressing:
                self._family_native_self_pre_mot(state, block_index=slot_index)
                mot_input = self._family_mot_input(state, block_index=slot_index)
                q_family, k_family, v_family = self._project_family_qkv(
                    state,
                    block_index=slot_index,
                    mot_input=mot_input,
                    include_query=True,
                )
                if q_family is None:
                    raise RuntimeError(f"Progressing family {family_name} did not create Q.")
                q_by_name[family_name] = q_family
                k_by_name[family_name] = k_family
                v_by_name[family_name] = v_family
            else:
                exhausted.append(family_name)
                tail_block = len(state.base.blocks) - 1
                mot_input = self._family_mot_input(state, block_index=tail_block)
                _unused_q, k_family, v_family = self._project_family_qkv(
                    state,
                    block_index=tail_block,
                    mot_input=mot_input,
                    include_query=False,
                )
                k_by_name[family_name] = k_family
                v_by_name[family_name] = v_family

        detail_summary = None
        if self.variant == "B":
            condition_hidden = und_hidden
            if raw_detail_source is not None or raw_detail_states is not None:
                if raw_detail_source is None or raw_detail_states is None:
                    raise RuntimeError("Detail-B raw source and adapted states must be present together.")
                condition_hidden, detail_summary = self._detail_condition_input(
                    und_hidden,
                    packed_batch=batch,
                    raw_source=raw_detail_source,
                    raw_detail=raw_detail_states,
                    timesteps=timesteps,
                    slot_index=slot_index,
                )
            canonical_condition = self.adapters.canonical_condition_states(
                slot_index,
                condition_hidden,
            )
            for family_name in query_families:
                if family_name == "und":
                    continue
                condition_block = slot_index
                k_condition, v_condition = self._project_condition_view(
                    family_name=family_name,
                    block_index=condition_block,
                    canonical_condition=canonical_condition,
                )
                k_by_name[f"condition_{family_name}"] = k_condition
                v_by_name[f"condition_{family_name}"] = v_condition

        query_values = [q_by_name[name] for name in index_map.query_slices if name != "padding"]
        key_values = [k_by_name[name] for name in index_map.key_value_slices if name != "padding"]
        value_values = [v_by_name[name] for name in index_map.key_value_slices if name != "padding"]
        query = self._pad_carrier(query_values, target_tokens=index_map.query_length)
        key = self._pad_carrier(key_values, target_tokens=index_map.key_value_length)
        value = self._pad_carrier(value_values, target_tokens=index_map.key_value_length)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        query, key = self._apply_interleaved_mrope(
            rotary_embedding=self.qwen.model.language_model.rotary_emb,
            query=query,
            key=key,
            query_position_ids=index_map.query.position_ids,
            key_position_ids=index_map.key_value.position_ids,
        )
        sampled_qk_probe = getattr(self, "_eva01_sampled_qk_probe", None)
        if sampled_qk_probe is not None:
            sampled_qk_probe(
                slot_index=int(slot_index),
                task=str(task),
                query=query.detach(),
                key=key.detach(),
                index_map=index_map,
                scale=float(self.flex_attention.scale),
            )
        attention_output = self.flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
        ).transpose(1, 2).contiguous()

        if "und" in query_families:
            start, end = index_map.query_slices["und"]
            carrier = attention_output[:, start:end].reshape(1, end - start, self.UND_HIDDEN)
            update = linear_with_lora(
                carrier,
                layer.self_attn.o_proj,
                self.adapters.qwen_slot(slot_index).o,
            )
            und_hidden = und_residual + update
            und_hidden = und_hidden + layer.mlp(layer.post_attention_layernorm(und_hidden))

        for family_name in TASK_FAMILIES[task]:
            if family_name not in query_families:
                continue
            start, end = index_map.query_slices[family_name]
            self._apply_family_attention_output(
                states[family_name],
                block_index=slot_index,
                carrier_output=attention_output[:, start:end],
            )

        mask_stats = getattr(block_mask, "eva01_stats", None)
        stats_payload = asdict(mask_stats) if isinstance(mask_stats, DirectBlockMaskStats) else {}
        trace = UnifiedMoTSlotTrace(
            slot_index=int(slot_index),
            backend=self.flex_attention.backend_name,
            flex_call_ordinal=int(self.flex_attention.call_count),
            query_length=index_map.query_length,
            key_value_length=index_map.key_value_length,
            query_families=tuple(name for name in index_map.query_slices if name != "padding"),
            key_value_families=tuple(
                name for name in index_map.key_value_slices if name != "padding"
            ),
            family_query_tokens={
                name: end - start
                for name, (start, end) in index_map.query_slices.items()
                if name != "padding"
            },
            family_key_value_tokens={
                name: end - start
                for name, (start, end) in index_map.key_value_slices.items()
                if name != "padding"
            },
            exhausted_read_only_families=tuple(exhausted),
            block_mask=stats_payload,
        )
        return und_hidden, trace, detail_summary

    def _finalize_family(
        self,
        state: _FamilyRuntimeState,
        *,
        reference: Union[torch.Tensor, sp.SparseTensor],
    ) -> Union[torch.Tensor, sp.SparseTensor]:
        if isinstance(state.hidden, sp.SparseTensor):
            hidden = state.hidden.replace(F.layer_norm(state.hidden.feats, state.hidden.feats.shape[-1:]))
            hidden = manual_cast(hidden, reference.dtype)
            return state.base.out_layer(hidden)
        hidden = manual_cast(state.hidden, reference.dtype)
        hidden = F.layer_norm(hidden, hidden.shape[-1:])
        velocity = state.base.out_layer(hidden)
        return velocity.permute(0, 2, 1).view(
            reference.shape[0],
            8,
            16,
            16,
            16,
        ).contiguous()

    def _prepare_und_embeddings(
        self,
        packed_batch: PackedGenerationBatch,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Optional[HaarDetailSource]]:
        hidden = self.qwen.model.language_model.embed_tokens(packed_batch.input_ids)
        if packed_batch.pixel_values is None:
            return hidden, (), None
        if packed_batch.image_grid_thw is None or packed_batch.visual_pos_mask is None:
            raise ValueError("Packed image metadata is incomplete.")
        pixel_values = packed_batch.pixel_values.to(dtype=self.qwen.model.visual.dtype)
        raw_source = None
        if self.visual_detail_mode == "haar_gated":
            tap_index = VISUAL_DETAIL_TAPS[self.visual_detail_tap]
            image_embeds, deepstack_embeds, premerge = self.qwen.model.visual(
                pixel_values,
                grid_thw=packed_batch.image_grid_thw,
                return_premerge_features=True,
                premerge_feature_indexes=(tap_index,),
            )
            tap_key = f"block{tap_index}"
            if tap_key not in premerge:
                raise RuntimeError(f"Qwen vision did not return the locked detail tap {tap_key}.")
            high, low_proxy, energy = grouped_haar_2x2(premerge[tap_key])
            merged_counts = packed_batch.image_grid_thw.prod(dim=-1) // 4
            if packed_batch.image_sample_ids is None:
                raise ValueError("Raw detail requires packed image sample IDs.")
            detail_sample_ids = torch.repeat_interleave(
                packed_batch.image_sample_ids.to(dtype=torch.long),
                merged_counts.to(dtype=torch.long),
            )
            raw_source = HaarDetailSource(
                high_frequency=high,
                low_frequency_proxy=low_proxy,
                sample_ids=detail_sample_ids,
                energy=energy,
            )
            raw_source.validate()
        else:
            image_embeds, deepstack_embeds = self.qwen.model.visual(
                pixel_values,
                grid_thw=packed_batch.image_grid_thw,
            )
        visual_mask = packed_batch.visual_pos_mask.to(
            device=hidden.device,
            dtype=torch.bool,
        )
        visual_indices = packed_batch.visual_token_indices
        if visual_indices is not None:
            visual_indices = visual_indices.to(
                device=hidden.device,
                dtype=torch.long,
            )
            visual_count = int(visual_indices.numel())
        else:
            visual_count = int(visual_mask.sum().detach().cpu())
        if tuple(image_embeds.shape) != (visual_count, self.UND_HIDDEN):
            raise RuntimeError(
                "Final dense image embeddings do not match packed visual placeholders: "
                f"embeds={tuple(image_embeds.shape)}, placeholders={visual_count}."
            )
        if len(deepstack_embeds) != 3 or any(
            tuple(value.shape) != (visual_count, self.UND_HIDDEN)
            for value in deepstack_embeds
        ):
            raise RuntimeError("Qwen3VL DeepStack must return three dense 2048-wide feature sets.")
        image_embeds = image_embeds.to(device=hidden.device, dtype=hidden.dtype)
        if visual_indices is not None:
            hidden = hidden.index_copy(
                1,
                visual_indices,
                image_embeds.unsqueeze(0),
            )
        else:
            hidden = hidden.clone()
            hidden[visual_mask] = image_embeds
        if raw_source is not None:
            observed_sample_ids = (
                packed_batch.und_sample_ids[0]
                .index_select(0, visual_indices)
                .to(dtype=torch.long)
                if visual_indices is not None
                else packed_batch.und_sample_ids[visual_mask].to(dtype=torch.long)
            )
            if raw_source.high_frequency.shape[0] != visual_count or not torch.equal(
                raw_source.sample_ids,
                observed_sample_ids,
            ):
                raise RuntimeError("Raw detail tokens do not align with packed visual placeholders.")
        return hidden, tuple(deepstack_embeds), raw_source

    def build_multimodal_position_ids(
        self,
        token_rows: Sequence[torch.Tensor],
        image_grid_rows: Sequence[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Use the original Qwen3VL MRoPE builder independently for every logical sample."""

        if not token_rows or len(token_rows) != len(image_grid_rows):
            raise ValueError("Token rows and image-grid rows must be nonempty and aligned.")
        positions = []
        for token_row, image_grid in zip(token_rows, image_grid_rows):
            if token_row.ndim != 1 or token_row.numel() <= 0:
                raise ValueError("Each logical Qwen token row must be nonempty and one-dimensional.")
            row = token_row.to(dtype=torch.long)
            attention_mask = torch.ones_like(row).view(1, -1)
            grid = None
            if image_grid is not None:
                grid = image_grid.to(device=row.device, dtype=torch.long).view(-1, 3)
            row_positions, _rope_delta = self.qwen.model.get_rope_index(
                row.view(1, -1),
                image_grid_thw=grid,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            if row_positions.shape != (3, 1, row.numel()):
                raise RuntimeError("Original Qwen3VL returned an unexpected MRoPE position shape.")
            positions.append(row_positions)
        return torch.cat(positions, dim=-1)

    @staticmethod
    def _inject_deepstack(
        hidden: torch.Tensor,
        visual_pos_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
        *,
        visual_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = torch.zeros_like(hidden)
        visual_embeds = visual_embeds.to(device=hidden.device, dtype=hidden.dtype)
        if visual_token_indices is not None:
            delta = delta.index_copy(
                1,
                visual_token_indices.to(device=hidden.device, dtype=torch.long),
                visual_embeds.unsqueeze(0),
            )
        else:
            visual_mask = visual_pos_mask.to(device=hidden.device, dtype=torch.bool)
            delta[visual_mask] = visual_embeds
        return hidden + delta

    def forward_generation(
        self,
        task: str,
        packed_batch: PackedGenerationBatch,
        timesteps: torch.Tensor,
        *,
        return_trace: bool = False,
        visual_detail_intervention: str = "correct",
        detail_intervention_sample_ids: Optional[Tuple[int, ...]] = None,
        **forbidden_kwargs: Any,
    ) -> UnifiedGenerationOutput:
        normalized_task = str(task).strip().lower()
        if forbidden_kwargs:
            forbidden = sorted(
                name
                for name in forbidden_kwargs
                if name in {"use_cache", "past_key_values", "und_memory_by_layer", "layer_memory"}
            )
            if forbidden:
                raise ValueError(f"EVA01 Unified Generation forbids runtime memory arguments: {forbidden}.")
            raise TypeError(f"Unexpected forward_generation arguments: {sorted(forbidden_kwargs)}.")
        if normalized_task not in TASK_FAMILIES:
            raise ValueError(f"Unsupported generation task `{task}`.")
        detail_intervention = str(visual_detail_intervention).strip().lower()
        if detail_intervention not in DETAIL_INTERVENTIONS:
            raise ValueError(f"Unknown visual-detail intervention `{visual_detail_intervention}`.")
        selected_detail_samples = (
            None
            if detail_intervention_sample_ids is None
            else tuple(dict.fromkeys(int(value) for value in detail_intervention_sample_ids))
        )
        if self.training and (detail_intervention != "correct" or selected_detail_samples):
            raise RuntimeError("Training forbids eval-only raw-detail interventions.")
        if self.visual_detail_mode == "none" and (
            detail_intervention != "correct" or selected_detail_samples
        ):
            raise RuntimeError("Current-B has no raw-detail path to intervene on.")
        requested_families = TASK_FAMILIES[normalized_task]
        if any(name not in self.active_families for name in requested_families):
            raise ValueError(
                f"Model task_scope {self.task_scope} does not instantiate {normalized_task} families."
            )
        packed_batch.validate(task=normalized_task, block_size=self.block_size)
        if timesteps.shape != (packed_batch.logical_sample_count,):
            raise ValueError("timesteps must contain one value per logical sample.")
        device = packed_batch.input_ids.device
        for tensor in (
            packed_batch.und_sample_ids,
            packed_batch.und_position_ids,
            timesteps,
        ):
            if tensor.device != device:
                raise ValueError("Packed metadata, inputs, and timesteps must share one device.")
        for tensor in (
            packed_batch.pixel_values,
            packed_batch.image_grid_thw,
            packed_batch.image_sample_ids,
            packed_batch.visual_pos_mask,
            packed_batch.und_token_kinds,
            packed_batch.condition_drop_mask,
        ):
            if tensor is not None and tensor.device != device:
                raise ValueError("Packed image metadata and inputs must share the model device.")

        states: Dict[str, _FamilyRuntimeState] = {}
        target_family = TASK_TARGET_FAMILY[normalized_task]
        for family_name in requested_families:
            role = "noisy" if family_name == target_family else "clean"
            family_time = timesteps if role == "noisy" else torch.zeros_like(timesteps)
            value = packed_batch.family_inputs[family_name]
            if family_name == "ss":
                states[family_name] = self._prepare_ss_state(
                    latent=value,
                    timesteps=family_time,
                    batch=packed_batch,
                    role=role,
                )
            else:
                concat = None
                if family_name == "material":
                    concat = packed_batch.family_inputs.get("material_concat_shape")
                    if concat is None:
                        raise ValueError("t2sm requires material_concat_shape SparseTensor.")
                    _validate_sparse_alignment(
                        value,
                        packed_batch.family_inputs["shape"],
                        noisy_name="noisy material",
                        clean_name="clean shape lane",
                    )
                states[family_name] = self._prepare_sparse_state(
                    family_name=family_name,
                    latent=value,
                    timesteps=family_time,
                    batch=packed_batch,
                    role=role,
                    concat_cond=concat,
                )

        active_query_tokens = int(packed_batch.input_ids.shape[1]) + sum(
            state.token_count for state in states.values()
        )
        if active_query_tokens > int(packed_batch.max_unified_tokens):
            raise ValueError(
                f"Packed item uses {active_query_tokens} unified tokens, exceeding "
                f"{packed_batch.max_unified_tokens}."
            )
        und_hidden, all_deepstack_visual_embeds, raw_detail_source = self._prepare_und_embeddings(
            packed_batch
        )
        raw_detail_states = self._prepare_raw_detail_states(
            raw_detail_source,
            intervention=detail_intervention,
            selected_sample_ids=selected_detail_samples,
        )
        deepstack_visual_embeds = all_deepstack_visual_embeds[
            : int(packed_batch.deepstack_stage_count)
        ]
        self.flex_attention.reset_call_count()
        slot_traces: List[UnifiedMoTSlotTrace] = []
        detail_summaries: List[torch.Tensor] = []
        slot_count = TASK_SLOT_COUNT[normalized_task]
        main_query_families = ("und", *requested_families)
        main_map = self._build_slot_index_map(
            task=normalized_task,
            batch=packed_batch,
            states=states,
            query_families=main_query_families,
        )
        main_mask = create_unified_block_mask(main_map, block_size=self.block_size, num_heads=1)
        tail_query_families = tuple(
            name for name in requested_families if len(states[name].base.blocks) > self.NUM_QWEN_LAYERS
        )
        tail_map = None
        tail_mask = None
        if slot_count > self.NUM_QWEN_LAYERS:
            if not tail_query_families:
                raise RuntimeError("A 30-slot task has no progressing tail GEN family.")
            tail_map = self._build_slot_index_map(
                task=normalized_task,
                batch=packed_batch,
                states=states,
                query_families=tail_query_families,
            )
            tail_mask = create_unified_block_mask(tail_map, block_size=self.block_size, num_heads=1)

        for slot_index in range(slot_count):
            if slot_index < self.NUM_QWEN_LAYERS:
                slot_map = main_map
                slot_mask = main_mask
            else:
                if tail_map is None or tail_mask is None:
                    raise RuntimeError("Tail slot metadata was not constructed.")
                slot_map = tail_map
                slot_mask = tail_mask
            und_hidden, slot_trace, detail_summary = self._run_slot(
                task=normalized_task,
                slot_index=slot_index,
                batch=packed_batch,
                und_hidden=und_hidden,
                states=states,
                index_map=slot_map,
                block_mask=slot_mask,
                raw_detail_source=raw_detail_source,
                raw_detail_states=raw_detail_states,
                timesteps=timesteps,
            )
            if deepstack_visual_embeds and slot_index < len(deepstack_visual_embeds):
                if packed_batch.visual_pos_mask is None:
                    raise RuntimeError("DeepStack features require a visual position mask.")
                und_hidden = self._inject_deepstack(
                    und_hidden,
                    packed_batch.visual_pos_mask,
                    deepstack_visual_embeds[slot_index],
                )
            slot_traces.append(slot_trace)
            if detail_summary is not None:
                detail_summaries.append(detail_summary)

        if self.flex_attention.call_count != slot_count:
            raise RuntimeError(
                f"A complete {normalized_task} forward requires {slot_count} FlexAttention calls, "
                f"got {self.flex_attention.call_count}."
            )
        und_final = self.qwen.model.language_model.norm(und_hidden)
        target_reference = packed_batch.family_inputs[target_family]
        velocity = self._finalize_family(states[target_family], reference=target_reference)
        trace = None
        max_kv = int(packed_batch.max_unified_tokens)
        if self.variant == "B":
            max_kv += len(requested_families) * int(packed_batch.max_condition_tokens)
        if return_trace:
            if detail_summaries:
                if len(detail_summaries) != slot_count:
                    raise RuntimeError("Detail-B did not produce one gate summary per global slot.")
                detail_rows = torch.stack(detail_summaries).float().cpu().tolist()
                decorated_slots = []
                time_labels = ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")
                for slot_trace, row in zip(slot_traces, detail_rows):
                    payload: Dict[str, Any] = {
                        "mean": row[0],
                        "variance": row[1],
                        "abs_mean": row[2],
                        "saturation_rate": row[3],
                        "raw_residual_to_und_norm_ratio": row[4],
                        "time_bins": {},
                    }
                    for bin_index, label in enumerate(time_labels):
                        start = 5 + 5 * bin_index
                        payload["time_bins"][label] = {
                            "mean": row[start],
                            "variance": row[start + 1],
                            "abs_mean": row[start + 2],
                            "saturation_rate": row[start + 3],
                            "token_count": int(row[start + 4]),
                        }
                    decorated_slots.append(replace(slot_trace, detail_gate=payload))
                slot_traces = decorated_slots
            trace = UnifiedMoTForwardTrace(
                task=normalized_task,
                variant=self.variant,
                slot_count=slot_count,
                flex_call_count=self.flex_attention.call_count,
                backend=self.flex_attention.backend_name,
                logical_sample_count=packed_batch.logical_sample_count,
                max_unified_tokens=int(packed_batch.max_unified_tokens),
                max_key_value_tokens=max_kv,
                active_query_tokens=active_query_tokens,
                query_padding_tokens=int(packed_batch.max_unified_tokens) - active_query_tokens,
                ss_ape_injections=1,
                slots=tuple(slot_traces),
                condition_mode_counts={
                    mode: (
                        packed_batch.condition_modes
                        or ("text",) * packed_batch.logical_sample_count
                    ).count(mode)
                    for mode in ("text", "image", "both")
                },
                visual_token_count=(
                    0
                    if packed_batch.visual_pos_mask is None
                    else int(packed_batch.visual_pos_mask.sum().detach().cpu())
                ),
                vision_patch_token_count=(
                    0
                    if packed_batch.image_grid_thw is None
                    else int(packed_batch.image_grid_thw.prod(dim=-1).sum().detach().cpu())
                ),
                deepstack_injections=len(deepstack_visual_embeds),
                visual_detail_mode=self.visual_detail_mode,
                visual_detail_tap=(
                    self.visual_detail_tap if self.visual_detail_mode == "haar_gated" else None
                ),
                raw_detail_token_count=(
                    0 if raw_detail_source is None else int(raw_detail_source.sample_ids.numel())
                ),
            )
        logical_map = torch.arange(
            packed_batch.logical_sample_count,
            device=device,
            dtype=torch.long,
        )
        return UnifiedGenerationOutput(
            velocity=velocity,
            logical_sample_ids=packed_batch.logical_sample_ids,
            logical_sample_map=logical_map,
            und_hidden=und_final,
            trace=trace,
            padding_stats={
                "query_active": active_query_tokens,
                "query_padding": int(packed_batch.max_unified_tokens) - active_query_tokens,
                "query_total": int(packed_batch.max_unified_tokens),
                "key_value_total": max_kv,
            },
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[Any] = None,
    ) -> UnifiedGenerationOutput:
        """Pack legacy logical rows into the physical-batch-one t2ss interface."""

        if use_cache or past_key_values is not None:
            raise ValueError("EVA01 Unified Generation forbids Qwen KV cache and layer memory.")
        if self.task_scope != "t2ss":
            raise ValueError("The legacy forward wrapper is available only for t2ss scope.")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must share logical shape [P,T].")
        logical_count = int(input_ids.shape[0])
        token_rows = []
        sample_rows = []
        position_rows = []
        for sample_index in range(logical_count):
            valid = attention_mask[sample_index].to(dtype=torch.bool)
            tokens = input_ids[sample_index, valid]
            if tokens.numel() == 0:
                raise ValueError("Every logical sample requires a non-empty complete caption.")
            token_rows.append(tokens)
            sample_rows.append(
                torch.full_like(tokens, sample_index, dtype=torch.long, device=tokens.device)
            )
            if position_ids is None:
                positions = torch.arange(tokens.numel(), device=tokens.device, dtype=torch.long)
                position_rows.append(positions.unsqueeze(0).expand(3, -1))
            else:
                normalized = position_ids
                if normalized.ndim == 2:
                    normalized = normalized.unsqueeze(0).expand(3, -1, -1)
                position_rows.append(normalized[:, sample_index, valid])
        packed_ids = torch.cat(token_rows).view(1, -1)
        packed_samples = torch.cat(sample_rows).view(1, -1)
        packed_positions = torch.cat(position_rows, dim=1).unsqueeze(1)
        active = int(packed_ids.shape[1]) + logical_count * self.SS_TOKENS
        max_tokens = ((active + self.block_size - 1) // self.block_size) * self.block_size
        packed = PackedGenerationBatch(
            input_ids=packed_ids,
            und_sample_ids=packed_samples,
            und_position_ids=packed_positions,
            logical_sample_ids=tuple(str(index) for index in range(logical_count)),
            family_inputs={"ss": x_t},
            max_unified_tokens=max_tokens,
            max_condition_tokens=int(packed_ids.shape[1]),
        )
        return self.forward_generation(
            "t2ss",
            packed,
            timesteps,
            return_trace=return_trace,
        )


# Backward-compatible imports now point to the shared executor; they do not
# represent a separate task-specific model or parameter bank.
TextToSSUnifiedMoT = EVA01UnifiedGenerationMoT
TextToSSUnifiedMoTOutput = UnifiedGenerationOutput


__all__ = [
    "EVA01UnifiedGenerationMoT",
    "PackedGenerationBatch",
    "TASK_FAMILIES",
    "TASK_SLOT_COUNT",
    "TASK_TARGET_FAMILY",
    "TextToSSUnifiedMoT",
    "TextToSSUnifiedMoTOutput",
    "UnifiedGenerationOutput",
    "UnifiedMoTForwardTrace",
    "UnifiedMoTSlotTrace",
]
