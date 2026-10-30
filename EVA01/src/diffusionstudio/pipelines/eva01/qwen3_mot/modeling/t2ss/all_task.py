"""EVA01 Unified All-Task V2 runtime.

This module deliberately lives beside the legacy t2ss executor.  The legacy
Current-B path remains available for regression, while this file owns the new
slot-canonical, family-native condition ABI used by all-task training.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...trellis2.modules import sparse as sp
from ...trellis2.modules.utils import manual_cast
from ..attention.rope import build_dense_raster_position_ids
from .all_task_adapters import (
    ALL_TASK_FAMILIES,
    ALL_TASK_FROZEN_NUMEL,
    ALL_TASK_SCHEMA,
    ALL_TASK_TRAINABLE_KEYS,
    ALL_TASK_TRAINABLE_NUMEL,
    GENERATION_BRANCH_SCHEMA,
    AllTaskAdapterRegistry,
)
from .all_task_checkpointing import AllTaskSlotCheckpointMixin
from .all_task_layout import (
    ROLE_CODE,
    AllTaskFamilyChunk,
    AllTaskIndexMap,
    AllTaskOutput,
    AllTaskPackedBatch,
    AllTaskTokenAxis,
    build_all_task_index_map,
    create_all_task_block_mask,
)
from .all_task_projection import alltask_batched_lora_triplet
from .all_task_condition import (
    project_condition_view,
    validate_native_condition_projections,
)
from .all_task_positions import (
    AllTaskVirtualPositionPlan,
    AllTaskVirtualStageSpec,
    build_all_task_running_position_plan,
)
from .all_task_runtime_layout import AllTaskRuntimeLayoutMixin
from .all_task_tokens import (
    SPECIAL_EMBEDDING_ROW_BY_TOKEN_ID,
    SPECIAL_EMBEDDING_TOKEN_IDS,
)
from .base_loader import (
    load_qwen_base,
    load_slat_base,
    load_ss_xl_base,
)
from .flex_only import require_flex_attention
from .runtime_types import (
    FamilyRuntimeState,
    PackedGenerationBatch,
    family_carrier_view,
    family_features,
)
from .unified_model import EVA01UnifiedGenerationMoT


ACTIVATION_CHECKPOINT_SCHEDULE = "evenly_spaced_slots_v1"
class AllTaskUnifiedMoT(AllTaskSlotCheckpointMixin, AllTaskRuntimeLayoutMixin, EVA01UnifiedGenerationMoT):
    """Full-forward slot-canonical executor for all-task carts."""

    _alltask_batched_lora_triplet = staticmethod(alltask_batched_lora_triplet)

    def __init__(
        self,
        *,
        qwen: nn.Module,
        ss_xl: nn.Module,
        shape_model: nn.Module,
        material_model: nn.Module,
        adapters: AllTaskAdapterRegistry,
        pointbert: Optional[nn.Module] = None,
        compile_flex_attention: bool = True,
        block_size: int = 128,
        activation_checkpointing: bool = False,
        activation_checkpoint_slots: Optional[int] = None,
        parallel_native_branches: bool = True,
    ) -> None:
        super().__init__(
            qwen=qwen,
            ss_xl=ss_xl,
            shape_model=shape_model,
            material_model=material_model,
            variant="B",
            task_scope="all_task",
            adapters=adapters,  # type: ignore[arg-type]
            compile_flex_attention=compile_flex_attention,
            block_size=block_size,
            visual_detail_mode="none",
        )
        self.pointbert = pointbert
        resolved_checkpoint_slots = (30 if bool(activation_checkpointing) else 0) if activation_checkpoint_slots is None else int(activation_checkpoint_slots)
        if not 0 <= resolved_checkpoint_slots <= 30:
            raise ValueError("activation_checkpoint_slots must be in [0,30].")
        if bool(activation_checkpointing) and resolved_checkpoint_slots != 30:
            raise ValueError("Legacy activation_checkpointing=True requires all 30 slots.")
        self.activation_checkpoint_slots = resolved_checkpoint_slots
        self.activation_checkpointing = resolved_checkpoint_slots > 0
        self.activation_checkpoint_schedule = ACTIVATION_CHECKPOINT_SCHEDULE
        self.parallel_native_branches = bool(parallel_native_branches)
        # Side-stream native branches are safe in the retained forward graph,
        # but not while non-reentrant checkpoint recompute mutates shared
        # FamilyRuntimeState objects.  The checkpoint mixin reads this flag;
        # test harnesses without the attribute keep their historical behavior.
        self.checkpoint_recompute_parallel_native = False
        self._native_branch_streams: Dict[
            Tuple[int, str], torch.cuda.Stream
        ] = {}
        if self.pointbert is not None:
            self.pointbert.eval().requires_grad_(False)
        validate_native_condition_projections(self)

    def train(self, mode: bool = True):
        """Keep the frozen PointBERT deterministic during adapter training."""

        super().train(mode)
        if self.pointbert is not None:
            self.pointbert.eval()
        return self

    @property
    def active_frozen_parameter_count(self) -> int:
        """Count the actually materialized frozen parameters after pruning.

        This is deliberately computed from ``named_parameters`` instead of a
        hand-written number.  TRELLIS checkpoints have changed native cross-KV
        ABI across releases; the receipt must expose the real post-prune count.
        PointBERT is included because it is a frozen active base in the
        all-task runtime.
        """

        return sum(
            parameter.numel()
            for base in (
                self.qwen,
                self.ss_xl,
                self.shape_model,
                self.material_model,
                self.pointbert,
            )
            if base is not None
            for parameter in base.parameters()
            if not parameter.requires_grad
        )

    def parameter_inventory(self) -> Dict[str, Any]:
        frozen = {
            "qwen": sum(p.numel() for p in self.qwen.parameters()),
            "ss": sum(p.numel() for p in self.ss_xl.parameters()),
            "shape": sum(p.numel() for p in self.shape_model.parameters()),
            "material": sum(p.numel() for p in self.material_model.parameters()),
            "pointbert": 0 if self.pointbert is None else sum(p.numel() for p in self.pointbert.parameters()),
        }
        trainable = self.adapters.trainable_parameter_count
        active_frozen = self.active_frozen_parameter_count
        expected_keys = ALL_TASK_TRAINABLE_KEYS
        expected_numel = ALL_TASK_TRAINABLE_NUMEL
        if getattr(self.adapters, "generation_only", False):
            expected_keys = len(tuple(self.adapters.state_dict()))
            expected_numel = self.adapters.trainable_parameter_count
        if trainable != expected_numel or len(tuple(self.adapters.state_dict())) != expected_keys:
            raise RuntimeError("All-task trainable inventory drifted.")
        return {
            "architecture": ALL_TASK_SCHEMA,
            "frozen_by_family": frozen,
            "frozen_total": sum(frozen.values()),
            "active_frozen_total": active_frozen,
            "plan_frozen_total": ALL_TASK_FROZEN_NUMEL,
            "plan_frozen_total_match": active_frozen == ALL_TASK_FROZEN_NUMEL,
            "trainable_total": trainable,
            "trainable_key_count": len(tuple(self.adapters.state_dict())),
            "model_total": sum(frozen.values()) + trainable,
            "adapter_schema": self.adapters.schema_inventory(),
        }

    def forward(
        self,
        batch: AllTaskPackedBatch,
        *,
        return_trace: bool = True,
    ) -> AllTaskOutput:
        """Standard Module/DDP entrypoint for the unified all-task graph."""

        output = self.forward_all_task(batch, return_trace=return_trace)
        # Formal carts naturally activate Qwen, PointBERT connector, the one
        # shared condition carrier and all three GEN banks.  Only a condition
        # path omitted by this concrete cart needs a zero-valued DDP edge.
        # Keeping the edge local to inactive modules avoids constructing 1,832
        # scalar autograd nodes on every forward while preserving a stable
        # reducer graph with find_unused_parameters=False.
        inactive_modules: list[nn.Module] = []
        if batch.pixel_values is None:
            inactive_modules.append(self.adapters.vision)
        if batch.point_pos_mask is None:
            inactive_modules.append(self.adapters.connector)
        if not any(
            bool(batch.input_ids.eq(int(token_id)).any())
            for token_id in SPECIAL_EMBEDDING_TOKEN_IDS
        ):
            inactive_modules.append(self.adapters.special_token_embeddings)
        active_family = False
        inactive_branch_parameters: list[nn.Parameter] = []
        for family in ALL_TASK_FAMILIES:
            family_chunks = tuple(batch.family_chunks.get(family, ()))
            if family_chunks:
                active_family = True
            else:
                inactive_modules.append(self.adapters.families[family])
        if not active_family:
            inactive_modules.extend(
                (self.adapters.canonical_condition, self.adapters.conditions)
            )
        inactive_parameters = [
            parameter
            for module in inactive_modules
            for parameter in module.parameters()
        ]
        inactive_parameters.extend(inactive_branch_parameters)
        # A family-level anchor and a branch-level anchor can refer to the same
        # tensor.  Keep one edge per tensor to avoid needless scalar nodes.
        unique_parameters = []
        seen_parameter_ids = set()
        for parameter in inactive_parameters:
            if id(parameter) not in seen_parameter_ids:
                seen_parameter_ids.add(id(parameter))
                unique_parameters.append(parameter)
        inactive_parameters = unique_parameters
        output.ddp_anchor = (
            torch.stack(
                [parameter.reshape(-1)[0] for parameter in inactive_parameters]
            ).sum()
            * 0.0
            if inactive_parameters
            else None
        )
        return output

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: torch.device,
        adapter_seed: int = 20260820,
        branch_schema: str = GENERATION_BRANCH_SCHEMA,
        gen_rank: int = 64,
        compile_flex_attention: bool = True,
        block_size: int = 128,
        qwen_root: Path,
        ss_xl_stem: Path,
        shape_stem: Path,
        material_stem: Path,
        activation_checkpointing: bool = False,
        activation_checkpoint_slots: Optional[int] = None,
        parallel_native_branches: bool = True,
    ) -> Tuple["AllTaskUnifiedMoT", Dict[str, Any]]:
        """Load the frozen Qwen and native 3D bases for generation."""

        qwen, qwen_receipt = load_qwen_base(Path(qwen_root), device=device)
        migration_receipts: Dict[str, Any] = {}
        migration_receipts["qwen_dense_initialization"] = {
            "passed": True,
            "source": "original_qwen3_vl_2b",
            "openeva_language_lora_merged": False,
            "openeva_language_lora_runtime": False,
        }
        ss, ss_receipt = load_ss_xl_base(Path(ss_xl_stem), device=device)
        shape, shape_receipt = load_slat_base(Path(shape_stem), family="shape", device=device)
        material, material_receipt = load_slat_base(Path(material_stem), family="material", device=device)
        adapters = AllTaskAdapterRegistry(
            seed=adapter_seed,
            branch_schema=branch_schema,
            gen_rank=gen_rank,
        ).to(device=device)
        migration_receipts[
            "special_token_embeddings"
        ] = adapters.initialize_special_token_embeddings(
            qwen.model.language_model.embed_tokens.weight
        )
        pointbert = None
        model = cls(
            qwen=qwen,
            ss_xl=ss,
            shape_model=shape,
            material_model=material,
            adapters=adapters,
            pointbert=pointbert,
            compile_flex_attention=compile_flex_attention,
            block_size=block_size,
            activation_checkpointing=activation_checkpointing,
            activation_checkpoint_slots=activation_checkpoint_slots,
            parallel_native_branches=parallel_native_branches,
        )
        return model, {
            "qwen": qwen_receipt.to_dict(),
            "ss": ss_receipt.to_dict(),
            "shape": shape_receipt.to_dict(),
            "material": material_receipt.to_dict(),
            "migration": migration_receipts,
            "parameters": model.parameter_inventory(),
            "runtime": {
                "activation_checkpointing": bool(model.activation_checkpointing),
                "activation_checkpoint_slots": int(model.activation_checkpoint_slots),
                "activation_checkpoint_schedule": (model.activation_checkpoint_schedule),
                "parallel_native_branches": bool(model.parallel_native_branches),
            },
        }

    @staticmethod
    def _legacy_batch(batch: AllTaskPackedBatch) -> PackedGenerationBatch:
        return PackedGenerationBatch(
            input_ids=batch.input_ids,
            und_sample_ids=batch.und_sample_ids,
            und_position_ids=batch.und_position_ids,
            logical_sample_ids=batch.logical_sample_ids,
            family_inputs={},
            max_unified_tokens=batch.max_unified_tokens,
            max_condition_tokens=batch.input_ids.shape[1],
            condition_modes=batch.condition_modes or ("text",) * len(batch.logical_sample_ids),
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
            image_sample_ids=batch.image_sample_ids,
            visual_pos_mask=batch.visual_pos_mask,
            visual_token_indices=batch.visual_token_indices,
            und_token_kinds=batch.und_token_kinds,
        )

    def _prepare_alltask_und(self, batch: AllTaskPackedBatch) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        hidden, deepstack, _ = self._prepare_und_embeddings(self._legacy_batch(batch))
        special_positions: List[torch.Tensor] = []
        special_rows: List[torch.Tensor] = []
        for token_id, row in SPECIAL_EMBEDDING_ROW_BY_TOKEN_ID.items():
            positions = batch.input_ids[0].eq(int(token_id)).nonzero(
                as_tuple=False
            ).flatten()
            if positions.numel():
                special_positions.append(positions)
                special_rows.append(
                    torch.full_like(positions, int(row), dtype=torch.long)
                )
        if special_positions:
            if not self.adapters.special_token_embeddings_initialized:
                raise RuntimeError(
                    "Special-token embeddings were used before Qwen-row initialization."
                )
            positions = torch.cat(special_positions).to(
                device=hidden.device,
                dtype=torch.long,
            )
            rows = torch.cat(special_rows).to(
                device=hidden.device,
                dtype=torch.long,
            )
            order = positions.argsort()
            positions = positions.index_select(0, order)
            rows = rows.index_select(0, order)
            overrides = self.adapters.special_token_embeddings(rows).to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            hidden = hidden.index_copy(1, positions, overrides.unsqueeze(0))
        if batch.point_pos_mask is not None:
            point_embeds = batch.point_embeds
            if point_embeds is None:
                if self.pointbert is None or batch.point_clouds is None:
                    raise RuntimeError("Raw point clouds require the frozen OpenEVA PointBERT.")
                if int(batch.point_clouds.shape[0]) > 6:
                    raise RuntimeError("One physical cart may contain at most six PointBERT inputs.")
                # PointBERT is frozen and its batch dimension is independent.
                # A real H20 probe verified batch<=6 is BF16-exact relative to
                # row-wise execution while using less than 0.7 GiB peak memory.
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=hidden.device.type,
                        enabled=False,
                    ),
                ):
                    tokens = self.pointbert(batch.point_clouds.to(device=hidden.device, dtype=torch.float32))
                expected = (int(batch.point_clouds.shape[0]), 513, 384)
                if tuple(tokens.shape) != expected:
                    raise RuntimeError(f"OpenEVA PointBERT ABI drifted: {tuple(tokens.shape)} != {expected}.")
                point_embeds = tokens.detach().flatten(0, 1)
            assert point_embeds is not None
            point_tokens = self.adapters.connector(point_embeds.to(device=hidden.device, dtype=hidden.dtype))
            point_indices = batch.point_token_indices
            if point_indices is None:
                point_indices = batch.point_pos_mask[0].nonzero(
                    as_tuple=False
                ).flatten()
            point_indices = point_indices.to(
                device=hidden.device,
                dtype=torch.long,
            )
            if int(point_indices.numel()) != int(point_tokens.shape[0]):
                raise RuntimeError(
                    "Point token indices do not match connector output rows."
                )
            hidden = hidden.index_copy(
                1,
                point_indices,
                point_tokens.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                ).unsqueeze(0),
            )
        return hidden, deepstack

    def build_all_task_position_plan(
        self,
        token_rows: Sequence[torch.Tensor],
        image_grid_rows: Sequence[Optional[torch.Tensor]],
        stage_specs: Sequence[AllTaskVirtualStageSpec],
    ) -> AllTaskVirtualPositionPlan:
        """Build running-base positions through the virtual expanded sequence."""

        config = self.qwen.config
        vision_config = getattr(config, "vision_config", None)
        spatial_merge_size = int(
            getattr(
                vision_config,
                "spatial_merge_size",
                getattr(self.qwen.model.visual, "spatial_merge_size", 2),
            )
        )
        return build_all_task_running_position_plan(
            token_rows=token_rows,
            image_grid_rows=image_grid_rows,
            stage_specs=stage_specs,
            spatial_merge_size=spatial_merge_size,
            image_token_id=int(getattr(config, "image_token_id")),
            video_token_id=int(getattr(config, "video_token_id")),
            vision_start_token_id=int(getattr(config, "vision_start_token_id")),
        )

    @staticmethod
    def _condition_mask(batch: AllTaskPackedBatch) -> torch.Tensor:
        if batch.condition_pos_mask is not None:
            return batch.condition_pos_mask.to(device=batch.input_ids.device, dtype=torch.bool)
        if batch.assistant_pos_mask is not None:
            return ~batch.assistant_pos_mask.to(device=batch.input_ids.device, dtype=torch.bool)
        return torch.ones_like(batch.input_ids, dtype=torch.bool)

    @staticmethod
    def _condition_indices(batch: AllTaskPackedBatch) -> torch.Tensor:
        if batch.condition_token_indices is not None:
            return batch.condition_token_indices.to(
                device=batch.input_ids.device,
                dtype=torch.long,
            )
        return AllTaskUnifiedMoT._condition_mask(batch)[0].nonzero(
            as_tuple=False
        ).flatten()

    @staticmethod
    def _alltask_fused_qkv_base(
        block: nn.Module,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return a non-persistent frozen cross-Q/native-KV projection cache.

        All-Task V1 reads cross-Q plus the native self-attention K/V slices.
        The three frozen matrices never change during adapter training, so one
        contiguous cache removes two GEMM launches without changing any
        parameter owner or checkpoint key.  The source tensor versions remain
        part of the cache identity so an in-place base reload invalidates it.
        """

        query_projection = block.cross_attn.to_q
        native_projection = block.self_attn.to_qkv
        hidden = int(query_projection.out_features)
        if (
            int(query_projection.in_features) != hidden
            or int(native_projection.in_features) != hidden
            or int(native_projection.out_features) != 3 * hidden
        ):
            raise ValueError("All-task fused QKV base projection ABI drifted.")
        cache_key = (
            int(query_projection.weight.data_ptr()),
            int(query_projection.weight._version),
            int(native_projection.weight.data_ptr()),
            int(native_projection.weight._version),
            str(query_projection.weight.device),
            str(query_projection.weight.dtype),
        )
        cached = getattr(block, "_eva01_alltask_fused_qkv_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]

        with torch.no_grad():
            weight = torch.cat(
                (
                    query_projection.weight,
                    native_projection.weight[hidden : 3 * hidden],
                ),
                dim=0,
            ).contiguous()
            query_bias = query_projection.bias
            native_bias = native_projection.bias
            if query_bias is None and native_bias is None:
                bias = None
            else:
                zero = weight.new_zeros((hidden,))
                bias = torch.cat(
                    (
                        zero if query_bias is None else query_bias,
                        (
                            weight.new_zeros((2 * hidden,))
                            if native_bias is None
                            else native_bias[hidden : 3 * hidden]
                        ),
                    ),
                    dim=0,
                ).contiguous()
        object.__setattr__(
            block,
            "_eva01_alltask_fused_qkv_cache",
            (cache_key, weight, bias),
        )
        return weight, bias

    def _project_alltask_family_qkv(
        self,
        state: FamilyRuntimeState,
        *,
        block_index: int,
        mot_input: Union[torch.Tensor, sp.SparseTensor],
        branch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project one family/slot with fused frozen QKV and batched LoRA."""

        block = state.base.blocks[int(block_index)]
        adapters = self.adapters.family_slot(state.name, block_index)
        features = family_features(mot_input)
        weight, bias = self._alltask_fused_qkv_base(block)
        projected = F.linear(features, weight, bias)
        prefix = tuple(int(value) for value in projected.shape[:-1])
        projected = projected.reshape(*prefix, 3, adapters.spec.hidden_size).movedim(
            -2, 0
        )
        projected = projected + self._alltask_batched_lora_triplet(
            features,
            (adapters.q, adapters.k, adapters.v),
            branch_mask=branch_mask,
        ).to(dtype=projected.dtype)
        query, key, value = projected.unbind(dim=0)
        query = query.view(
            *query.shape[:-1], adapters.spec.num_heads, adapters.spec.head_dim
        )
        key = key.view(
            *key.shape[:-1], adapters.spec.num_heads, adapters.spec.head_dim
        )
        value = value.view(
            *value.shape[:-1], adapters.spec.num_heads, adapters.spec.head_dim
        )
        if getattr(block.cross_attn, "qk_rms_norm", False):
            query = block.cross_attn.q_rms_norm(query)
        if getattr(block.self_attn, "qk_rms_norm", False):
            key = block.self_attn.k_rms_norm(key)
        return (
            adapters.q_bridge(family_carrier_view(query)),
            adapters.k_bridge(family_carrier_view(key)),
            adapters.v_bridge(family_carrier_view(value)),
        )

    @staticmethod
    def _u_offset(batch: AllTaskPackedBatch, sample_id: int) -> int:
        if batch.und_position_offsets:
            return int(batch.und_position_offsets[int(sample_id)])
        positions = batch.und_position_ids[:, 0, batch.und_sample_ids[0].eq(int(sample_id))]
        if positions.numel() == 0:
            raise ValueError(f"No UND tokens for logical sample {sample_id}.")
        return int(positions.max().item()) + 1

    def _prepare_ss_chunk(self, chunk: AllTaskFamilyChunk, batch: AllTaskPackedBatch) -> FamilyRuntimeState:
        latent = chunk.value
        if not isinstance(latent, torch.Tensor) or latent.shape != (1, 8, 16, 16, 16):
            raise ValueError(f"SS chunk {chunk.name} must be [1,8,16,16,16].")
        hidden = latent.view(1, 8, -1).permute(0, 2, 1).contiguous()
        input_dtype = next(self.ss_xl.input_layer.parameters()).dtype
        hidden = self.ss_xl.input_layer(hidden.to(dtype=input_dtype)) + self.ss_xl.pos_emb[None].to(dtype=input_dtype)
        t_emb = manual_cast(self.ss_xl.t_embedder(chunk.timestep.float() * 1000.0), self.ss_xl.dtype)
        hidden = manual_cast(hidden, self.ss_xl.dtype)
        if chunk.position_ids is not None:
            positions = chunk.position_ids.to(device=latent.device, dtype=torch.long)
            if positions.shape != (3, 1, 4096):
                raise ValueError("SS virtual position rows must be [3,1,4096].")
        else:
            offset = self._u_offset(batch, chunk.sample_id) + int(
                chunk.position_offset
            )
            positions = build_dense_raster_position_ids(
                (16, 16, 16),
                offset,
                device=latent.device,
            ).unsqueeze(1)
        return FamilyRuntimeState(
            name="ss",
            base=self.ss_xl,
            hidden=hidden,
            timestep_embedding=t_emb,
            sample_ids=torch.full(
                (1, self.ss_xl.resolution**3),
                chunk.sample_id,
                device=latent.device,
                dtype=torch.long,
            ),
            local_indices=torch.arange(self.ss_xl.resolution**3, device=latent.device, dtype=torch.long).view(1, -1),
            position_ids=positions,
            role=chunk.role,
        )

    def _prepare_sparse_chunk(self, chunk: AllTaskFamilyChunk, batch: AllTaskPackedBatch) -> FamilyRuntimeState:
        if not isinstance(chunk.value, sp.SparseTensor):
            raise TypeError(f"{chunk.name} must use native SparseTensor.")
        base = self.shape_model if chunk.family == "shape" else self.material_model
        value = chunk.value
        if chunk.concat_shape is not None:
            if chunk.family != "material":
                raise ValueError("Only Material accepts a per-sample shape channel concat.")
            if not torch.equal(value.coords, chunk.concat_shape.coords):
                raise ValueError("Material and its clean Shape half must have identical coords.")
            value = sp.sparse_cat([value, chunk.concat_shape], dim=-1)
        hidden = base.input_layer(value)
        hidden = manual_cast(hidden, base.dtype)
        t_embed_dtype = next(base.t_embedder.parameters()).dtype
        t_emb = base.t_embedder((chunk.timestep.float() * 1000.0).to(dtype=t_embed_dtype))
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, base.dtype)
        if base.pe_mode == "ape":
            position_embedding = manual_cast(
                base.pos_embedder(hidden.coords[:, 1:]),
                base.dtype,
            )
            hidden = hidden.replace(hidden.feats + position_embedding)
        count = hidden.feats.shape[0]
        native_coords = hidden.coords[:, 1:4].to(dtype=torch.long)
        if chunk.position_ids is not None:
            positions = chunk.position_ids.to(
                device=native_coords.device,
                dtype=torch.long,
            )
            if positions.shape != (3, 1, count):
                raise ValueError(
                    f"{chunk.family} virtual position rows must be [3,1,N]."
                )
            coords = positions[:, 0]
        else:
            offset = self._u_offset(batch, chunk.sample_id) + int(
                chunk.position_offset
            )
            coords = native_coords.T + offset
        return FamilyRuntimeState(
            name=chunk.family,
            base=base,
            hidden=hidden,
            timestep_embedding=t_emb,
            sample_ids=torch.full((1, count), chunk.sample_id, device=coords.device, dtype=torch.long),
            local_indices=torch.arange(count, device=coords.device, dtype=torch.long).view(1, -1),
            position_ids=coords.unsqueeze(1),
            role=chunk.role,
        )

    def _prepare_family_states(self, batch: AllTaskPackedBatch) -> Dict[str, List[Tuple[AllTaskFamilyChunk, FamilyRuntimeState]]]:
        result: Dict[str, List[Tuple[AllTaskFamilyChunk, FamilyRuntimeState]]] = {name: [] for name in ALL_TASK_FAMILIES}
        has_chunks = any(batch.family_chunks.get(name) for name in ALL_TASK_FAMILIES)
        if batch.und_boundary_family is not None and has_chunks:
            if not batch.virtual_position_plan_passed:
                raise RuntimeError(
                    "Boundary-bearing V2 batches require a passed virtual position plan."
                )
        for family, chunks in batch.family_chunks.items():
            if family not in result:
                raise ValueError(f"Unknown all-task family {family}.")
            for chunk in chunks:
                if chunk.family != family or chunk.role not in ROLE_CODE:
                    raise ValueError(f"Invalid chunk metadata for {chunk.name}.")
                if int(chunk.position_offset) < 0:
                    raise ValueError(f"Chunk {chunk.name} has a negative position offset.")
                if batch.virtual_position_plan_passed:
                    if int(chunk.position_offset) != 0 or chunk.position_ids is None:
                        raise RuntimeError(
                            "Formal running-base chunks forbid manual position offsets."
                        )
                state = self._prepare_ss_chunk(chunk, batch) if family == "ss" else self._prepare_sparse_chunk(chunk, batch)
                result[family].append((chunk, state))
        return result

    def _run_all_slot(
        self,
        batch: AllTaskPackedBatch,
        states: Dict[str, List[Tuple[AllTaskFamilyChunk, FamilyRuntimeState]]],
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
        *,
        parallel_native_branches: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        q: Dict[str, torch.Tensor] = {}
        k: Dict[str, torch.Tensor] = {}
        v: Dict[str, torch.Tensor] = {}
        active_states: Dict[str, Tuple[AllTaskFamilyChunk, FamilyRuntimeState]] = {}
        parallel_native = bool(
            parallel_native_branches
            and self.parallel_native_branches
            and und_hidden.is_cuda
            and torch.cuda.is_available()
        )
        branch_streams: Dict[str, torch.cuda.Stream] = {}
        current_stream: Optional[torch.cuda.Stream] = None
        if parallel_native:
            device_index = int(und_hidden.device.index or 0)
            current_stream = torch.cuda.current_stream(und_hidden.device)
            for name in ("und", *ALL_TASK_FAMILIES):
                key = (device_index, str(name))
                stream = self._native_branch_streams.get(key)
                if stream is None:
                    stream = torch.cuda.Stream(device=und_hidden.device)
                    self._native_branch_streams[key] = stream
                stream.wait_stream(current_stream)
                branch_streams[str(name)] = stream
        if slot < 28:
            if parallel_native:
                und_hidden.record_stream(branch_streams["und"])
            und_context = (
                torch.cuda.stream(branch_streams["und"])
                if parallel_native
                else nullcontext()
            )
            with und_context:
                layer = self.qwen.model.language_model.layers[slot]
                und_residual = und_hidden
                normalized = layer.input_layernorm(und_hidden)
                q_und, k_und, v_und = self._project_qwen_qkv(
                    layer=layer,
                    adapters=self.adapters.qwen_slot(slot),
                    normalized_hidden=normalized,
                )
                assert q_und is not None
                common_rows, boundary_rows = self._und_axis_indices(batch)
                q["und"] = q_und.index_select(1, common_rows)
                k["und"] = k_und.index_select(1, common_rows)
                v["und"] = v_und.index_select(1, common_rows)
                if int(boundary_rows.numel()):
                    q["und_boundary"] = q_und.index_select(1, boundary_rows)
                    k["und_boundary"] = k_und.index_select(1, boundary_rows)
                    v["und_boundary"] = v_und.index_select(1, boundary_rows)
        for family in ALL_TASK_FAMILIES:
            active_family = [
                (chunk, state)
                for chunk, state in states[family]
                if slot < len(state.base.blocks)
            ]
            if not active_family:
                continue
            if parallel_native:
                for _chunk, state in active_family:
                    self._record_runtime_state_on_stream(
                        state,
                        branch_streams[family],
                    )
            family_context = (
                torch.cuda.stream(branch_streams[family])
                if parallel_native
                else nullcontext()
            )
            with family_context:
                grouped_sparse = bool(
                    len(active_family) > 1
                    and all(
                        isinstance(state.hidden, sp.SparseTensor)
                        for _chunk, state in active_family
                    )
                )
                if grouped_sparse:
                    self._grouped_sparse_native_self_pre_mot(
                        active_family,
                        block_index=slot,
                    )
                family_inputs: List[
                    Tuple[
                        AllTaskFamilyChunk,
                        FamilyRuntimeState,
                        torch.Tensor,
                        int,
                        torch.Tensor,
                    ]
                ] = []
                for chunk, state in active_family:
                    if not grouped_sparse:
                        self._family_native_self_pre_mot(state, block_index=slot)
                    mot_input = self._family_mot_input(state, block_index=slot)
                    features = family_features(mot_input)
                    token_count = int(
                        features.shape[1]
                        if features.ndim == 3
                        else features.shape[0]
                    )
                    branch_mask = torch.full(
                        (token_count,),
                        False,
                        device=features.device,
                        dtype=torch.bool,
                    )
                    family_inputs.append(
                        (chunk, state, features, token_count, branch_mask)
                    )
                    active_states[chunk.name] = (chunk, state)

                feature_rank = family_inputs[0][2].ndim
                if any(value[2].ndim != feature_rank for value in family_inputs):
                    raise RuntimeError(
                        f"{family} chunks mixed dense and sparse layouts."
                    )
                concat_dim = 1 if feature_rank == 3 else 0
                combined = (
                    family_inputs[0][2]
                    if len(family_inputs) == 1
                    else torch.cat(
                        tuple(value[2] for value in family_inputs),
                        dim=concat_dim,
                    )
                )
                combined_branch = torch.cat(
                    tuple(value[4] for value in family_inputs),
                    dim=0,
                )
                first_state = family_inputs[0][1]
                q_all, k_all, v_all = self._project_alltask_family_qkv(
                    first_state,
                    block_index=slot,
                    mot_input=combined,
                    branch_mask=combined_branch,
                )
                cursor = 0
                for chunk, _state, _features, token_count, _branch_mask in family_inputs:
                    end = cursor + token_count
                    q[chunk.name] = q_all[:, cursor:end]
                    k[chunk.name] = k_all[:, cursor:end]
                    v[chunk.name] = v_all[:, cursor:end]
                    cursor = end
                if cursor != int(q_all.shape[1]):
                    raise RuntimeError(
                        f"{family} grouped QKV split did not cover its rows."
                    )
        if parallel_native:
            assert current_stream is not None
            for tensor in (*q.values(), *k.values(), *v.values()):
                tensor.record_stream(current_stream)
            for stream in branch_streams.values():
                current_stream.wait_stream(stream)
        condition_names = tuple(
            name
            for name in index_map.key_value_slices
            if name.startswith("condition_")
        )
        if condition_names:
            active_condition_sample_ids_by_family = {
                family: tuple(
                    dict.fromkeys(
                        int(chunk.sample_id)
                        for chunk, state in states[family]
                        if slot < len(state.base.blocks)
                    )
                )
                for family in ALL_TASK_FAMILIES
            }
            active_condition_sample_ids = tuple(
                dict.fromkeys(
                    sample_id
                    for family in ALL_TASK_FAMILIES
                    for sample_id in active_condition_sample_ids_by_family[family]
                )
            )
            global_condition_indices = self._condition_indices_for_samples(
                batch,
                active_condition_sample_ids,
            )
            if not int(global_condition_indices.numel()):
                raise RuntimeError("Active GEN chunks have no condition rows.")
            canonical_condition = self.adapters.canonical_condition_states(
                slot,
                und_hidden.index_select(1, global_condition_indices),
            )
            for family in ALL_TASK_FAMILIES:
                family_names = tuple(
                    name
                    for name in condition_names
                    if self._condition_axis_family(name) == family
                )
                if not family_names:
                    continue
                family_sample_ids = active_condition_sample_ids_by_family[family]
                relative_rows = []
                lengths = []
                for name in family_names:
                    role = self._condition_axis_role(name)
                    condition_indices = self._condition_axis_indices(
                        family,
                        role,
                        batch,
                        family_sample_ids,
                    )
                    expected_start, expected_end = index_map.key_value_slices[name]
                    expected_length = int(expected_end) - int(expected_start)
                    if int(condition_indices.numel()) != expected_length:
                        raise RuntimeError(
                            "Condition axis rows drifted from the immutable slot layout: "
                            f"slot={slot}, axis={name}, expected={expected_length}, "
                            f"observed={int(condition_indices.numel())}, "
                            f"sample_ids={family_sample_ids}."
                        )
                    relative = torch.searchsorted(
                        global_condition_indices,
                        condition_indices,
                    )
                    relative_rows.append(relative)
                    lengths.append(int(relative.numel()))
                combined_relative = torch.cat(relative_rows)
                if batch.branch_by_sample:
                    global_sample_ids = batch.und_sample_ids[0].index_select(
                        0, global_condition_indices
                    )
                    global_branch = torch.tensor(
                        [
                            False
                            for sample_id in global_sample_ids.detach().cpu().tolist()
                        ],
                        device=global_condition_indices.device,
                        dtype=torch.bool,
                    )
                    condition_branch_mask = global_branch.index_select(
                        0, combined_relative
                    )
                elif getattr(self.adapters, "routed_lora", False):
                    condition_branch_mask = torch.zeros(
                        int(combined_relative.numel()),
                        device=combined_relative.device,
                        dtype=torch.bool,
                    )
                else:
                    condition_branch_mask = None
                condition_k, condition_v = project_condition_view(
                    self,
                    family=family,
                    slot=slot,
                    canonical_condition=canonical_condition.index_select(
                        1,
                        combined_relative,
                    ),
                    branch_mask=condition_branch_mask,
                )
                cursor = 0
                for name, length in zip(family_names, lengths):
                    end = cursor + length
                    k[name] = condition_k[:, cursor:end]
                    v[name] = condition_v[:, cursor:end]
                    cursor = end
                if cursor != int(condition_k.shape[1]):
                    raise RuntimeError(
                        f"{family} condition split did not cover its rows."
                    )
        query = (
            self._pad(
                [q[name] for name in index_map.query_slices if name != "padding"],
                index_map.query_length,
            )
            .transpose(1, 2)
            .contiguous()
        )
        key = (
            self._pad(
                [k[name] for name in index_map.key_value_slices if name != "padding"],
                index_map.key_value_length,
            )
            .transpose(1, 2)
            .contiguous()
        )
        value = (
            self._pad(
                [v[name] for name in index_map.key_value_slices if name != "padding"],
                index_map.key_value_length,
            )
            .transpose(1, 2)
            .contiguous()
        )
        query, key = self._apply_cached_interleaved_mrope(
            query,
            key,
            mrope_factors,
        )
        output = self.flex_attention(query, key, value, block_mask=block_mask).transpose(1, 2).contiguous()
        if parallel_native:
            for stream in branch_streams.values():
                output.record_stream(stream)
        if slot < 28:
            und_context = (
                torch.cuda.stream(branch_streams["und"])
                if parallel_native
                else nullcontext()
            )
            if parallel_native:
                branch_streams["und"].wait_stream(current_stream)
            with und_context:
                common_rows, boundary_rows = self._und_axis_indices(batch)
                carrier = output.new_zeros((1, batch.input_ids.shape[1], 2048))
                start, end = index_map.query_slices["und"]
                carrier = carrier.index_copy(
                    1,
                    common_rows,
                    output[:, start:end].reshape(1, end - start, 2048),
                )
                if int(boundary_rows.numel()):
                    start, end = index_map.query_slices["und_boundary"]
                    carrier = carrier.index_copy(
                        1,
                        boundary_rows,
                        output[:, start:end].reshape(1, end - start, 2048),
                    )
                und_hidden = und_residual + self._project_qwen_output(
                    carrier, layer, slot
                )
                und_hidden = und_hidden + layer.mlp(
                    layer.post_attention_layernorm(und_hidden)
                )
        for family in ALL_TASK_FAMILIES:
            active_family = [
                (chunk, state)
                for chunk, state in states[family]
                if chunk.name in active_states
            ]
            family_context = (
                torch.cuda.stream(branch_streams[family])
                if parallel_native
                else nullcontext()
            )
            if parallel_native:
                branch_streams[family].wait_stream(current_stream)
            with family_context:
                carrier_outputs = []
                for chunk, _state in active_family:
                    start, end = index_map.query_slices[chunk.name]
                    carrier_outputs.append(output[:, start:end])
                grouped_sparse = bool(
                    len(active_family) > 1
                    and all(
                        isinstance(state.hidden, sp.SparseTensor)
                        for _chunk, state in active_family
                    )
                )
                if grouped_sparse:
                    self._apply_grouped_sparse_family_attention_output(
                        active_family,
                        block_index=slot,
                        carrier_outputs=carrier_outputs,
                    )
                else:
                    for (_chunk, state), carrier_output in zip(
                        active_family,
                        carrier_outputs,
                    ):
                        self._apply_family_attention_output(
                            state,
                            block_index=slot,
                            carrier_output=carrier_output,
                            branch_mask=torch.full(
                                (int(carrier_output.shape[1]),),
                                False,
                                device=carrier_output.device,
                                dtype=torch.bool,
                            ),
                        )
        if parallel_native:
            assert current_stream is not None
            for stream in branch_streams.values():
                current_stream.wait_stream(stream)
            if slot < 28:
                und_hidden.record_stream(current_stream)
            for family in ALL_TASK_FAMILIES:
                for _chunk, state in states[family]:
                    if slot < len(state.base.blocks):
                        self._record_runtime_state_on_stream(
                            state,
                            current_stream,
                        )
        stats = getattr(block_mask, "eva01_stats", None)
        trace = {
            "slot": int(slot),
            "query_length": index_map.query_length,
            "key_value_length": index_map.key_value_length,
            "query_slices": dict(index_map.query_slices),
            "key_value_slices": dict(index_map.key_value_slices),
            "condition_views": list(condition_names),
            "flex_call": int(self.flex_attention.call_count),
            "backend": self.flex_attention.backend_name,
            "native_branch_stream_parallel": parallel_native,
            "block_mask": {} if stats is None else stats.__dict__,
        }
        return und_hidden, trace

    def _project_qwen_output(self, carrier: torch.Tensor, layer: nn.Module, slot: int) -> torch.Tensor:
        return self.adapters.qwen_slot(slot).o(carrier) + layer.self_attn.o_proj(carrier)

    def forward_all_task(self, batch: AllTaskPackedBatch, *, return_trace: bool = True) -> AllTaskOutput:
        require_flex_attention()
        batch.validate()
        und_hidden, deepstack = self._prepare_alltask_und(batch)
        states = self._prepare_family_states(batch)
        self.flex_attention.reset_call_count()
        traces: List[Dict[str, Any]] = []
        has_long_family = any(states[name] for name in ("shape", "material"))
        slot_count = 30 if has_long_family else 28
        checkpoint_slots = frozenset(self.activation_checkpoint_slot_indices(slot_count))
        layout_cache: Dict[
            Tuple[bool, Tuple[str, ...]],
            Tuple[
                AllTaskIndexMap,
                Any,
                Tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
            ],
        ] = {}
        for slot in range(slot_count):
            active_names = tuple(chunk.name for family in ALL_TASK_FAMILIES for chunk, state in states[family] if slot < len(state.base.blocks))
            layout_key = (slot < 28, active_names)
            cached_layout = layout_cache.get(layout_key)
            if cached_layout is None:
                index_map = self._slot_map(batch, states, slot)
                mask = create_all_task_block_mask(
                    index_map,
                    block_size=self.block_size,
                )
                mrope_factors = self._interleaved_mrope_factors(
                    rotary_embedding=self.qwen.model.language_model.rotary_emb,
                    reference=und_hidden,
                    query_position_ids=index_map.query.position_ids,
                    key_position_ids=index_map.key_value.position_ids,
                )
                layout_cache[layout_key] = (
                    index_map,
                    mask,
                    mrope_factors,
                )
            else:
                index_map, mask, mrope_factors = cached_layout
            if slot in checkpoint_slots and self.training and torch.is_grad_enabled():
                und_hidden, trace = self._run_all_slot_checkpointed(
                    batch,
                    states,
                    und_hidden,
                    slot,
                    index_map,
                    mask,
                    mrope_factors,
                )
            else:
                und_hidden, trace = self._run_all_slot(
                    batch,
                    states,
                    und_hidden,
                    slot,
                    index_map,
                    mask,
                    mrope_factors,
                )
                trace = dict(trace)
                trace["activation_checkpointed"] = False
            if slot < len(deepstack) and batch.visual_pos_mask is not None:
                und_hidden = self._inject_deepstack(
                    und_hidden,
                    batch.visual_pos_mask,
                    deepstack[slot],
                    visual_token_indices=batch.visual_token_indices,
                )
            traces.append(trace)
        if self.flex_attention.call_count != slot_count:
            raise RuntimeError(f"All-task forward requires {slot_count} FlexAttention calls, got {self.flex_attention.call_count}.")
        und_final = self.qwen.model.language_model.norm(und_hidden)
        velocities: Dict[str, Union[torch.Tensor, sp.SparseTensor]] = {}
        for family in ALL_TASK_FAMILIES:
            for chunk, state in states[family]:
                if chunk.role != "target_noisy":
                    continue
                velocities[chunk.name] = self._finalize_family(state, reference=chunk.value)
        qa_logits = None
        if batch.qa_query_pos_mask is not None and batch.qa_label_ids is not None and batch.qa_label_ids.numel():
            qa_indices = batch.qa_query_token_indices
            if qa_indices is None:
                qa_indices = batch.qa_query_pos_mask[0].nonzero(
                    as_tuple=False
                ).flatten()
            qa_logits = self.qwen.lm_head(
                und_final[0].index_select(0, qa_indices)
            )
        return AllTaskOutput(
            family_velocities=velocities,
            und_hidden=und_final,
            qa_logits=qa_logits,
            trace={
                "schema": ALL_TASK_SCHEMA,
                "flex_calls": self.flex_attention.call_count,
                "slots": traces,
                "block_mask_layout_count": len(layout_cache),
                "activation_checkpointing": bool(checkpoint_slots and self.training and torch.is_grad_enabled()),
                "activation_checkpoint_slots_requested": int(self.activation_checkpoint_slots),
                "activation_checkpoint_slot_indices": sorted(checkpoint_slots),
                "activation_checkpoint_schedule": self.activation_checkpoint_schedule,
            },
        )
__all__ = ["ALL_TASK_SCHEMA", "ACTIVATION_CHECKPOINT_SCHEDULE", "AllTaskAdapterRegistry",
           "AllTaskFamilyChunk", "AllTaskIndexMap", "AllTaskOutput", "AllTaskPackedBatch", "AllTaskTokenAxis", "AllTaskUnifiedMoT", "OpenEVAConnector", "build_all_task_index_map", "create_all_task_block_mask"]
