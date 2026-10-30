"""Generation-only inference primitives for EVA01 Unified MoT."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

from PIL import Image
import torch
import torch.nn.functional as F

from ..modeling.t2ss.all_task import (
    ALL_TASK_FAMILIES,
    AllTaskFamilyChunk,
    AllTaskPackedBatch,
    AllTaskUnifiedMoT,
)
from ..modeling.t2ss.runtime_types import UND_TOKEN_IMAGE, UND_TOKEN_SPECIAL, UND_TOKEN_TEXT
from ..modeling.t2ss.all_task_positions import (
    AllTaskVirtualStageSpec,
    gather_all_task_reference_position_plan,
)
from ..modeling.t2ss.all_task_tokens import FAMILY_TOKEN_IDS
from ..trellis2.modules import sparse as sp
from .normalization import Trellis2LatentNormalization

GENERATION_CONDITION_SYSTEM = (
    "You are EVA01.\n"
    "Task: 3D asset generation.\n"
    "Create a new 3D asset from the provided condition.\n"
    "There is no source mesh to preserve."
)

def _generation_condition_task(mode: str, caption: str = "") -> str:
    if mode == "text":
        return f"Task: 3D generation.\nInput modality: text.\nDescription:\n{caption}"
    if mode == "image":
        if caption:
            raise ValueError("Image-only generation cannot carry a caption.")
        return "Task: 3D generation.\nInput modality: image."
    raise ValueError(f"Unsupported generation mode: {mode!r}.")

def _extract_ids(encoded: Any) -> torch.Tensor:
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    values = values.detach().cpu() if isinstance(values, torch.Tensor) else torch.tensor(values)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError("Expected one token row.")
        values = values[0]
    return values.to(dtype=torch.long)

def _token_kinds(ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    image_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    kinds = torch.full_like(ids, UND_TOKEN_TEXT)
    kinds[ids.eq(image_id)] = UND_TOKEN_IMAGE
    special_mask = torch.zeros_like(ids, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        special_mask |= ids.eq(int(token_id))
    kinds[special_mask & ~ids.eq(image_id)] = UND_TOKEN_SPECIAL
    return kinds

def _append_generation_boundaries(ids: torch.Tensor) -> torch.Tensor:
    boundary_ids = [token_id for family in ("ss", "shape", "material") for token_id in FAMILY_TOKEN_IDS[family][:2]]
    return torch.cat((ids, torch.tensor(boundary_ids, dtype=torch.long)))

def _deletion_map(conditional: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
    conditional = conditional.flatten().cpu()
    null = null.flatten().cpu()
    indices = []
    cursor = 0
    for token in null.tolist():
        while cursor < conditional.numel() and int(conditional[cursor]) != int(token):
            cursor += 1
        if cursor >= conditional.numel():
            raise ValueError("CFG null tokens are not an ordered deletion of the condition.")
        indices.append(cursor)
        cursor += 1
    result = torch.tensor(indices, dtype=torch.long)
    if not torch.equal(conditional.index_select(0, result), null):
        raise RuntimeError("CFG reference map is invalid.")
    return result

def prepare_generation_condition(
    tokenizer: Any,
    processor: Any,
    *,
    prompt: str,
    image: str | Path | None,
    max_unified_tokens: int,
) -> "AllTaskPreparedCondition":
    mode = "image" if image is not None else "text"
    pixels = grid = None
    if mode == "text":
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": GENERATION_CONDITION_SYSTEM},
             {"role": "user", "content": _generation_condition_task(mode, prompt)}],
            tokenize=False, add_generation_prompt=True,
        )
        conditional = _append_generation_boundaries(_extract_ids(tokenizer(rendered, add_special_tokens=False, truncation=False)))
    else:
        with Image.open(Path(image).expanduser()) as handle:
            rgb = handle.convert("RGB")
            rendered = processor.apply_chat_template(
                [{"role": "system", "content": GENERATION_CONDITION_SYSTEM},
                 {"role": "user", "content": [
                     {"type": "image", "image": rgb},
                     {"type": "text", "text": _generation_condition_task(mode)},
                 ]}], tokenize=False, add_generation_prompt=True,
            )
            encoded = processor(text=[rendered], images=[rgb], padding=False, truncation=False, return_tensors="pt")
        conditional = _append_generation_boundaries(_extract_ids(encoded))
        pixels = encoded["pixel_values"].detach().cpu()
        grid = encoded["image_grid_thw"].detach().cpu()
    null_rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": GENERATION_CONDITION_SYSTEM},
         {"role": "user", "content": _generation_condition_task(mode, "")}],
        tokenize=False, add_generation_prompt=True,
    )
    null = _append_generation_boundaries(_extract_ids(tokenizer(null_rendered, add_special_tokens=False, truncation=False)))
    reference = _deletion_map(conditional, null)
    if conditional.numel() > int(max_unified_tokens):
        raise RuntimeError("Generation condition exceeds the Unified MoT token budget.")
    zeros = torch.zeros_like(conditional, dtype=torch.bool)
    collated = {
        "conditional_token_rows": (conditional,),
        "null_token_rows": (null,),
        "null_reference_indices": (reference,),
        "conditional_token_kind_rows": (_token_kinds(conditional, tokenizer),),
        "null_token_kind_rows": (_token_kinds(null, tokenizer),),
        "condition_mask_rows": (torch.ones_like(conditional, dtype=torch.bool),),
        "assistant_mask_rows": (zeros,),
        "qa_query_mask_rows": (zeros,),
        "qa_label_rows": (),
        "conditional_pixel_values": () if pixels is None else (pixels,),
        "conditional_image_grids": () if grid is None else (grid,),
        "point_clouds": None,
    }
    return AllTaskPreparedCondition(
        task="generation", mode=mode, record_id="public-generation",
        collated=collated, max_unified_tokens=int(max_unified_tokens),
    )

@dataclass(frozen=True)
class AllTaskPreparedCondition:
    task: str
    mode: str
    record_id: str
    collated: Mapping[str, Any]
    max_unified_tokens: int


@dataclass(frozen=True)
class EulerTrace:
    family: str
    steps: int
    cfg_scale: float
    flex_calls_first_conditional: int
    flex_calls_first_null: int
    path_length: float
    straight_distance: float
    path_over_straight: float
    endpoint_velocity_norm: float
    condition_delta_norm: float
    finite: bool


@dataclass(frozen=True)
class AllTaskCascadeOutput:
    ss: torch.Tensor
    coords: torch.Tensor
    shape_normalized: sp.SparseTensor
    shape: sp.SparseTensor
    material_normalized: sp.SparseTensor
    material: sp.SparseTensor
    traces: Mapping[str, EulerTrace]


_GENERATION_FAMILY_ORDER = ("ss", "shape", "material")


def _generation_prefix_length(row: torch.Tensor, family: str) -> int:
    """Keep the formal macro prefix through the family sampled this pass.

    Cascade inference runs one native GEN at a time, while packed training owns
    one ``SS -> Shape -> Material`` virtual macro.  Earlier native chunks must
    therefore remain in the running-base position layout, and later chunks must
    not be declared before their sparse coordinates exist.  This helper only
    trims the trailing, not-yet-active boundary pairs; it never removes the
    active V2 family boundary or switches to the legacy position path.
    """

    if family not in _GENERATION_FAMILY_ORDER:
        raise ValueError(f"Unknown generation family {family!r}.")
    boundary_ids = torch.tensor(
        [
            token_id
            for name in _GENERATION_FAMILY_ORDER
            for token_id in FAMILY_TOKEN_IDS[name][:2]
        ],
        device=row.device,
        dtype=row.dtype,
    )
    if int(row.numel()) < int(boundary_ids.numel()) or not torch.equal(
        row[-boundary_ids.numel() :], boundary_ids
    ):
        raise RuntimeError(
            "V2 generation condition must end with the complete ordered "
            "SS/Shape/Material boundary ABI."
        )
    prefix = int(row.numel()) - int(boundary_ids.numel())
    return prefix + 2 * (_GENERATION_FAMILY_ORDER.index(family) + 1)


def _generation_position_prefix_specs(
    family: str,
    chunks: Mapping[str, Tuple[AllTaskFamilyChunk, ...]],
) -> list[AllTaskVirtualStageSpec]:
    """Represent already-preceding macro chunks without native hidden copies."""

    active = tuple(chunks.get(family, ()))
    if len(active) != 1 or active[0].role != "target_noisy":
        raise RuntimeError(
            "Generation cascade inference requires exactly one active target chunk."
        )
    specs: list[AllTaskVirtualStageSpec] = []
    if family in {"shape", "material"}:
        specs.append(
            AllTaskVirtualStageSpec(
                name=f"ss:runtime:{family}:position-prefix",
                family="ss",
                role="target_noisy",
                sample_id=0,
                token_count=4096,
            )
        )
    if family == "material":
        value = active[0].value
        if not isinstance(value, sp.SparseTensor):
            raise TypeError("Material cascade state must remain SparseTensor.")
        coords = value.coords[:, 1:4].detach().to(device="cpu", dtype=torch.long)
        specs.append(
            AllTaskVirtualStageSpec(
                name="shape:runtime:material:position-prefix",
                family="shape",
                role="target_noisy",
                sample_id=0,
                token_count=int(coords.shape[0]),
                coords=coords,
            )
        )
    return specs

def _condition_batch(
    model: AllTaskUnifiedMoT,
    tokenizer: Any,
    prepared: AllTaskPreparedCondition,
    family_chunks: Mapping[str, Tuple[AllTaskFamilyChunk, ...]],
    *,
    use_null: bool,
) -> AllTaskPackedBatch:
    data = prepared.collated
    active_families = tuple(
        family for family in ALL_TASK_FAMILIES if family_chunks.get(family)
    )
    if prepared.task == "generation" and len(active_families) != 1:
        raise RuntimeError(
            "Cascade generation must activate exactly one native GEN family."
        )
    device = next(model.parameters()).device
    conditional_reference_row = data["conditional_token_rows"][0].to(
        device=device,
        dtype=torch.long,
    )
    row = (
        data["null_token_rows"][0]
        if use_null
        else data["conditional_token_rows"][0]
    ).to(device=device, dtype=torch.long)
    reference_indices = (
        data["null_reference_indices"][0]
        if use_null
        else torch.arange(
            int(conditional_reference_row.numel()),
            dtype=torch.long,
        )
    ).detach().to(device="cpu", dtype=torch.long)
    kinds = (
        data["null_token_kind_rows"][0]
        if use_null
        else data["conditional_token_kind_rows"][0]
    ).to(device=row.device, dtype=torch.long)
    condition_mask = (
        torch.ones_like(row, dtype=torch.bool)
        if use_null
        else data["condition_mask_rows"][0].to(device=row.device, dtype=torch.bool)
    )
    if prepared.task == "generation":
        reference_keep = _generation_prefix_length(
            conditional_reference_row,
            active_families[0],
        )
        selected_keep = _generation_prefix_length(row, active_families[0])
        conditional_reference_row = conditional_reference_row[:reference_keep]
        row = row[:selected_keep]
        kinds = kinds[:selected_keep]
        condition_mask = condition_mask[:selected_keep]
        reference_indices = reference_indices[:selected_keep]
        if int(reference_indices[-1]) >= int(reference_keep):
            raise RuntimeError("CFG prefix map crosses a later generation family boundary.")
    if not torch.equal(
        conditional_reference_row.detach().cpu().index_select(
            0,
            reference_indices,
        ),
        row.detach().cpu(),
    ):
        raise RuntimeError("CFG row does not match its conditional reference map.")
    pixels = None
    grid = None
    image_ids = None
    reference_grid = (
        data["conditional_image_grids"][0].to(device=row.device)
        if data["conditional_image_grids"]
        else None
    )
    if not use_null and data["conditional_pixel_values"]:
        pixels = data["conditional_pixel_values"][0].to(device=row.device)
        grid = reference_grid
        image_ids = torch.zeros((1,), device=row.device, dtype=torch.long)
    input_ids = row.view(1, -1)
    kinds = kinds.view(1, -1)
    point_clouds = data.get("point_clouds")
    if point_clouds is not None:
        point_clouds = point_clouds.to(device=row.device, dtype=torch.float32)
    point_mask = input_ids.eq(
        int(tokenizer.convert_tokens_to_ids("<point_patch>"))
    )
    if point_clouds is None:
        point_mask = None
    chunks = {
        family: tuple(family_chunks.get(family, ()))
        for family in ALL_TASK_FAMILIES
    }
    stage_specs = (
        _generation_position_prefix_specs(active_families[0], chunks)
        if prepared.task == "generation"
        else []
    )
    for family in ALL_TASK_FAMILIES:
        for chunk in chunks[family]:
            if family == "ss":
                stage_specs.append(
                    AllTaskVirtualStageSpec(
                        name=chunk.name,
                        family=family,
                        role=chunk.role,
                        sample_id=chunk.sample_id,
                        token_count=4096,
                    )
                )
            else:
                if not isinstance(chunk.value, sp.SparseTensor):
                    raise TypeError(f"{family} runtime chunk must remain SparseTensor.")
                coords = chunk.value.coords[:, 1:4].detach().to(
                    device="cpu",
                    dtype=torch.long,
                )
                stage_specs.append(
                    AllTaskVirtualStageSpec(
                        name=chunk.name,
                        family=family,
                        role=chunk.role,
                        sample_id=chunk.sample_id,
                        token_count=int(coords.shape[0]),
                        coords=coords,
                    )
                )
    reference_position_plan = model.build_all_task_position_plan(
        [conditional_reference_row],
        [reference_grid],
        stage_specs,
    )
    position_plan = gather_all_task_reference_position_plan(
        reference_position_plan,
        [reference_indices],
    )
    chunks = {
        family: tuple(
            replace(
                chunk,
                position_offset=0,
                position_ids=position_plan.family_position_ids[chunk.name].to(
                    device=row.device
                ),
            )
            for chunk in chunks[family]
        )
        for family in ALL_TASK_FAMILIES
    }
    position_ids = position_plan.compact_position_ids.to(device=row.device)
    qa_query_row = data.get("qa_query_mask_rows", (torch.zeros_like(row, dtype=torch.bool),))[0]
    if int(qa_query_row.numel()) != int(row.numel()):
        qa_query_row = torch.zeros_like(row, dtype=torch.bool)
    qa_label_rows = data.get("qa_label_rows", ())
    qa_labels = (
        torch.cat(tuple(qa_label_rows)).to(device=row.device, dtype=torch.long)
        if qa_label_rows
        else torch.empty((0,), device=row.device, dtype=torch.long)
    )
    qa_query_row = qa_query_row.to(device=row.device, dtype=torch.bool)
    qa_query_indices = qa_query_row.nonzero(as_tuple=False).flatten()
    qa_label_slices = ((0, int(qa_labels.numel())),) if qa_labels.numel() else ()
    assistant_row = data.get("assistant_mask_rows", (torch.zeros_like(row, dtype=torch.bool),))[0]
    if int(assistant_row.numel()) != int(row.numel()):
        assistant_row = torch.zeros_like(row, dtype=torch.bool)
    batch = AllTaskPackedBatch(
        input_ids=input_ids,
        und_sample_ids=torch.zeros_like(input_ids),
        und_position_ids=position_ids,
        logical_sample_ids=(prepared.record_id,),
        family_chunks=chunks,
        max_unified_tokens=int(prepared.max_unified_tokens),
        max_key_value_tokens=int(prepared.max_unified_tokens),
        und_position_offsets=tuple(
            int(value) for value in position_plan.next_position_bases
        ),
        condition_modes=(prepared.mode,),
        pixel_values=pixels,
        image_grid_thw=grid,
        image_sample_ids=image_ids,
        visual_pos_mask=(kinds.eq(UND_TOKEN_IMAGE) if pixels is not None else None),
        und_token_kinds=kinds,
        point_clouds=point_clouds,
        point_cloud_sample_ids=(
            torch.zeros((1,), device=row.device, dtype=torch.long)
            if point_clouds is not None
            else None
        ),
        point_pos_mask=point_mask,
        condition_pos_mask=condition_mask.view(1, -1),
        und_boundary_family=position_plan.boundary_family.to(device=row.device),
        und_boundary_role=position_plan.boundary_role.to(device=row.device),
        virtual_position_plan_passed=True,
        virtual_sample_lengths=position_plan.virtual_sample_lengths,
        assistant_pos_mask=assistant_row.to(device=row.device, dtype=torch.bool).view(1, -1),
        qa_query_pos_mask=qa_query_row.view(1, -1),
        qa_query_token_indices=qa_query_indices,
        qa_label_ids=qa_labels,
        qa_label_slices=qa_label_slices,
        record_ids=(prepared.record_id,),
        pack_id=0,
        branch_by_sample=("gen",),
    )
    batch.validate()
    return batch


def _sparse(coords: torch.Tensor, feats: torch.Tensor) -> sp.SparseTensor:
    if coords.ndim != 2 or coords.shape[1] not in {3, 4}:
        raise ValueError("Sparse runtime coords must have shape [N,3] or [N,4].")
    if coords.shape[1] == 3:
        coords = torch.cat(
            [
                torch.zeros((coords.shape[0], 1), device=coords.device, dtype=coords.dtype),
                coords,
            ],
            dim=1,
        )
    return sp.SparseTensor(feats=feats, coords=coords.to(dtype=torch.int32))


def _family_chunks(
    family: str,
    state: Union[torch.Tensor, sp.SparseTensor],
    timestep: torch.Tensor,
    *,
    target_shape: Optional[sp.SparseTensor] = None,
) -> Mapping[str, Tuple[AllTaskFamilyChunk, ...]]:
    chunks = [
        AllTaskFamilyChunk(
            name=f"{family}:runtime:target",
            family=family,
            role="target_noisy",
            sample_id=0,
            value=state,
            timestep=timestep,
            concat_shape=target_shape,
            chunk_id=f"{family}:runtime:target",
            branch="gen",
            task_kind="generation",
        )
    ]
    return {
        name: tuple(chunks if name == family else ()) for name in ALL_TASK_FAMILIES
    }


@torch.inference_mode()
def _velocity(
    model: AllTaskUnifiedMoT,
    tokenizer: Any,
    prepared: AllTaskPreparedCondition,
    family: str,
    state: Union[torch.Tensor, sp.SparseTensor],
    timestep: torch.Tensor,
    *,
    use_null: bool,
    target_shape: Optional[sp.SparseTensor] = None,
) -> Tuple[Union[torch.Tensor, sp.SparseTensor], int]:
    chunks = _family_chunks(
        family,
        state,
        timestep,
        target_shape=target_shape,
    )
    batch = _condition_batch(
        model,
        tokenizer,
        prepared,
        chunks,
        use_null=use_null,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.forward_all_task(batch, return_trace=True)
    expected = 28 if family == "ss" else 30
    flex_calls = int(output.trace.get("flex_calls", -1))
    if flex_calls != expected or len(output.trace.get("slots", ())) != expected:
        raise RuntimeError(
            f"{family} runtime requires {expected} direct Flex calls, got {flex_calls}."
        )
    result = output.family_velocities[f"{family}:runtime:target"]
    return result, flex_calls


@torch.inference_mode()
def sample_family_euler_cfg(
    model: AllTaskUnifiedMoT,
    tokenizer: Any,
    prepared: AllTaskPreparedCondition,
    family: str,
    initial: Union[torch.Tensor, sp.SparseTensor],
    *,
    steps: int,
    cfg_scale: float,
    target_shape: Optional[sp.SparseTensor] = None,
) -> Tuple[Union[torch.Tensor, sp.SparseTensor], EulerTrace]:
    if family not in ALL_TASK_FAMILIES or int(steps) <= 0:
        raise ValueError("Euler sampling requires a known family and steps > 0.")
    state = initial
    initial_feats = initial if isinstance(initial, torch.Tensor) else initial.feats
    path_length = 0.0
    condition_delta = 0.0
    endpoint_velocity_norm = 0.0
    first_conditional = first_null = -1
    time_grid = torch.linspace(
        1.0,
        0.0,
        int(steps) + 1,
        device=initial_feats.device,
        dtype=torch.float32,
    )
    for index in range(int(steps)):
        timestep = time_grid[index].view(1)
        conditional, conditional_calls = _velocity(
            model,
            tokenizer,
            prepared,
            family,
            state,
            timestep,
            use_null=False,
            target_shape=target_shape,
        )
        unconditional, null_calls = _velocity(
            model,
            tokenizer,
            prepared,
            family,
            state,
            timestep,
            use_null=True,
            target_shape=target_shape,
        )
        if index == 0:
            first_conditional = conditional_calls
            first_null = null_calls
        conditional_feats = (
            conditional
            if isinstance(conditional, torch.Tensor)
            else conditional.feats
        ).float()
        unconditional_feats = (
            unconditional
            if isinstance(unconditional, torch.Tensor)
            else unconditional.feats
        ).float()
        velocity = unconditional_feats + float(cfg_scale) * (
            conditional_feats - unconditional_feats
        )
        delta = (time_grid[index + 1] - time_grid[index]) * velocity
        path_length += float(delta.reshape(-1).norm().cpu())
        condition_delta += float(
            (conditional_feats - unconditional_feats).reshape(-1).norm().cpu()
        )
        endpoint_velocity_norm = float(velocity.reshape(-1).norm().cpu())
        if isinstance(state, torch.Tensor):
            state = state.float() + delta
        else:
            state = state.replace(state.feats.float() + delta)
    endpoint_feats = state if isinstance(state, torch.Tensor) else state.feats
    straight = float((endpoint_feats.float() - initial_feats.float()).reshape(-1).norm().cpu())
    finite = bool(torch.isfinite(endpoint_feats).all())
    trace = EulerTrace(
        family=family,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        flex_calls_first_conditional=first_conditional,
        flex_calls_first_null=first_null,
        path_length=path_length,
        straight_distance=straight,
        path_over_straight=path_length / max(straight, 1.0e-12),
        endpoint_velocity_norm=endpoint_velocity_norm,
        condition_delta_norm=condition_delta / float(steps),
        finite=finite,
    )
    if not finite:
        raise FloatingPointError(f"Non-finite {family} Euler endpoint.")
    return state, trace


@torch.inference_mode()
def decode_ss_to_shape_coords(
    ss: torch.Tensor,
    decoder: torch.nn.Module,
    *,
    target_resolution: int = 32,
    max_sparse_tokens: int | None = None,
) -> torch.Tensor:
    """Decode SS occupancy into Shape/Material sparse coordinates.

    The generated cascade is an inference path, so it preserves every
    decoded occupied voxel.  In particular, a sample whose decoded support is
    larger than the 8,192-token training-domain reference is still passed to
    Shape and Material unchanged.  Callers that are auditing a fixed domain
    may provide ``max_sparse_tokens`` explicitly; that optional audit limit
    raises instead of truncating, while the default inference behavior is
    unlimited.
    """
    runtime = next(decoder.parameters())
    logits = decoder(ss.to(device=runtime.device, dtype=runtime.dtype)).float()
    occupied = logits > 0
    native_resolution = int(occupied.shape[-1])
    if native_resolution != int(target_resolution):
        if native_resolution % int(target_resolution):
            raise RuntimeError("SS decoder resolution cannot map exactly to Shape-512 coords.")
        ratio = native_resolution // int(target_resolution)
        occupied = F.max_pool3d(occupied.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(occupied)[:, [0, 2, 3, 4]].to(dtype=torch.int32)
    if coords.numel() == 0:
        raise RuntimeError("Generated SS decoded to an empty Shape coordinate set.")
    if max_sparse_tokens is not None and int(coords.shape[0]) > int(max_sparse_tokens):
        raise RuntimeError(
            f"Generated SS produced {coords.shape[0]} coords, above {max_sparse_tokens}; "
            "inference does not truncate sparse tokens."
        )
    return coords


def _noise_like_sparse(
    coords: torch.Tensor,
    channels: int,
    *,
    seed: int,
    device: torch.device,
) -> sp.SparseTensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    feats = torch.randn(
        (coords.shape[0], int(channels)),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return _sparse(coords.to(device=device), feats)


def _noise_ss(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(
        (1, 8, 16, 16, 16),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

@torch.inference_mode()
def sample_generation_cascade(
    model: AllTaskUnifiedMoT,
    tokenizer: Any,
    prepared: AllTaskPreparedCondition,
    normalization: Trellis2LatentNormalization,
    ss_decoder: torch.nn.Module,
    *,
    seed: int,
    steps: int = 25,
    cfg_scale: float = 3.0,
) -> AllTaskCascadeOutput:
    device = next(model.parameters()).device
    ss, ss_trace = sample_family_euler_cfg(
        model,
        tokenizer,
        prepared,
        "ss",
        _noise_ss(seed, device),
        steps=steps,
        cfg_scale=cfg_scale,
    )
    if not isinstance(ss, torch.Tensor):
        raise TypeError("SS generation must return a dense tensor.")
    coords = decode_ss_to_shape_coords(ss, ss_decoder).to(device=device)
    shape_normalized, shape_trace = sample_family_euler_cfg(
        model,
        tokenizer,
        prepared,
        "shape",
        _noise_like_sparse(coords, 32, seed=seed + 1, device=device),
        steps=steps,
        cfg_scale=cfg_scale,
    )
    if not isinstance(shape_normalized, sp.SparseTensor):
        raise TypeError("Shape generation must return a TRELLIS2 SparseTensor.")
    material_normalized, material_trace = sample_family_euler_cfg(
        model,
        tokenizer,
        prepared,
        "material",
        _noise_like_sparse(coords, 32, seed=seed + 2, device=device),
        target_shape=shape_normalized,
        steps=steps,
        cfg_scale=cfg_scale,
    )
    if not isinstance(material_normalized, sp.SparseTensor):
        raise TypeError("Material generation must return a TRELLIS2 SparseTensor.")
    if not torch.equal(shape_normalized.coords, material_normalized.coords):
        raise RuntimeError("Material coordinates must exactly match Shape coordinates.")
    if not torch.equal(shape_normalized.coords, coords.to(shape_normalized.coords.device)):
        raise RuntimeError("Shape coordinates must exactly match decoded SS coordinates.")
    if shape_normalized.feats.ndim != 2 or shape_normalized.feats.shape[-1] != 32:
        raise RuntimeError("Shape features must have native width 32.")
    if material_normalized.feats.ndim != 2 or material_normalized.feats.shape[-1] != 32:
        raise RuntimeError("Material features must have native width 32.")
    shape = shape_normalized.replace(
        normalization.denormalize_shape(shape_normalized.feats)
    )
    material = material_normalized.replace(
        normalization.denormalize_material(material_normalized.feats)
    )
    if not torch.isfinite(shape.feats).all() or not torch.isfinite(material.feats).all():
        raise FloatingPointError("Non-finite denormalized Shape or Material endpoint.")
    return AllTaskCascadeOutput(
        ss=ss,
        coords=coords,
        shape_normalized=shape_normalized,
        shape=shape,
        material_normalized=material_normalized,
        material=material,
        traces={
            "ss": ss_trace,
            "shape": shape_trace,
            "material": material_trace,
        },
    )

__all__ = [
    "AllTaskCascadeOutput",
    "AllTaskPreparedCondition",
    "EulerTrace",
    "decode_ss_to_shape_coords",
    "prepare_generation_condition",
    "sample_family_euler_cfg",
    "sample_generation_cascade",
]
