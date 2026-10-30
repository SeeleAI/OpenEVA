"""Strict Qwen3VL-2B and TRELLIS1 SS-XL base loading for EVA01 t2ss."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from ....qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from ....qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)
from ...trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from ...trellis2.models.structured_latent_flow import SLatFlowModel


EXPECTED_QWEN_PARAMETER_COUNT = 2_127_532_032
EXPECTED_SS_PARAMETER_COUNT = 975_440_648
EXPECTED_SS_BUFFER_COUNT = 5_242_880
EXPECTED_SHAPE_PARAMETER_COUNT = 1_292_253_728
EXPECTED_MATERIAL_PARAMETER_COUNT = 1_292_302_880


@dataclass(frozen=True)
class CheckpointInspection:
    """Machine-readable checkpoint/header validation result."""

    family: str
    config_path: str
    weight_path: str
    checkpoint_key_count: int
    checkpoint_tensor_numel: int
    parameter_numel: int
    buffer_numel: int
    missing_model_keys: Tuple[str, ...]
    unexpected_checkpoint_keys: Tuple[str, ...]
    shape_mismatches: Tuple[Tuple[str, Tuple[int, ...], Tuple[int, ...]], ...]
    accepted_tied_keys: Tuple[str, ...]
    @property
    def passed(self) -> bool:
        unaccepted_missing = set(self.missing_model_keys).difference(self.accepted_tied_keys)
        return not unaccepted_missing and not self.unexpected_checkpoint_keys and not self.shape_mismatches

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class BaseLoadReceipt:
    """Actual materialized base-load result."""

    family: str
    device: str
    dtype_counts: Dict[str, int]
    parameter_numel: int
    buffer_numel: int
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    accepted_tied_keys: Tuple[str, ...]
    meta_tensor_names: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        unaccepted_missing = set(self.missing_keys).difference(self.accepted_tied_keys)
        return not unaccepted_missing and not self.unexpected_keys and not self.meta_tensor_names

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def _state_shape_map(module: nn.Module) -> Dict[str, Tuple[int, ...]]:
    return {name: tuple(tensor.shape) for name, tensor in module.state_dict().items()}


def _checkpoint_shape_map(path: Path) -> Tuple[Dict[str, Tuple[int, ...]], int]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    tensor_numel = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            shape = tuple(handle.get_slice(name).get_shape())
            shapes[name] = shape
            numel = 1
            for dimension in shape:
                numel *= int(dimension)
            tensor_numel += numel
    return shapes, tensor_numel


def _compare_shapes(
    model_shapes: Dict[str, Tuple[int, ...]],
    checkpoint_shapes: Dict[str, Tuple[int, ...]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[Tuple[str, Tuple[int, ...], Tuple[int, ...]], ...]]:
    model_keys = set(model_shapes)
    checkpoint_keys = set(checkpoint_shapes)
    missing = tuple(sorted(model_keys - checkpoint_keys))
    unexpected = tuple(sorted(checkpoint_keys - model_keys))
    mismatches = tuple(
        (name, model_shapes[name], checkpoint_shapes[name])
        for name in sorted(model_keys & checkpoint_keys)
        if model_shapes[name] != checkpoint_shapes[name]
    )
    return missing, unexpected, mismatches


def _build_ss_xl_meta_model(config_path: Path) -> SparseStructureFlowModel:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("name") != "SparseStructureFlowModel":
        raise ValueError("SS XL config must declare SparseStructureFlowModel.")
    arguments = dict(payload["args"])
    use_fp16 = bool(arguments.pop("use_fp16", True))
    arguments["dtype"] = "float16" if use_fp16 else "float32"
    with torch.device("meta"):
        model = SparseStructureFlowModel(**arguments)
    return model


def _build_slat_meta_model(config_path: Path) -> SLatFlowModel:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("name") != "SLatFlowModel":
        raise ValueError("TRELLIS2 SLat config must declare SLatFlowModel.")
    with torch.device("meta"):
        model = SLatFlowModel(**dict(payload["args"]))
    return model


def inspect_qwen_checkpoint(
    qwen_root: Path,
) -> CheckpointInspection:
    resolved_root = Path(qwen_root).expanduser().resolve()
    config_path = resolved_root / "config.json"
    weight_path = resolved_root / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete Qwen base at {resolved_root}.")
    config = Qwen3VLConfig.from_pretrained(str(resolved_root), local_files_only=True)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    model_shapes = _state_shape_map(model)
    checkpoint_shapes, tensor_numel = _checkpoint_shape_map(weight_path)
    missing, unexpected, mismatches = _compare_shapes(model_shapes, checkpoint_shapes)
    parameter_numel = sum(parameter.numel() for parameter in model.parameters())
    if parameter_numel != EXPECTED_QWEN_PARAMETER_COUNT:
        raise RuntimeError(
            f"Qwen parameter count mismatch: expected {EXPECTED_QWEN_PARAMETER_COUNT}, got {parameter_numel}."
        )
    receipt = CheckpointInspection(
        family="qwen3vl_2b",
        config_path=str(config_path),
        weight_path=str(weight_path),
        checkpoint_key_count=len(checkpoint_shapes),
        checkpoint_tensor_numel=tensor_numel,
        parameter_numel=parameter_numel,
        buffer_numel=sum(buffer.numel() for buffer in model.buffers()),
        missing_model_keys=missing,
        unexpected_checkpoint_keys=unexpected,
        shape_mismatches=mismatches,
        accepted_tied_keys=("lm_head.weight",),
    )
    if not receipt.passed:
        raise RuntimeError(f"Qwen checkpoint inspection failed: {receipt.to_dict()}.")
    return receipt


def inspect_ss_xl_checkpoint(
    ss_xl_stem: Path,
) -> CheckpointInspection:
    stem = Path(ss_xl_stem).expanduser().resolve()
    config_path = stem.with_suffix(".json")
    weight_path = stem.with_suffix(".safetensors")
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete TRELLIS1 SS XL base at {stem}.")
    model = _build_ss_xl_meta_model(config_path)
    model_shapes = _state_shape_map(model)
    checkpoint_shapes, tensor_numel = _checkpoint_shape_map(weight_path)
    missing, unexpected, mismatches = _compare_shapes(model_shapes, checkpoint_shapes)
    parameter_numel = sum(parameter.numel() for parameter in model.parameters())
    buffer_numel = sum(buffer.numel() for buffer in model.buffers())
    if parameter_numel != EXPECTED_SS_PARAMETER_COUNT or buffer_numel != EXPECTED_SS_BUFFER_COUNT:
        raise RuntimeError(
            "SS XL parameter/buffer count mismatch: "
            f"parameters={parameter_numel}, buffers={buffer_numel}."
        )
    receipt = CheckpointInspection(
        family="trellis1_ss_xl",
        config_path=str(config_path),
        weight_path=str(weight_path),
        checkpoint_key_count=len(checkpoint_shapes),
        checkpoint_tensor_numel=tensor_numel,
        parameter_numel=parameter_numel,
        buffer_numel=buffer_numel,
        missing_model_keys=missing,
        unexpected_checkpoint_keys=unexpected,
        shape_mismatches=mismatches,
        accepted_tied_keys=(),
    )
    if not receipt.passed:
        raise RuntimeError(f"SS XL checkpoint inspection failed: {receipt.to_dict()}.")
    return receipt


def inspect_slat_checkpoint(
    stem: Path,
    *,
    family: str,
) -> CheckpointInspection:
    """Inspect one real TRELLIS2 Shape or Material base without allocating weights."""

    normalized_family = str(family).strip().lower()
    expected_counts = {
        "shape": EXPECTED_SHAPE_PARAMETER_COUNT,
        "material": EXPECTED_MATERIAL_PARAMETER_COUNT,
    }
    if normalized_family not in expected_counts:
        raise ValueError("SLat family must be `shape` or `material`.")
    resolved_stem = Path(stem).expanduser().resolve()
    config_path = resolved_stem.with_suffix(".json")
    weight_path = resolved_stem.with_suffix(".safetensors")
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete TRELLIS2 {normalized_family} base at {resolved_stem}.")
    model = _build_slat_meta_model(config_path)
    model_shapes = _state_shape_map(model)
    checkpoint_shapes, tensor_numel = _checkpoint_shape_map(weight_path)
    missing, unexpected, mismatches = _compare_shapes(model_shapes, checkpoint_shapes)
    parameter_numel = sum(parameter.numel() for parameter in model.parameters())
    if parameter_numel != expected_counts[normalized_family]:
        raise RuntimeError(
            f"{normalized_family} parameter count mismatch: expected "
            f"{expected_counts[normalized_family]}, got {parameter_numel}."
        )
    receipt = CheckpointInspection(
        family=f"trellis2_{normalized_family}_512",
        config_path=str(config_path),
        weight_path=str(weight_path),
        checkpoint_key_count=len(checkpoint_shapes),
        checkpoint_tensor_numel=tensor_numel,
        parameter_numel=parameter_numel,
        buffer_numel=sum(buffer.numel() for buffer in model.buffers()),
        missing_model_keys=missing,
        unexpected_checkpoint_keys=unexpected,
        shape_mismatches=mismatches,
        accepted_tied_keys=(),
    )
    if not receipt.passed:
        raise RuntimeError(f"{normalized_family} checkpoint inspection failed: {receipt.to_dict()}.")
    return receipt


def _materialize_qwen_rotary_buffers_(
    model: Qwen3VLForConditionalGeneration,
    *,
    device: torch.device,
) -> None:
    text_rotary = Qwen3VLTextRotaryEmbedding(model.config.text_config, device=device)
    target_text = model.model.language_model.rotary_emb
    target_text.register_buffer("inv_freq", text_rotary.inv_freq.detach().clone(), persistent=False)
    target_text.original_inv_freq = target_text.inv_freq
    target_text.attention_scaling = text_rotary.attention_scaling

    vision_config = model.config.vision_config
    vision_head_dim = int(vision_config.hidden_size) // int(vision_config.num_heads)
    vision_rotary = Qwen3VLVisionRotaryEmbedding(vision_head_dim // 2).to(device)
    target_vision = model.model.visual.rotary_pos_emb
    target_vision.register_buffer("inv_freq", vision_rotary.inv_freq.detach().clone(), persistent=False)


def _dtype_counts(module: nn.Module) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for parameter in module.parameters():
        name = str(parameter.dtype).replace("torch.", "")
        counts[name] = counts.get(name, 0) + parameter.numel()
    return counts


def _meta_tensor_names(module: nn.Module) -> Tuple[str, ...]:
    names = [name for name, tensor in module.named_parameters() if tensor.device.type == "meta"]
    names.extend(name for name, tensor in module.named_buffers() if tensor.device.type == "meta")
    return tuple(sorted(names))


def load_qwen_base(
    qwen_root: Path,
    *,
    device: torch.device,
) -> Tuple[Qwen3VLForConditionalGeneration, BaseLoadReceipt]:
    """Materialize the complete Qwen base directly on one DDP rank device."""

    inspection = inspect_qwen_checkpoint(qwen_root)
    config = Qwen3VLConfig.from_pretrained(str(qwen_root), local_files_only=True)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    state_dict = load_file(inspection.weight_path, device=str(device))
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    model.tie_weights()
    _materialize_qwen_rotary_buffers_(model, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    receipt = BaseLoadReceipt(
        family="qwen3vl_2b",
        device=str(device),
        dtype_counts=_dtype_counts(model),
        parameter_numel=sum(parameter.numel() for parameter in model.parameters()),
        buffer_numel=sum(buffer.numel() for buffer in model.buffers()),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        accepted_tied_keys=("lm_head.weight",),
        meta_tensor_names=_meta_tensor_names(model),
    )
    if receipt.parameter_numel != EXPECTED_QWEN_PARAMETER_COUNT or not receipt.passed:
        raise RuntimeError(f"Materialized Qwen base load failed: {receipt.to_dict()}.")
    return model, receipt


def load_ss_xl_base(
    ss_xl_stem: Path,
    *,
    device: torch.device,
) -> Tuple[SparseStructureFlowModel, BaseLoadReceipt]:
    """Materialize the complete TRELLIS1 SS XL base directly on one rank device."""

    inspection = inspect_ss_xl_checkpoint(ss_xl_stem)
    model = _build_ss_xl_meta_model(Path(inspection.config_path))
    state_dict = load_file(inspection.weight_path, device=str(device))
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    receipt = BaseLoadReceipt(
        family="trellis1_ss_xl",
        device=str(device),
        dtype_counts=_dtype_counts(model),
        parameter_numel=sum(parameter.numel() for parameter in model.parameters()),
        buffer_numel=sum(buffer.numel() for buffer in model.buffers()),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        accepted_tied_keys=(),
        meta_tensor_names=_meta_tensor_names(model),
    )
    if (
        receipt.parameter_numel != EXPECTED_SS_PARAMETER_COUNT
        or receipt.buffer_numel != EXPECTED_SS_BUFFER_COUNT
        or not receipt.passed
    ):
        raise RuntimeError(f"Materialized SS XL base load failed: {receipt.to_dict()}.")
    return model, receipt


def _materialize_slat_rope_constants_(model: SLatFlowModel, *, device: torch.device) -> None:
    for block in model.blocks:
        rope = getattr(block.self_attn, "rope", None)
        if rope is None:
            continue
        frequencies = torch.arange(rope.freq_dim, device=device, dtype=torch.float32) / rope.freq_dim
        rope.freqs = rope.rope_freq[0] / (rope.rope_freq[1] ** frequencies)


def load_slat_base(
    stem: Path,
    *,
    family: str,
    device: torch.device,
) -> Tuple[SLatFlowModel, BaseLoadReceipt]:
    """Strictly materialize one complete TRELLIS2 sparse Shape/Material base."""

    normalized_family = str(family).strip().lower()
    expected_counts = {
        "shape": EXPECTED_SHAPE_PARAMETER_COUNT,
        "material": EXPECTED_MATERIAL_PARAMETER_COUNT,
    }
    inspection = inspect_slat_checkpoint(stem, family=normalized_family)
    model = _build_slat_meta_model(Path(inspection.config_path))
    state_dict = load_file(inspection.weight_path, device=str(device))
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    _materialize_slat_rope_constants_(model, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    receipt = BaseLoadReceipt(
        family=f"trellis2_{normalized_family}_512",
        device=str(device),
        dtype_counts=_dtype_counts(model),
        parameter_numel=sum(parameter.numel() for parameter in model.parameters()),
        buffer_numel=sum(buffer.numel() for buffer in model.buffers()),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        accepted_tied_keys=(),
        meta_tensor_names=_meta_tensor_names(model),
    )
    if receipt.parameter_numel != expected_counts[normalized_family] or not receipt.passed:
        raise RuntimeError(
            f"Materialized TRELLIS2 {normalized_family} load failed: {receipt.to_dict()}."
        )
    return model, receipt


__all__ = [
    "BaseLoadReceipt",
    "CheckpointInspection",
    "EXPECTED_MATERIAL_PARAMETER_COUNT",
    "EXPECTED_QWEN_PARAMETER_COUNT",
    "EXPECTED_SHAPE_PARAMETER_COUNT",
    "EXPECTED_SS_BUFFER_COUNT",
    "EXPECTED_SS_PARAMETER_COUNT",
    "inspect_qwen_checkpoint",
    "inspect_slat_checkpoint",
    "inspect_ss_xl_checkpoint",
    "load_qwen_base",
    "load_slat_base",
    "load_ss_xl_base",
]
