"""Public text-to-3D and image-to-3D runtime for EVA01 Unified MoT.

The public surface intentionally exposes only the generation cascade.  It
uses the trained Unified All-Task model, the native family representations,
and the formal direct-FlexAttention sampler.  Mesh editing is not part of
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer

from diffusionstudio.pipelines.eva01.qwen3_mot.runtime.normalization import (
    Trellis2LatentNormalization,
    load_trellis2_latent_normalization,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.runtime.generation import (
    AllTaskCascadeOutput,
    prepare_generation_condition,
    sample_generation_cascade,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.all_task import (
    AllTaskUnifiedMoT,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.all_task_adapters import (
    ALL_TASK_SCHEMA,
    GENERATION_BRANCH_SCHEMA,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.all_task_tokens import (
    validate_all_task_tokenizer,
)


MAX_UNIFIED_TOKENS = 24_576


@dataclass(frozen=True)
class GenerationResult:
    """Native outputs of one Unified MoT generation cascade."""

    mode: str
    ss: torch.Tensor
    coords: torch.Tensor
    shape: Any
    material: Any
    traces: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ss": self.ss,
            "coords": self.coords,
            "shape": self.shape,
            "material": self.material,
            "traces": dict(self.traces),
        }


def _adapter_paths(checkpoint: str | Path) -> tuple[Path, Path]:
    root = Path(checkpoint).expanduser().resolve()
    if root.is_file():
        adapter_path = root
    else:
        candidates = (root / "adapter_model.safetensors", root / "lora_model.safetensors")
        existing = tuple(path for path in candidates if path.is_file())
        if len(existing) > 1:
            raise RuntimeError(
                "A checkpoint directory cannot contain both adapter_model.safetensors "
                "and lora_model.safetensors."
            )
        adapter_path = existing[0] if existing else candidates[0]
    config_path = adapter_path.with_name("adapter_config.json")
    if not adapter_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "A Unified MoT checkpoint must contain adapter_config.json and "
            f"adapter_model.safetensors or lora_model.safetensors: {adapter_path}"
        )
    return adapter_path, config_path


def _load_generation_adapter(model: AllTaskUnifiedMoT, checkpoint: str | Path) -> dict[str, Any]:
    """Load one complete generation-only adapter export."""

    adapter_path, config_path = _adapter_paths(checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architecture") != ALL_TASK_SCHEMA:
        raise RuntimeError("Checkpoint is not an EVA01 Unified All-Task adapter.")
    if config.get("branch_schema") != GENERATION_BRANCH_SCHEMA:
        raise RuntimeError("Unified generation requires branch_schema=generation_v1.")
    if config.get("generation_only") is not True:
        raise RuntimeError("Checkpoint must declare generation_only=true.")
    checkpoint_format = config.get("format")
    expected_filename = {
        "adapter": "adapter_model.safetensors",
        "lora": "lora_model.safetensors",
    }.get(checkpoint_format)
    if expected_filename is None or adapter_path.name != expected_filename:
        raise RuntimeError("Checkpoint format does not match its weight filename.")
    if int(config.get("gen_rank", -1)) != 64:
        raise RuntimeError("Unified generation requires gen_rank=64.")
    if config.get("active_families") != ["ss", "shape", "material"]:
        raise RuntimeError("Checkpoint must contain the SS, Shape, and Material families.")
    if config.get("cascade") != ["ss", "shape", "material"]:
        raise RuntimeError("Checkpoint has the wrong generation cascade.")
    if config.get("frozen_base_included") is not False or config.get("ema") is not False:
        raise RuntimeError("Generation exports must be adapter-only and non-EMA.")
    if config.get("requires_flex_attention") is not True:
        raise RuntimeError("Checkpoint must require direct FlexAttention.")

    state = load_file(str(adapter_path), device="cpu")
    expected = model.adapters.state_dict()
    unknown = sorted(set(state) - set(expected))
    if unknown:
        raise RuntimeError(f"Checkpoint contains unknown adapter keys: {unknown[:3]}")
    missing = sorted(set(expected) - set(state))
    if missing:
        raise RuntimeError(f"Generation checkpoint is incomplete: {missing[:3]}")
    model.adapters.load_adapter_state_dict(state)
    return {
        "architecture": ALL_TASK_SCHEMA,
        "branch_schema": GENERATION_BRANCH_SCHEMA,
        "generation_only": True,
        "adapter_path": str(adapter_path),
        "adapter_format": str(checkpoint_format),
        "loaded_keys": len(state),
    }


class EVA01UnifiedGeneration:
    """Minimal correct public runtime for text-to-3D and image-to-3D."""

    def __init__(
        self,
        *,
        model: AllTaskUnifiedMoT,
        tokenizer: Any,
        processor: Any,
        normalization: Trellis2LatentNormalization,
        adapter_receipt: Mapping[str, Any],
        device: torch.device,
        steps: int = 25,
        cfg_scale: float = 3.0,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("Unified MoT generation requires CUDA FlexAttention.")
        if int(steps) <= 0:
            raise ValueError("steps must be positive.")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.normalization = normalization
        self.adapter_receipt = dict(adapter_receipt)
        self.device = device
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        tokenizer_root: str | Path,
        processor_root: str | Path | None = None,
        qwen_root: str | Path,
        ss_xl_stem: str | Path,
        shape_stem: str | Path,
        material_stem: str | Path,
        normalization_config: str | Path,
        device: str | torch.device = "cuda",
        compile_flex_attention: bool = True,
        steps: int = 25,
        cfg_scale: float = 3.0,
    ) -> "EVA01UnifiedGeneration":
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise RuntimeError("Unified MoT generation requires a CUDA device.")
        model, _ = AllTaskUnifiedMoT.from_pretrained(
            device=resolved_device,
            branch_schema=GENERATION_BRANCH_SCHEMA,
            gen_rank=64,
            compile_flex_attention=bool(compile_flex_attention),
            qwen_root=Path(qwen_root),
            ss_xl_stem=Path(ss_xl_stem),
            shape_stem=Path(shape_stem),
            material_stem=Path(material_stem),
            activation_checkpointing=False,
            parallel_native_branches=True,
        )
        adapter_receipt = _load_generation_adapter(model, checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_root), use_fast=True, local_files_only=True
        )
        validate_all_task_tokenizer(tokenizer)
        resolved_processor = AutoProcessor.from_pretrained(
            str(processor_root or qwen_root), local_files_only=True
        )
        normalization = load_trellis2_latent_normalization(normalization_config)
        return cls(
            model=model,
            tokenizer=tokenizer,
            processor=resolved_processor,
            normalization=normalization,
            adapter_receipt=adapter_receipt,
            device=resolved_device,
            steps=steps,
            cfg_scale=cfg_scale,
        )

    def _condition(self, *, mode: str, prompt: str, image: str | Path | None) -> Any:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"text", "image"}:
            raise ValueError("mode must be `text` or `image`.")
        if normalized_mode == "image" and image is None:
            raise ValueError("image mode requires an image path.")
        if normalized_mode == "text" and image is not None:
            raise ValueError("text mode cannot receive an image path.")
        prepared = prepare_generation_condition(
            self.tokenizer,
            self.processor,
            prompt=str(prompt),
            image=image,
            max_unified_tokens=MAX_UNIFIED_TOKENS,
        )
        return prepared

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str = "",
        image: str | Path | None = None,
        ss_decoder: torch.nn.Module,
        seed: int = 0,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> GenerationResult:
        mode = "image" if image is not None else "text"
        prepared = self._condition(mode=mode, prompt=prompt, image=image)
        result: AllTaskCascadeOutput = sample_generation_cascade(
            self.model,
            self.tokenizer,
            prepared,
            self.normalization,
            ss_decoder,
            seed=int(seed),
            steps=self.steps if steps is None else int(steps),
            cfg_scale=self.cfg_scale if cfg_scale is None else float(cfg_scale),
        )
        return GenerationResult(
            mode=mode,
            ss=result.ss,
            coords=result.coords,
            shape=result.shape,
            material=result.material,
            traces=result.traces,
        )


__all__ = ["EVA01UnifiedGeneration", "GenerationResult", "MAX_UNIFIED_TOKENS"]
