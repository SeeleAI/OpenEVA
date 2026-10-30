"""Pinned TRELLIS2 Shape/Material latent normalization for all-task V1."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch


def _vector(values: Sequence[Any], name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 32:
        raise ValueError(f"{name} must contain exactly 32 channels.")
    if not torch.isfinite(torch.tensor(result, dtype=torch.float64)).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


@dataclass(frozen=True)
class Trellis2LatentNormalization:
    """The exact native numeric ABI used by TRELLIS2 512 Flow checkpoints."""

    config_path: str
    shape_mean: Tuple[float, ...]
    shape_std: Tuple[float, ...]
    material_mean: Tuple[float, ...]
    material_std: Tuple[float, ...]

    @staticmethod
    def _transform(
        values: torch.Tensor,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        if values.ndim < 1 or int(values.shape[-1]) != 32:
            raise ValueError(
                f"TRELLIS2 latent features must end in 32 channels, got {tuple(values.shape)}."
            )
        source = values.float()
        mean_tensor = torch.tensor(mean, device=source.device, dtype=torch.float32)
        std_tensor = torch.tensor(std, device=source.device, dtype=torch.float32)
        return (
            source * std_tensor + mean_tensor
            if inverse
            else (source - mean_tensor) / std_tensor
        )

    def normalize_shape(self, values: torch.Tensor) -> torch.Tensor:
        return self._transform(
            values, self.shape_mean, self.shape_std, inverse=False
        )

    def denormalize_shape(self, values: torch.Tensor) -> torch.Tensor:
        return self._transform(
            values, self.shape_mean, self.shape_std, inverse=True
        )

    def normalize_material(self, values: torch.Tensor) -> torch.Tensor:
        return self._transform(
            values, self.material_mean, self.material_std, inverse=False
        )

    def denormalize_material(self, values: torch.Tensor) -> torch.Tensor:
        return self._transform(
            values, self.material_mean, self.material_std, inverse=True
        )

    def receipt(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "passed": True,
            "source": "trellis2_pipeline_native_latent_normalization",
            "config_path": self.config_path,
            "shape_channels": len(self.shape_mean),
            "material_channels": len(self.material_mean),
            "shape_key": "shape_slat_normalization",
            "material_key": "tex_slat_normalization",
            "applied_before_flow_noise": True,
            "material_shape_half_uses_shape_normalization": True,
        }


@lru_cache(maxsize=8)
def load_trellis2_latent_normalization(
    config_path: str | Path,
) -> Trellis2LatentNormalization:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("name") != "Trellis2ImageTo3DPipeline":
        raise ValueError("TRELLIS2 normalization source has the wrong pipeline schema.")
    arguments = payload.get("args", {})
    shape = arguments.get("shape_slat_normalization", {})
    material = arguments.get("tex_slat_normalization", {})
    shape_mean = _vector(shape.get("mean", ()), "shape mean")
    shape_std = _vector(shape.get("std", ()), "shape std")
    material_mean = _vector(material.get("mean", ()), "material mean")
    material_std = _vector(material.get("std", ()), "material std")
    if min(shape_std) <= 0.0 or min(material_std) <= 0.0:
        raise ValueError("TRELLIS2 latent normalization std must be positive.")
    return Trellis2LatentNormalization(
        config_path=str(path),
        shape_mean=shape_mean,
        shape_std=shape_std,
        material_mean=material_mean,
        material_std=material_std,
    )


__all__ = [
    "Trellis2LatentNormalization",
    "load_trellis2_latent_normalization",
]
