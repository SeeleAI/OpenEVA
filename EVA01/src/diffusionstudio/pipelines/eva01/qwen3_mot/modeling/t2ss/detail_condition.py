"""Token-preserving raw visual-detail conditioning for EVA01 Detail-B."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


VISUAL_DETAIL_MODES = ("none", "haar_gated")
VISUAL_DETAIL_TAPS: Mapping[str, int] = {"block2": 2, "block5": 5}
DETAIL_INTERVENTIONS = (
    "correct",
    "zero",
    "wrong-object",
    "spatial-shuffle",
    "low-frequency-only",
)
RAW_VISION_DIM = 1024
RAW_HAAR_DIM = 3 * RAW_VISION_DIM
UND_HIDDEN_DIM = 2048
RAW_DETAIL_SCHEMA_VERSION = 2
RAW_DETAIL_PROJECTION = "linear_pca_3072x2048"


def _rms_norm(states: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(states, (states.shape[-1],), eps=1.0e-6)


def _linear(module: nn.Linear, states: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        return module(states)
    input_dtype = states.dtype
    return module(states.to(dtype=module.weight.dtype)).to(dtype=input_dtype)


@dataclass(frozen=True)
class HaarDetailSource:
    """One raw-detail token per already-existing merged visual token."""

    high_frequency: torch.Tensor
    low_frequency_proxy: torch.Tensor
    sample_ids: torch.Tensor
    energy: torch.Tensor

    def validate(self) -> None:
        token_count = int(self.high_frequency.shape[0])
        if self.high_frequency.shape != (token_count, RAW_HAAR_DIM):
            raise ValueError("Haar high-frequency source must have shape [T_visual,3072].")
        if self.low_frequency_proxy.shape != self.high_frequency.shape:
            raise ValueError("Low-frequency proxy must match the Haar source shape.")
        if self.sample_ids.shape != (token_count,) or self.energy.shape != (token_count,):
            raise ValueError("Raw-detail sample IDs and energy must contain one value per token.")


def grouped_haar_2x2(premerge_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an orthonormal 2x2 Haar transform without creating condition tokens."""

    if premerge_states.ndim != 2 or premerge_states.shape[-1] != RAW_VISION_DIM:
        raise ValueError("Pre-merge vision features must have shape [4*T_visual,1024].")
    if premerge_states.shape[0] % 4:
        raise ValueError("Pre-merge vision token count must be divisible by the 2x2 merge unit.")
    a, b, c, d = premerge_states.reshape(-1, 4, RAW_VISION_DIM).unbind(dim=1)
    scale = 0.5
    low = (a + b + c + d) * scale
    horizontal = (a + b - c - d) * scale
    vertical = (a - b + c - d) * scale
    diagonal = (a - b - c + d) * scale
    high = torch.cat((horizontal, vertical, diagonal), dim=-1)
    low_proxy = torch.cat((low, low, low), dim=-1)
    energy = high.float().square().mean(dim=-1).sqrt().to(dtype=premerge_states.dtype)
    return high, low_proxy, energy


def raw_detail_weight_sha256(weight: torch.Tensor) -> str:
    """Hash one projection tensor independently of a checkpoint container header."""

    value = weight.detach().cpu().contiguous()
    digest = hashlib.sha256()
    dtype = str(value.dtype).encode("ascii")
    digest.update(struct.pack("<Q", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack("<Q", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("<q", int(dimension)))
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


class RawDetailAdapter(nn.Module):
    """Shared bias-free PCA projection for every image-bearing task.

    The seed-based signed-coordinate basis is only a deterministic,
    semi-orthogonal construction default.  A trainable Detail-B checkpoint is
    created by the migration workflow, which replaces this weight with the
    locked Gate-1 FP64-PCA initialization and records its tensor SHA256.
    """

    def __init__(self, *, seed: int = 20260731) -> None:
        super().__init__()
        self.projection = nn.Linear(RAW_HAAR_DIM, UND_HIDDEN_DIM, bias=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        columns = torch.randperm(RAW_HAAR_DIM, generator=generator)[:UND_HIDDEN_DIM]
        signs = torch.randint(0, 2, (UND_HIDDEN_DIM,), generator=generator)
        signs = signs.mul(2).sub(1).to(dtype=self.projection.weight.dtype)
        with torch.no_grad():
            self.projection.weight.zero_()
            rows = torch.arange(UND_HIDDEN_DIM)
            self.projection.weight[rows, columns] = signs

    def forward(self, haar_states: torch.Tensor) -> torch.Tensor:
        if haar_states.ndim != 2 or haar_states.shape[-1] != RAW_HAAR_DIM:
            raise ValueError("RawDetailAdapter input must have shape [T_visual,3072].")
        return _linear(self.projection, haar_states)

    @property
    def is_semi_orthogonal(self) -> bool:
        weight = self.projection.weight.detach().double()
        identity = torch.eye(weight.shape[0], dtype=weight.dtype, device=weight.device)
        return bool(torch.equal(weight @ weight.transpose(0, 1), identity))


class DepthTimeTokenGate(nn.Module):
    """Shared token gate conditioned on UND state, detail, flow time, and slot depth."""

    NUM_DEPTHS = 28
    TIME_FOURIER_DIM = 64

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        g_max: float = 0.25,
        seed: int = 20260731,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or not 0.0 < float(g_max) <= 1.0:
            raise ValueError("Detail gate hidden_dim/g_max are invalid.")
        self.hidden_dim = int(hidden_dim)
        self.g_max = float(g_max)
        self.hidden_projection = nn.Linear(UND_HIDDEN_DIM, self.hidden_dim, bias=False)
        self.detail_projection = nn.Linear(UND_HIDDEN_DIM, self.hidden_dim, bias=False)
        self.time_projection = nn.Linear(self.TIME_FOURIER_DIM, self.hidden_dim, bias=False)
        self.depth_embedding = nn.Embedding(self.NUM_DEPTHS, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, 1, bias=True)
        frequencies = torch.logspace(0.0, 3.0, self.TIME_FOURIER_DIM // 2)
        self.register_buffer("time_frequencies", frequencies, persistent=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        for projection in (
            self.hidden_projection,
            self.detail_projection,
            self.time_projection,
        ):
            nn.init.xavier_uniform_(projection.weight, generator=generator)
        nn.init.normal_(self.depth_embedding.weight, mean=0.0, std=0.02, generator=generator)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _time_features(self, timesteps: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        phases = (
            timesteps.float().unsqueeze(-1)
            * self.time_frequencies.float().unsqueeze(0)
            * (2.0 * math.pi)
        )
        return torch.cat((phases.sin(), phases.cos()), dim=-1).to(dtype=dtype)

    def forward(
        self,
        visual_hidden: torch.Tensor,
        raw_detail: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        slot_index: int,
    ) -> torch.Tensor:
        token_count = int(visual_hidden.shape[0])
        if visual_hidden.shape != (token_count, UND_HIDDEN_DIM):
            raise ValueError("Visual UND states must have shape [T_visual,2048].")
        if raw_detail.shape != visual_hidden.shape or timesteps.shape != (token_count,):
            raise ValueError("Detail gate inputs are not token-aligned.")
        depth = min(max(int(slot_index), 0), self.NUM_DEPTHS - 1)
        hidden = _linear(self.hidden_projection, _rms_norm(visual_hidden))
        hidden = hidden + _linear(self.detail_projection, _rms_norm(raw_detail))
        hidden = hidden + _linear(
            self.time_projection,
            self._time_features(timesteps, dtype=visual_hidden.dtype),
        )
        hidden = hidden + self.depth_embedding.weight[depth].to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        logits = _linear(self.output_projection, F.silu(hidden))
        return self.g_max * torch.tanh(logits)

    @property
    def is_zero_initialized(self) -> bool:
        return bool(
            torch.count_nonzero(self.output_projection.weight).item() == 0
            and torch.count_nonzero(self.output_projection.bias).item() == 0
        )


def summarize_gate(
    gate: torch.Tensor,
    residual: torch.Tensor,
    visual_hidden: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    g_max: float,
) -> torch.Tensor:
    """Return fixed-shape GPU summaries; callers perform one host transfer per forward."""

    values = gate.detach().float().flatten()
    residual_ratio = (
        residual.detach().float().norm(dim=-1)
        / visual_hidden.detach().float().norm(dim=-1).clamp_min(1.0e-8)
    ).mean()
    summaries = [
        values.mean(),
        values.var(unbiased=False),
        values.abs().mean(),
        values.abs().ge(float(g_max) * 0.95).float().mean(),
        residual_ratio,
    ]
    time_values = timesteps.detach().float()
    for lower in (0.0, 0.25, 0.5, 0.75):
        selected = time_values.ge(lower) & time_values.lt(lower + 0.25 + 1.0e-7)
        weights = selected.float()
        count = weights.sum()
        denominator = count.clamp_min(1.0)
        mean = (values * weights).sum() / denominator
        variance = ((values - mean).square() * weights).sum() / denominator
        absolute = (values.abs() * weights).sum() / denominator
        saturation = (
            values.abs().ge(float(g_max) * 0.95).float() * weights
        ).sum() / denominator
        valid = count.gt(0).float()
        summaries.extend((mean * valid, variance * valid, absolute * valid, saturation * valid, count))
    return torch.stack(summaries)


__all__ = [
    "DETAIL_INTERVENTIONS",
    "DepthTimeTokenGate",
    "HaarDetailSource",
    "RAW_DETAIL_PROJECTION",
    "RAW_DETAIL_SCHEMA_VERSION",
    "RawDetailAdapter",
    "VISUAL_DETAIL_MODES",
    "VISUAL_DETAIL_TAPS",
    "grouped_haar_2x2",
    "raw_detail_weight_sha256",
    "summarize_gate",
]
