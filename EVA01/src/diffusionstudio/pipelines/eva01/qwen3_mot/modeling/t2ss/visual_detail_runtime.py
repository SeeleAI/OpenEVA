"""Detail-B intervention helpers shared by the unified runtime."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .detail_condition import HaarDetailSource, summarize_gate
from .runtime_types import PackedGenerationBatch


class VisualDetailRuntimeMixin:
    """Raw-detail preparation and slot-local condition injection."""

    @staticmethod
    def _transform_detail_by_sample(
        detail: torch.Tensor,
        sample_ids: torch.Tensor,
        *,
        intervention: str,
    ) -> torch.Tensor:
        unique_ids = torch.unique_consecutive(sample_ids)
        if intervention == "wrong-object":
            if unique_ids.numel() < 2:
                raise ValueError(
                    "wrong-object detail intervention requires at least two images."
                )
            transformed = torch.empty_like(detail)
            for index, sample_id in enumerate(unique_ids):
                source_id = unique_ids[(index + 1) % unique_ids.numel()]
                target = sample_ids.eq(sample_id).nonzero(as_tuple=False).flatten()
                source = sample_ids.eq(source_id).nonzero(as_tuple=False).flatten()
                if target.numel() != source.numel():
                    raise ValueError(
                        "wrong-object detail intervention requires aligned image grids."
                    )
                transformed[target] = detail[source]
            return transformed
        if intervention == "spatial-shuffle":
            transformed = detail.clone()
            for sample_id in unique_ids:
                selected = sample_ids.eq(sample_id).nonzero(as_tuple=False).flatten()
                transformed[selected] = detail[selected.flip(0)]
            return transformed
        raise ValueError(
            f"Unsupported sample-wise detail intervention `{intervention}`."
        )

    def _prepare_raw_detail_states(
        self,
        source: Optional[HaarDetailSource],
        *,
        intervention: str,
        selected_sample_ids: Optional[Tuple[int, ...]],
    ) -> Optional[torch.Tensor]:
        if source is None:
            if intervention != "correct" or selected_sample_ids:
                raise ValueError(
                    "A visual-detail intervention requires image-bearing raw features."
                )
            return None
        adapter = self.adapters.raw_detail_adapter
        if adapter is None:
            raise RuntimeError(
                "Detail-B source exists without a shared RawDetailAdapter."
            )
        correct = adapter(source.high_frequency)
        if intervention == "correct":
            transformed = correct
        elif intervention == "zero":
            transformed = torch.zeros_like(correct)
        elif intervention == "low-frequency-only":
            transformed = adapter(source.low_frequency_proxy)
        else:
            transformed = self._transform_detail_by_sample(
                correct,
                source.sample_ids,
                intervention=intervention,
            )
        if selected_sample_ids is None:
            return transformed
        selected = torch.zeros_like(source.sample_ids, dtype=torch.bool)
        for sample_id in selected_sample_ids:
            selected |= source.sample_ids.eq(int(sample_id))
        if not bool(selected.any().detach().cpu()):
            raise ValueError(
                "Detail intervention sample selection matches no visual token."
            )
        return torch.where(selected.unsqueeze(-1), transformed, correct)

    def _detail_condition_input(
        self,
        und_hidden: torch.Tensor,
        *,
        packed_batch: PackedGenerationBatch,
        raw_source: HaarDetailSource,
        raw_detail: torch.Tensor,
        timesteps: torch.Tensor,
        slot_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if packed_batch.visual_pos_mask is None:
            raise RuntimeError("Detail-B requires the packed visual position mask.")
        gate_module = self.adapters.detail_gate
        if gate_module is None:
            raise RuntimeError(
                "Detail-B source exists without a shared depth-time-token gate."
            )
        visual_mask = packed_batch.visual_pos_mask.to(
            device=und_hidden.device,
            dtype=torch.bool,
        )
        visual_hidden = und_hidden[visual_mask]
        if visual_hidden.shape != raw_detail.shape:
            raise RuntimeError(
                "Detail-B residual is not aligned with the current UND visual view."
            )
        visual_timesteps = timesteps[raw_source.sample_ids]
        gate = gate_module(
            visual_hidden,
            raw_detail,
            visual_timesteps,
            slot_index=slot_index,
        )
        residual = gate * raw_detail.to(dtype=gate.dtype)
        enhanced = und_hidden.clone()
        enhanced[visual_mask] = visual_hidden + residual.to(dtype=visual_hidden.dtype)
        summary = summarize_gate(
            gate,
            residual,
            visual_hidden,
            visual_timesteps,
            g_max=gate_module.g_max,
        )
        probe = getattr(self, "_eva01_visual_detail_probe", None)
        if probe is not None:
            probe(
                slot_index=int(slot_index),
                gate=gate.detach(),
                raw_detail=raw_detail.detach(),
                haar_energy=raw_source.energy.detach(),
                sample_ids=raw_source.sample_ids.detach(),
                timesteps=visual_timesteps.detach(),
            )
        return enhanced, summary


__all__ = ["VisualDetailRuntimeMixin"]
