from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from EVA01.tools.export_generation_checkpoint import export_generation_checkpoint
from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.all_task_adapters import (
    ALL_TASK_SCHEMA,
    GENERATION_BRANCH_SCHEMA,
    AllTaskAdapterRegistry,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.adapters import (
    GenerationRouteCompatibleLoRA,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.runtime import generation
from eva01.unified_generation import _adapter_paths, _load_generation_adapter


def _public_config(checkpoint_format: str) -> dict[str, object]:
    return {
        "architecture": ALL_TASK_SCHEMA,
        "branch_schema": GENERATION_BRANCH_SCHEMA,
        "generation_only": True,
        "format": checkpoint_format,
        "gen_rank": 64,
        "active_families": ["ss", "shape", "material"],
        "cascade": ["ss", "shape", "material"],
        "frozen_base_included": False,
        "ema": False,
        "requires_flex_attention": True,
    }


def test_generation_registry_has_exact_public_inventory() -> None:
    registry = AllTaskAdapterRegistry(branch_schema=GENERATION_BRANCH_SCHEMA)
    state = registry.state_dict()

    assert len(state) == 2_065
    assert sum(tensor.numel() for tensor in state.values()) == 154_782_976
    assert not any(".gen." in key or ".edit." in key for key in state)


def test_generation_lora_preserves_routed_projection_compute_shape() -> None:
    delta = GenerationRouteCompatibleLoRA(8, 12, rank=4, alpha=4, seed=7)
    with torch.no_grad():
        delta.lora_b.weight.copy_(torch.randn_like(delta.lora_b.weight))
    hidden = torch.randn(5, 8)

    padded_a = torch.cat(
        (delta.lora_a.weight, torch.zeros_like(delta.lora_a.weight)), dim=0
    )
    low_rank = torch.nn.functional.linear(hidden, padded_a)
    generation_low, inactive_low = low_rank.split((delta.rank, delta.rank), dim=-1)
    selected_low = torch.cat(
        (generation_low * 1.0, inactive_low * 0.0), dim=-1
    )
    padded_b = torch.cat(
        (delta.lora_b.weight * delta.scaling, torch.zeros_like(delta.lora_b.weight)),
        dim=1,
    )
    expected = torch.nn.functional.linear(selected_low, padded_b)

    assert torch.equal(delta(hidden), expected)
    assert set(delta.state_dict()) == {"lora_a.weight", "lora_b.weight"}


@pytest.mark.parametrize(
    ("checkpoint_format", "filename"),
    (("adapter", "adapter_model.safetensors"), ("lora", "lora_model.safetensors")),
)
def test_export_removes_unreleased_branch_and_uses_requested_filename(
    tmp_path: Path,
    checkpoint_format: str,
    filename: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {
            "families.ss.0.q.gen.lora_a.weight": torch.ones(2, 2),
            "families.ss.0.q.edit.lora_a.weight": torch.zeros(2, 2),
        },
        str(source / "adapter_model.safetensors"),
    )
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "architecture": ALL_TASK_SCHEMA,
                "branch_schema": "gen_edit_routed_v1",
                "gen_rank": 64,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / checkpoint_format
    receipt = export_generation_checkpoint(
        source,
        output,
        checkpoint_format=checkpoint_format,
    )

    weights, config_path = _adapter_paths(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert weights.name == filename
    assert config["format"] == checkpoint_format
    assert config["branch_schema"] == GENERATION_BRANCH_SCHEMA
    assert receipt["exported_tensor_count"] == 1
    assert receipt["removed_tensor_count"] == 1
    assert not any("sha256" in key.lower() for key in config)


def test_public_loader_rejects_incomplete_generation_checkpoint(tmp_path: Path) -> None:
    save_file({"required": torch.ones(1)}, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(_public_config("adapter")), encoding="utf-8"
    )

    class Adapters:
        def state_dict(self) -> dict[str, torch.Tensor]:
            return {"required": torch.empty(1), "missing": torch.empty(1)}

        def load_adapter_state_dict(self, state: object) -> None:
            raise AssertionError("An incomplete checkpoint must not be loaded.")

    class Model:
        adapters = Adapters()

    with pytest.raises(RuntimeError, match="incomplete"):
        _load_generation_adapter(Model(), tmp_path)  # type: ignore[arg-type]


def test_public_loader_rejects_internal_routed_checkpoint(tmp_path: Path) -> None:
    save_file({"value": torch.ones(1)}, str(tmp_path / "adapter_model.safetensors"))
    config = _public_config("adapter")
    config["branch_schema"] = "gen_edit_routed_v1"
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="generation_v1"):
        _load_generation_adapter(object(), tmp_path)  # type: ignore[arg-type]


def test_checkpoint_directory_cannot_mix_adapter_and_lora_files(tmp_path: Path) -> None:
    save_file({"value": torch.ones(1)}, str(tmp_path / "adapter_model.safetensors"))
    save_file({"value": torch.ones(1)}, str(tmp_path / "lora_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="both"):
        _adapter_paths(tmp_path)


def test_cascade_rejects_material_coordinate_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    coords = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    trace = generation.EulerTrace(
        family="test",
        steps=1,
        cfg_scale=3.0,
        flex_calls_first_conditional=30,
        flex_calls_first_null=30,
        path_length=0.0,
        straight_distance=0.0,
        path_over_straight=0.0,
        endpoint_velocity_norm=0.0,
        condition_delta_norm=0.0,
        finite=True,
    )

    def sample(*args: object, **kwargs: object):
        family = args[3]
        initial = args[4]
        if family == "ss":
            return initial, trace
        if family == "shape":
            return initial, trace
        drifted = coords.clone()
        drifted[0, 1] += 1
        return generation._sparse(drifted, initial.feats), trace

    monkeypatch.setattr(generation, "sample_family_euler_cfg", sample)
    monkeypatch.setattr(
        generation,
        "decode_ss_to_shape_coords",
        lambda *args, **kwargs: coords,
    )

    class Normalization:
        @staticmethod
        def denormalize_shape(values: torch.Tensor) -> torch.Tensor:
            return values

        @staticmethod
        def denormalize_material(values: torch.Tensor) -> torch.Tensor:
            return values

    with pytest.raises(RuntimeError, match="Material coordinates"):
        generation.sample_generation_cascade(
            model, object(), object(), Normalization(), object(), seed=0, steps=1
        )
