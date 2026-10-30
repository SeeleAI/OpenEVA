#!/usr/bin/env python3
"""Export the generation branch of a trained Unified All-Task checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.all_task_adapters import (
    ALL_TASK_SCHEMA,
    GENERATION_BRANCH_SCHEMA,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.all_task_positions import (
    VIRTUAL_POSITION_ABI,
)
from diffusionstudio.pipelines.eva01.qwen3_mot.modeling.t2ss.all_task_tokens import (
    SPECIAL_EMBEDDING_ABI,
)

SOURCE_BRANCH_SCHEMA = "gen_edit_routed_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the unreleased branch and write a generation-only Unified MoT checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("adapter", "lora"), required=True)
    return parser.parse_args()


def export_generation_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    checkpoint_format: str,
) -> dict[str, object]:
    if checkpoint_format not in {"adapter", "lora"}:
        raise ValueError("checkpoint_format must be `adapter` or `lora`.")
    root = checkpoint.expanduser().resolve()
    source_weights = root if root.is_file() else root / "adapter_model.safetensors"
    source_config = source_weights.with_name("adapter_config.json")
    if not source_weights.is_file() or not source_config.is_file():
        raise FileNotFoundError("The source checkpoint is missing adapter weights or config.")
    config = json.loads(source_config.read_text(encoding="utf-8"))
    if config.get("architecture") != ALL_TASK_SCHEMA:
        raise RuntimeError("The source checkpoint has the wrong architecture.")
    if config.get("branch_schema") != SOURCE_BRANCH_SCHEMA:
        raise RuntimeError("The source checkpoint is not the final routed Unified MoT schema.")

    source = load_file(str(source_weights), device="cpu")
    exported = {}
    dropped = 0
    for name, tensor in source.items():
        if ".edit." in name:
            dropped += 1
            continue
        public_name = name.replace(".gen.", ".")
        if public_name in exported:
            raise RuntimeError(f"Generation key collision after export: {public_name}")
        exported[public_name] = tensor.contiguous()
    if dropped == 0 or any(".gen." in name or ".edit." in name for name in exported):
        raise RuntimeError("Generation branch filtering did not match the routed checkpoint ABI.")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    filename = "adapter_model.safetensors" if checkpoint_format == "adapter" else "lora_model.safetensors"
    conflicts = [
        path
        for path in (
            output / "adapter_model.safetensors",
            output / "lora_model.safetensors",
            output / "adapter_config.json",
        )
        if path.exists()
    ]
    if conflicts:
        raise FileExistsError(
            "Generation export output already contains checkpoint files: "
            + ", ".join(path.name for path in conflicts)
        )
    save_file(exported, str(output / filename))
    public_config = {
        "schema_version": 1,
        "architecture": ALL_TASK_SCHEMA,
        "branch_schema": GENERATION_BRANCH_SCHEMA,
        "generation_only": True,
        "format": checkpoint_format,
        "gen_rank": int(config.get("gen_rank", 64)),
        "active_families": ["ss", "shape", "material"],
        "cascade": ["ss", "shape", "material"],
        "frozen_base_included": False,
        "ema": False,
        "special_embedding_abi": SPECIAL_EMBEDDING_ABI,
        "virtual_position_abi": VIRTUAL_POSITION_ABI,
        "requires_flex_attention": True,
    }
    (output / "adapter_config.json").write_text(
        json.dumps(public_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "format": checkpoint_format,
        "exported_tensor_count": len(exported),
        "removed_tensor_count": dropped,
    }


def main() -> None:
    args = parse_args()
    receipt = export_generation_checkpoint(
        args.checkpoint,
        args.output,
        checkpoint_format=args.format,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
