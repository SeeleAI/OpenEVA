#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eva01_instruct.conversation import DEFAULT_QUESTION
from eva01_instruct.generation import OpenEVARuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-turn OpenEVA-Instruct mesh or point-cloud captioning.")
    parser.add_argument("--checkpoint", default="checkpoints/openeva-instruct-lora")
    parser.add_argument("--mesh", required=True, help="Input .glb/.gltf/.obj/.ply/.stl/.npy/.npz file.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None, choices=[None, "none", "auto"])
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = OpenEVARuntime(
        checkpoint_dir=args.checkpoint,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        pointnum=args.pointnum,
        max_length=args.max_length,
    )
    mesh_input = runtime.load_mesh_or_point_cloud(args.mesh, object_id=args.object_id, pointnum=args.pointnum)
    response, _ = runtime.answer(
        mesh_input=mesh_input,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    row = {
        "object_id": mesh_input.object_id,
        "input_path": str(mesh_input.input_path),
        "question": args.question,
        "prediction": response,
    }
    print(json.dumps(row, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
