#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eva01_instruct.conversation import DEFAULT_QUESTION
from eva01_instruct.generation import MeshInput, OpenEVARuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with OpenEVA-Instruct using one mesh or point cloud.")
    parser.add_argument("--checkpoint", default="checkpoints/openeva-instruct-lora")
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None, choices=[None, "none", "auto"])
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-history-turns", type=int, default=4)
    return parser.parse_args()


def print_help() -> None:
    print("Commands: /load PATH, /reset, /history, /save PATH, /exit")


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
    mesh_input: MeshInput | None = None
    history: List[Dict[str, str]] = []
    transcript: List[Dict[str, Any]] = []

    def load_mesh(path: str) -> None:
        nonlocal mesh_input, history
        mesh_input = runtime.load_mesh_or_point_cloud(path, pointnum=args.pointnum)
        history = []
        print(f"Loaded {mesh_input.input_path} as object_id={mesh_input.object_id}; history reset.")

    if args.mesh:
        load_mesh(args.mesh)
    else:
        print("No mesh loaded yet.")
    print_help()

    while True:
        try:
            text = input("user> ").strip()
        except EOFError:
            print()
            break
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text.startswith("/load "):
            load_mesh(text[len("/load ") :].strip())
            continue
        if text == "/reset":
            history = []
            print("History reset.")
            continue
        if text == "/history":
            print(json.dumps(history, ensure_ascii=False, indent=2))
            continue
        if text.startswith("/save "):
            output_path = Path(text[len("/save ") :].strip())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved transcript to {output_path}")
            continue
        if text == "/help":
            print_help()
            continue
        if mesh_input is None:
            print("Load a mesh first with /load PATH.")
            continue

        question = text or DEFAULT_QUESTION
        response, history = runtime.answer(
            mesh_input=mesh_input,
            question=question,
            history=history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_history_turns=args.max_history_turns,
        )
        transcript.append(
            {
                "object_id": mesh_input.object_id,
                "input_path": str(mesh_input.input_path),
                "question": question,
                "prediction": response,
            }
        )
        print(f"assistant> {response}")


if __name__ == "__main__":
    main()
