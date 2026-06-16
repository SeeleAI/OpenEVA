#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eva01 import EVA01ForConditionalGeneration, EVA01Processor


DEFAULT_MODEL = "SEELE-AI/EVA01-2B-Instruct"
DEFAULT_QUESTION = "Describe this 3D object in detail."
DEFAULT_SEED = 20260615


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVA01 mesh understanding inference.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--base-model", default=None, help="Optional local Qwen3VL base path for LoRA checkpoints.")
    parser.add_argument("--mesh", required=True, help="Input GLB mesh.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device_map(value: str | None) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    processor = EVA01Processor.from_pretrained(args.checkpoint)
    model = EVA01ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        device_map=normalize_device_map(args.device_map),
        base_model_name_or_path=args.base_model,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "mesh", "mesh": args.mesh},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    set_seed(args.seed)
    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    row = {"mesh": str(Path(args.mesh).expanduser()), "question": args.question, "answer": text}
    print(json.dumps(row, ensure_ascii=False, indent=2))
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
