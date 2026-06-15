#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from EVA01 import EVA01ForConditionalGeneration, EVA01Processor
from EVA01.metrics import (
    compute_caption_overlap_metrics,
    compute_gpt_average_score,
    compute_meteor_score,
    compute_sentence_bert_similarity,
    compute_simcse_similarity,
    save_caption_metric_bundle,
)
from EVA01.render_judge import score_manifest


FULL_REPO = "SEELE-AI/EVA01-2B-Instruct"
LORA_REPO = "SEELE-AI/EVA01-2B-Instruct-LoRA"
PROMPT = "Caption this 3D model in detail."
SEED = 20260615


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EVA01 on PointLLM-200.")
    parser.add_argument("--variant", choices=["full", "lora"], default="full")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default=None, help="Optional local Qwen3VL base path for LoRA checkpoints.")
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pointllm200"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--with-gpt", action="store_true")
    parser.add_argument("--gpt-model", default=None)
    parser.add_argument("--gpt-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
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


def resolve_benchmark(path: str | None) -> Path:
    if path:
        return Path(path).expanduser()
    snapshot = snapshot_download(FULL_REPO, allow_patterns=["benchmark/pointllm200/**"])
    return Path(snapshot) / "benchmark" / "pointllm200"


def load_annotations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in payload:
        conversations = item.get("conversations") or []
        gt = ""
        for turn in conversations:
            if turn.get("from") == "gpt":
                gt = str(turn.get("value") or "")
                break
        rows.append({"object_id": str(item["object_id"]), "ground_truth": gt})
    return rows


def build_render_manifest(rows: list[dict[str, Any]], render_dir: Path, prediction_by_id: dict[str, str]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        object_id = row["object_id"]
        base = render_dir / object_id
        renders = {view: str(base / f"{view}.png") for view in ("front", "right", "back", "left")}
        if all(Path(path).exists() for path in renders.values()):
            items.append({"object_id": object_id, "caption": prediction_by_id[object_id], "renders": renders})
    return items


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = args.checkpoint or (LORA_REPO if args.variant == "lora" else FULL_REPO)
    benchmark = resolve_benchmark(args.benchmark_dir)
    output_dir = Path(args.output_dir) / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = benchmark / "PointLLM_brief_description_val_200_GT.json"
    pointcloud_dir = benchmark / "8192_npy"
    rows = load_annotations(annotation_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    processor = EVA01Processor.from_pretrained(checkpoint)
    model = EVA01ForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        device_map=normalize_device_map(args.device_map),
        base_model_name_or_path=args.base_model,
    )

    predictions: list[dict[str, str]] = []
    for row in tqdm(rows, desc=f"EVA01 {args.variant}"):
        set_seed(args.seed)
        npy_path = pointcloud_dir / f"{row['object_id']}_8192.npy"
        messages = [{"role": "user", "content": [{"type": "mesh", "mesh": str(npy_path)}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        predictions.append({"object_id": row["object_id"], "ground_truth": row["ground_truth"], "model_output": text})

    predictions_json = output_dir / "objaverse_captioning_prompt2_predictions.json"
    predictions_json.write_text(json.dumps({"prompt": PROMPT, "results": predictions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle = compute_caption_overlap_metrics(predictions)
    bundle = compute_meteor_score(bundle)
    bundle = compute_sentence_bert_similarity(bundle)
    bundle = compute_simcse_similarity(bundle, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu")
    bundle.metadata = {
        "variant": args.variant,
        "checkpoint": checkpoint,
        "benchmark_dir": str(benchmark),
        "seed": args.seed,
        "prompt": PROMPT,
        "num_samples": len(predictions),
    }
    if args.with_gpt:
        if args.gpt_model:
            bundle = compute_gpt_average_score(bundle, model_name=args.gpt_model, workers=args.gpt_workers)
        render_dir = benchmark / "renders"
        prediction_by_id = {row["object_id"]: row["model_output"] for row in predictions}
        manifest = build_render_manifest(predictions, render_dir, prediction_by_id)
        manifest_path = output_dir / "gpt_img_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.gpt_model and manifest:
            judge_payload = score_manifest(manifest_path, model_name=args.gpt_model, workers=args.gpt_workers)
            (output_dir / "gpt_img_scores.json").write_text(json.dumps(judge_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics_json = output_dir / "objaverse_captioning_prompt2_metrics.json"
    save_caption_metric_bundle(bundle, metrics_json)
    print(json.dumps({"predictions": str(predictions_json), "metrics": str(metrics_json), "scores": bundle.overall_scores}, indent=2))


if __name__ == "__main__":
    main()
