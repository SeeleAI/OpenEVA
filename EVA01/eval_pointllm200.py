#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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

from eva01 import EVA01ForConditionalGeneration, EVA01Processor
from eva01.metrics import (
    compute_caption_overlap_metrics,
    compute_gpt_average_score,
    compute_meteor_score,
    compute_sentence_bert_similarity,
    compute_simcse_similarity,
    save_caption_metric_bundle,
)
from eva01.render_judge import score_manifest


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
    parser.add_argument("--render-dir", default=None, help="Directory with Blender PBR GLB renders for GPT-img. Defaults to <benchmark-dir>/renders_blender_pbr.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pointllm200"))
    parser.add_argument("--predictions-json", default=None, help="Score an existing predictions JSON without loading EVA01.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--with-gpt", action="store_true", help="Also run GPT-ref and GPT-img judges when OPENAI_API_KEY is set.")
    parser.add_argument("--gpt-model", default=None, help="GPT judge model. Defaults to OPENAI_JUDGE_MODEL or gpt-4o.")
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


def resolve_render_dir(path: str | None, benchmark: Path) -> Path:
    if path:
        return Path(path).expanduser()
    return benchmark / "renders_blender_pbr"


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


def load_predictions(path: str | Path) -> tuple[str, list[dict[str, str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompt = str(payload.get("prompt") or PROMPT)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Predictions JSON must contain a 'results' list.")
    return prompt, [dict(row) for row in results]


def build_render_manifest(rows: list[dict[str, Any]], render_dir: Path, prediction_by_id: dict[str, str]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        object_id = str(row["object_id"])
        base = render_dir / object_id
        renders = {view: str(base / f"{view}.png") for view in ("front", "right", "back", "left")}
        if all(Path(path).exists() for path in renders.values()):
            items.append({"object_id": object_id, "caption": prediction_by_id[object_id], "renders": renders})
    return items


def score_predictions(
    predictions: list[dict[str, str]],
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    with_gpt: bool = False,
    gpt_model: str | None = None,
    gpt_workers: int = 1,
) -> tuple[Path, dict[str, float]]:
    bundle = compute_caption_overlap_metrics(predictions)
    bundle = compute_meteor_score(bundle)
    bundle = compute_sentence_bert_similarity(bundle)
    bundle = compute_simcse_similarity(bundle, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu")
    bundle.metadata = metadata
    if with_gpt:
        bundle = compute_gpt_average_score(bundle, model_name=gpt_model or "", workers=gpt_workers)
    metrics_json = output_dir / "objaverse_captioning_prompt2_metrics.json"
    save_caption_metric_bundle(bundle, metrics_json)
    return metrics_json, bundle.overall_scores


def maybe_score_gpt_img(
    predictions: list[dict[str, str]],
    *,
    render_dir: Path,
    output_dir: Path,
    gpt_model: str | None,
    gpt_workers: int,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_by_id = {row["object_id"]: row["model_output"] for row in predictions}
    manifest = build_render_manifest(predictions, render_dir, prediction_by_id)
    manifest_path = output_dir / "gpt_img_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not manifest:
        print(f"Skip GPT-img: no complete render sets found under {render_dir}.", file=sys.stderr)
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skip GPT-img: OPENAI_API_KEY is missing; wrote manifest only.", file=sys.stderr)
        return None
    model_name = gpt_model or os.environ.get("OPENAI_JUDGE_MODEL") or "gpt-4o"
    judge_payload = score_manifest(manifest_path, model_name=model_name, workers=gpt_workers)
    scores_path = output_dir / "gpt_img_scores.json"
    scores_path.write_text(json.dumps(judge_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scores_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = args.checkpoint or (LORA_REPO if args.variant == "lora" else FULL_REPO)
    output_dir = Path(args.output_dir) / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.predictions_json:
        prompt, predictions = load_predictions(args.predictions_json)
        benchmark = resolve_benchmark(args.benchmark_dir) if args.with_gpt else None
        metrics_json, scores = score_predictions(
            predictions,
            output_dir=output_dir,
            metadata={
                "variant": args.variant,
                "checkpoint": checkpoint,
                "predictions_json": str(Path(args.predictions_json).expanduser()),
                "seed": args.seed,
                "prompt": prompt,
                "num_samples": len(predictions),
            },
            with_gpt=args.with_gpt,
            gpt_model=args.gpt_model,
            gpt_workers=args.gpt_workers,
        )
        gpt_img_json = None
        if args.with_gpt and benchmark is not None:
            gpt_img_json = maybe_score_gpt_img(
                predictions,
                render_dir=resolve_render_dir(args.render_dir, benchmark),
                output_dir=output_dir,
                gpt_model=args.gpt_model,
                gpt_workers=args.gpt_workers,
            )
        print(json.dumps({"metrics": str(metrics_json), "gpt_img_scores": str(gpt_img_json) if gpt_img_json else None, "scores": scores}, indent=2))
        return

    benchmark = resolve_benchmark(args.benchmark_dir)
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

    metrics_json, scores = score_predictions(
        predictions,
        output_dir=output_dir,
        metadata={
            "variant": args.variant,
            "checkpoint": checkpoint,
            "benchmark_dir": str(benchmark),
            "seed": args.seed,
            "prompt": PROMPT,
            "num_samples": len(predictions),
        },
        with_gpt=args.with_gpt,
        gpt_model=args.gpt_model,
        gpt_workers=args.gpt_workers,
    )
    gpt_img_json = None
    if args.with_gpt:
        gpt_img_json = maybe_score_gpt_img(
            predictions,
            render_dir=resolve_render_dir(args.render_dir, benchmark),
            output_dir=output_dir,
            gpt_model=args.gpt_model,
            gpt_workers=args.gpt_workers,
        )
    print(json.dumps({"predictions": str(predictions_json), "metrics": str(metrics_json), "gpt_img_scores": str(gpt_img_json) if gpt_img_json else None, "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
