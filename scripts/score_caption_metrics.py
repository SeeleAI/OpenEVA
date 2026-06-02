#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eva01_instruct.metrics import (
    compute_caption_overlap_metrics,
    compute_gpt_average_score,
    compute_meteor_score,
    compute_sentence_bert_similarity,
    compute_simcse_similarity,
    save_caption_metric_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score PointLLM-format caption predictions.")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--enable-meteor", action="store_true")
    parser.add_argument("--enable-sentence-bert", action="store_true")
    parser.add_argument("--sentence-bert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--enable-simcse", action="store_true")
    parser.add_argument("--simcse-model", default="princeton-nlp/sup-simcse-roberta-large")
    parser.add_argument("--simcse-batch-size", type=int, default=16)
    parser.add_argument("--simcse-device", default=None)
    parser.add_argument("--enable-gpt-average-score", action="store_true")
    parser.add_argument("--gpt-model-name", default="gpt-5.5")
    parser.add_argument("--gpt-workers", type=int, default=1)
    return parser.parse_args()


def load_rows(path: str | Path):
    results_path = Path(path)
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list or a dict with key 'results': {results_path}")
    return payload.get("prompt") if isinstance(payload, dict) else None, [dict(row) for row in rows]


def main() -> None:
    args = parse_args()
    prompt, rows = load_rows(args.results_path)
    bundle = compute_caption_overlap_metrics(rows)
    metadata = dict(bundle.metadata or {})
    metadata["source_results_path"] = args.results_path
    metadata["prompt"] = prompt
    metadata["num_samples"] = len(rows)
    bundle.metadata = metadata
    if args.enable_meteor:
        bundle = compute_meteor_score(bundle)
    if args.enable_sentence_bert:
        bundle = compute_sentence_bert_similarity(bundle, model_name=args.sentence_bert_model)
    if args.enable_simcse:
        bundle = compute_simcse_similarity(
            bundle,
            model_name=args.simcse_model,
            batch_size=args.simcse_batch_size,
            device=args.simcse_device,
        )
    if args.enable_gpt_average_score:
        bundle = compute_gpt_average_score(bundle, model_name=args.gpt_model_name, workers=args.gpt_workers)
    output_path = Path(args.output_path) if args.output_path else Path(args.results_path).with_name(Path(args.results_path).stem + "_metrics.json")
    save_caption_metric_bundle(bundle, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "overall_scores": {key: round(value, 6) for key, value in bundle.overall_scores.items()},
                "unavailable_metrics": bundle.unavailable_metrics,
                "metadata": bundle.metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
