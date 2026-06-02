#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eva01_instruct.mesh_io import POINT_PATCH_TOKEN, build_point_patch_text
from eva01_instruct.metrics import (
    POINTLLM_PROMPTS,
    compute_caption_overlap_metrics,
    compute_gpt_average_score,
    compute_meteor_score,
    compute_sentence_bert_similarity,
    compute_simcse_similarity,
    dump_pointllm_results_json,
    load_pointllm_eval_annotations,
    save_caption_metric_bundle,
)
from eva01_instruct.generation import OpenEVARuntime
from eva01_instruct.model import select_generated_tokens
from eva01_instruct.pointllm200 import PointLLM200CaptionDataset, collate_pointllm200


LOGGER = logging.getLogger("eval_pointllm200_caption")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PointLLM-200 captioning eval for OpenEVA-Instruct.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-annotation-path", required=True)
    parser.add_argument("--pointcloud-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-index", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-metrics", action="store_true")
    parser.add_argument("--enable-meteor", action="store_true")
    parser.add_argument("--enable-sentence-bert", action="store_true")
    parser.add_argument("--sentence-bert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--enable-simcse", action="store_true")
    parser.add_argument("--simcse-model", default="princeton-nlp/sup-simcse-roberta-large")
    parser.add_argument("--simcse-batch-size", type=int, default=16)
    parser.add_argument("--enable-gpt-average-score", action="store_true")
    parser.add_argument("--gpt-model-name", default="gpt-5.5")
    parser.add_argument("--gpt-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_pointllm_eval_annotations(args.eval_annotation_path)
    if args.limit is not None:
        annotations = annotations[: int(args.limit)]
    runtime = OpenEVARuntime(
        checkpoint_dir=args.checkpoint,
        device=args.device,
        torch_dtype=args.torch_dtype,
        pointnum=args.pointnum,
    )

    prompt_raw = f"{POINT_PATCH_TOKEN}\n{POINTLLM_PROMPTS[args.prompt_index]}"
    point_patch_text = build_point_patch_text(runtime.point_patch_token, runtime.point_token_len)
    prompt_text = prompt_raw.replace(POINT_PATCH_TOKEN, point_patch_text)
    rendered = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = runtime.tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=runtime.max_length,
        return_tensors="pt",
    )
    prompt_len = int(encoded.input_ids.shape[1])

    dataset = PointLLM200CaptionDataset(
        annotations=annotations,
        pointcloud_dir=args.pointcloud_dir,
        pointnum=args.pointnum,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pointllm200,
    )

    rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in dataloader:
            batch_size = int(batch["point_clouds"].shape[0])
            input_ids = encoded.input_ids.repeat(batch_size, 1).to(runtime.model.device)
            attention_mask = encoded.attention_mask.repeat(batch_size, 1).to(runtime.model.device)
            do_sample = float(args.temperature) > 0.0
            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "do_sample": do_sample,
            }
            if do_sample:
                generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
            sequences = runtime.model.generate_from_point_clouds(
                input_ids=input_ids,
                attention_mask=attention_mask,
                point_clouds=batch["point_clouds"].to(runtime.model.device),
                **generate_kwargs,
            )
            for index, object_id in enumerate(batch["object_ids"]):
                generated = select_generated_tokens(sequences[index : index + 1], prompt_len=prompt_len)[0]
                prediction = runtime.tokenizer.decode(generated, skip_special_tokens=True).strip()
                rows.append(
                    {
                        "object_id": object_id,
                        "ground_truth": batch["ground_truths"][index],
                        "model_output": prediction,
                    }
                )
            LOGGER.info("Generated %d/%d predictions", len(rows), len(dataset))

    prediction_json_path = output_dir / f"objaverse_captioning_prompt{args.prompt_index}_predictions.json"
    prediction_jsonl_path = output_dir / f"objaverse_captioning_prompt{args.prompt_index}_predictions.jsonl"
    dump_pointllm_results_json(prompt_text=POINTLLM_PROMPTS[args.prompt_index], rows=rows, output_path=prediction_json_path)
    with prediction_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "prediction_json": str(prediction_json_path),
        "prediction_jsonl": str(prediction_jsonl_path),
        "num_samples": len(rows),
    }
    if args.score_metrics:
        bundle = compute_caption_overlap_metrics(rows)
        metadata = dict(bundle.metadata or {})
        metadata.update(
            {
                "checkpoint": args.checkpoint,
                "eval_annotation_path": args.eval_annotation_path,
                "pointcloud_dir": args.pointcloud_dir,
                "prompt_index": args.prompt_index,
                "num_samples": len(rows),
            }
        )
        bundle.metadata = metadata
        if args.enable_meteor:
            bundle = compute_meteor_score(bundle)
        if args.enable_sentence_bert:
            bundle = compute_sentence_bert_similarity(bundle, model_name=args.sentence_bert_model)
        if args.enable_simcse:
            bundle = compute_simcse_similarity(bundle, model_name=args.simcse_model, batch_size=args.simcse_batch_size, device=args.device)
        if args.enable_gpt_average_score:
            bundle = compute_gpt_average_score(bundle, model_name=args.gpt_model_name, workers=args.gpt_workers)
        metrics_path = output_dir / f"objaverse_captioning_prompt{args.prompt_index}_metrics.json"
        save_caption_metric_bundle(bundle, metrics_path)
        summary["metrics_json"] = str(metrics_path)
        summary["overall_scores"] = {key: round(value, 6) for key, value in bundle.overall_scores.items()}
        summary["unavailable_metrics"] = bundle.unavailable_metrics

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
