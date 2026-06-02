# Evaluation

This repository provides inference-time evaluation scripts only. It does not include training data, model outputs, render images, or benchmark tables.

## PointLLM-200 Caption Eval

The PointLLM-200 caption protocol uses the official point-cloud inputs and the prompt:

```text
Caption this 3D model in detail.
```

The script expects:

```text
PointLLM_brief_description_val_200_GT.json
<object_id>_8192.npy
```

Run:

```bash
python scripts/eval_pointllm200_caption.py \
  --checkpoint checkpoints/openeva-instruct-lora \
  --eval-annotation-path data/pointllm200/annotations/PointLLM_brief_description_val_200_GT.json \
  --pointcloud-dir data/pointllm200/pointclouds/8192_npy \
  --output-dir outputs/pointllm200_lora \
  --prompt-index 2
```

For a small smoke test, add:

```bash
--limit 2 --batch-size 1
```

The generation script writes:

```text
objaverse_captioning_prompt2_predictions.json
objaverse_captioning_prompt2_predictions.jsonl
```

## Caption Metrics

The metrics script supports:

- BLEU-1, BLEU-2, BLEU-3, BLEU-4.
- ROUGE-1, ROUGE-2, ROUGE-L.
- METEOR.
- Sentence-BERT.
- SimCSE.
- GPT `average_score`.

Default scoring runs overlap metrics only. Optional metrics are enabled by flags:

```bash
python scripts/score_caption_metrics.py \
  --results-path outputs/pointllm200_lora/objaverse_captioning_prompt2_predictions.json \
  --output-path outputs/pointllm200_lora/objaverse_captioning_prompt2_metrics.json \
  --enable-meteor \
  --enable-sentence-bert \
  --enable-simcse
```

GPT scoring is default off. To enable it, provide `OPENAI_API_KEY` and run:

```bash
python scripts/score_caption_metrics.py \
  --results-path outputs/pointllm200_lora/objaverse_captioning_prompt2_predictions.json \
  --output-path outputs/pointllm200_lora/objaverse_captioning_prompt2_metrics_gpt.json \
  --enable-gpt-average-score \
  --gpt-model-name gpt-5.5
```

The output JSON contains:

- `overall_scores`
- `results`
- `metadata`
- `unavailable_metrics`

## Render-Based GPT Judge

The render judge is judge-only. It does not render meshes. Provide a manifest with render paths:

```json
[
  {
    "object_id": "sample_001",
    "caption": "model prediction",
    "renders": {
      "front": "path/to/front.png",
      "right": "path/to/right.png",
      "back": "path/to/back.png",
      "left": "path/to/left.png"
    }
  }
]
```

Run:

```bash
python scripts/score_render_gpt_judge.py \
  --manifest render_manifest.json \
  --output-path outputs/render_gpt_judge.json \
  --judge-model gpt-5.5
```

The judge uses the OpenAI Responses API shape. `OPENAI_BASE_URL` may be empty for the official OpenAI API, or set to an OpenAI-compatible endpoint.
