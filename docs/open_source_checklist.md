# Open Source Checklist

This release is intended to contain only inference and evaluation code.

Before pushing:

- Confirm no checkpoint weights are committed.
- Confirm no local absolute data paths are committed.
- Confirm no API keys, auth files, or private endpoints are committed.
- Confirm no training code, training configs, ablation scripts, or experiment logs are committed.
- Confirm code and documentation are English only.
- Confirm `scripts/infer_mesh.py` runs with `OpenEVA-Instruct-LoRA`.
- Confirm `scripts/infer_mesh.py` runs with `OpenEVA-Instruct-Full`.
- Confirm `scripts/chat_mesh_cli.py` supports a two-turn mesh chat.
- Confirm `scripts/app_gradio.py` returns HTTP 200 on `127.0.0.1`.
- Confirm PointLLM-200 smoke eval works with two samples for both variants.
- Confirm `scripts/score_caption_metrics.py` produces a JSON with `overall_scores`, `results`, `metadata`, and `unavailable_metrics`.
- Confirm `scripts/score_render_gpt_judge.py` accepts the documented manifest format.
