#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eva01_instruct.render_judge import DEFAULT_MODEL, score_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run render-based GPT judge from a user-provided render manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = score_manifest(
        args.manifest,
        model_name=args.judge_model,
        workers=args.workers,
        request_timeout=args.request_timeout,
        max_output_tokens=args.max_output_tokens,
        max_image_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
        limit=args.limit,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
