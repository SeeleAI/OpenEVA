from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any, Dict, List

import requests
from PIL import Image


DEFAULT_MODEL = os.environ.get("OPENAI_JUDGE_MODEL") or "gpt-4o"
RGB_VIEWS = ("front", "right", "back", "left")

RUBRIC = """Render-based GPT judge for PointLLM-200 mesh captioning.

Inputs:
- Four RGB renders of the same 3D object: front, right, back, and left.
- One model-generated caption.
- The human caption is intentionally not provided.

Goal:
Score how faithfully the caption describes the visible 3D object in the renders.
The score is a scalar integer from 0 to 100, where higher is better.

Rubric:
1. Core object identity and function (0-35): award high credit when the caption names the correct object category or a close synonym.
2. Geometry, structure, and parts (0-25): reward correct major visible components, shape, attachments, symmetry, and distinctive geometry.
3. Color, material, and texture (0-15): reward correct visible colors, material cues, texture patterns, and surface finish.
4. Fine-grained attributes and style (0-15): reward accurate style, decorative motifs, proportions, pose, orientation, and special identifying features.
5. Caption quality and specificity (0-10): reward concise but informative captions and penalize vague or repetitive captions.

Penalties and caps:
- Severe hallucination should reduce the score.
- Empty, non-natural-language, or token-like captions should score 0-10.
- Captions for a completely different object should score 0-20.
- Broadly correct category with many wrong details should usually score 40-70.

Output strict JSON only:
{"score": <integer 0-100>, "reason": "<one short sentence>", "matched": ["..."], "errors": ["..."]}
"""


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Render judge manifest must be a list or a dict with an 'items' list.")
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        object_id = str(row.get("object_id", "")).strip()
        caption = str(row.get("caption", row.get("model_output", ""))).strip()
        renders = row.get("renders") or row.get("paths") or {}
        if not object_id:
            raise ValueError("Each manifest row must contain object_id.")
        missing = [view for view in RGB_VIEWS if not renders.get(view)]
        if missing:
            raise ValueError(f"{object_id} is missing render paths for: {', '.join(missing)}")
        normalized.append({"object_id": object_id, "caption": caption, "renders": {view: str(renders[view]) for view in RGB_VIEWS}})
    return normalized


def _resolve_openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"


def image_to_data_url(path: Path, *, max_side: int, quality: int) -> str:
    image = Image.open(path).convert("RGB")
    if max_side > 0:
        width, height = image.size
        scale = min(1.0, float(max_side) / float(max(width, height)))
        if scale < 1.0:
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_content(caption: str, image_paths: Dict[str, str], *, max_side: int, quality: int) -> List[Dict[str, Any]]:
    prompt = (
        f"{RUBRIC}\n\n"
        "Now score the following generated caption against the provided RGB renders only.\n"
        "Do not use any external ground-truth caption or dataset knowledge.\n\n"
        f"Generated caption:\n{caption.strip() or '[EMPTY]'}"
    )
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for view in RGB_VIEWS:
        content.append({"type": "input_text", "text": f"RGB render view: {view}"})
        content.append({"type": "input_image", "image_url": image_to_data_url(Path(image_paths[view]), max_side=max_side, quality=quality)})
    return content


def parse_sse_response(lines: Any) -> tuple[str, Dict[str, int]]:
    chunks: List[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line.startswith("data: "):
            continue
        payload_text = line[len("data: ") :].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "response.output_text.delta":
            chunks.append(str(payload.get("delta", "")))
        response_payload = payload.get("response")
        if isinstance(response_payload, dict):
            usage_payload = response_payload.get("usage")
            if isinstance(usage_payload, dict):
                usage["prompt_tokens"] = int(usage_payload.get("input_tokens", usage["prompt_tokens"]) or 0)
                usage["completion_tokens"] = int(usage_payload.get("output_tokens", usage["completion_tokens"]) or 0)
    return "".join(chunks).strip(), usage


def request_judge(*, content: List[Dict[str, Any]], model: str, timeout: float, max_output_tokens: int) -> tuple[str, Dict[str, int]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    response = requests.post(
        _resolve_openai_base_url().rstrip("/") + "/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(max_output_tokens),
            "stream": True,
        },
        timeout=float(timeout),
        stream=True,
    )
    response.raise_for_status()
    return parse_sse_response(response.iter_lines(decode_unicode=True))


def parse_score(raw_text: str) -> tuple[float | None, str, List[str], List[str], bool]:
    text = raw_text.strip()
    parsed: Dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            return None, "", [], [], True
        score = float(match.group(1))
        converted = False
        if 0.0 <= score <= 1.0:
            score *= 100.0
            converted = True
        return max(0.0, min(100.0, score)), text[:300], [], [], converted

    try:
        score = float(parsed.get("score"))
    except Exception:
        return None, str(parsed.get("reason", "")), [], [], True
    converted = False
    if 0.0 <= score <= 1.0:
        score *= 100.0
        converted = True
    matched = parsed.get("matched", [])
    errors = parsed.get("errors", [])
    if not isinstance(matched, list):
        matched = [str(matched)]
    if not isinstance(errors, list):
        errors = [str(errors)]
    return max(0.0, min(100.0, score)), str(parsed.get("reason", "")), [str(x) for x in matched], [str(x) for x in errors], converted


def score_one(
    row: Dict[str, Any],
    *,
    model_name: str,
    timeout: float,
    max_output_tokens: int,
    max_image_side: int,
    jpeg_quality: int,
    retries: int = 2,
) -> Dict[str, Any]:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            content = build_content(row["caption"], row["renders"], max_side=max_image_side, quality=jpeg_quality)
            raw, usage = request_judge(content=content, model=model_name, timeout=timeout, max_output_tokens=max_output_tokens)
            score, reason, matched, errors, converted = parse_score(raw)
            return {
                "object_id": row["object_id"],
                "caption": row["caption"],
                "score": score,
                "valid": score is not None,
                "reason": reason,
                "matched": matched,
                "errors": errors,
                "score_scale_converted_from_0_1": converted,
                "raw_response": raw,
                "usage": usage,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(2.0 * (attempt + 1))
    return {
        "object_id": row["object_id"],
        "caption": row["caption"],
        "score": None,
        "valid": False,
        "reason": "",
        "matched": [],
        "errors": [],
        "score_scale_converted_from_0_1": False,
        "raw_response": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "attempts": retries + 1,
        "request_error": last_error,
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    return {
        "num_samples": len(rows),
        "valid_scores": len(scores),
        "invalid_scores": len(rows) - len(scores),
        "mean": sum(scores) / len(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "score_scale_converted_from_0_1_count": sum(1 for row in rows if row.get("score_scale_converted_from_0_1")),
        "prompt_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens", 0) or 0) for row in rows),
        "completion_tokens": sum(int((row.get("usage") or {}).get("completion_tokens", 0) or 0) for row in rows),
    }


def score_manifest(
    manifest_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    workers: int = 4,
    request_timeout: float = 180.0,
    max_output_tokens: int = 256,
    max_image_side: int = 1024,
    jpeg_quality: int = 92,
    limit: int | None = None,
) -> Dict[str, Any]:
    rows = load_manifest(manifest_path)
    if limit is not None:
        rows = rows[: int(limit)]
    results: List[Dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                score_one,
                row,
                model_name=model_name,
                timeout=request_timeout,
                max_output_tokens=max_output_tokens,
                max_image_side=max_image_side,
                jpeg_quality=jpeg_quality,
            ): index
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    final_rows = [row for row in results if row is not None]
    return {
        "judge_model": model_name,
        "manifest_path": str(manifest_path),
        "rgb_views": list(RGB_VIEWS),
        "rubric": RUBRIC,
        "results": final_rows,
        "summary": summarize_rows(final_rows),
    }


__all__ = ["DEFAULT_MODEL", "RGB_VIEWS", "RUBRIC", "score_manifest"]
