from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

LOGGER = logging.getLogger("eva01.metrics")

GPT_CAPTION_PROMPT = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score.

Provide your score (0-100) and a short justification (less than 15 words) in the format of 'score#reason'

Example:
Human: A white brown skeleton
Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.
Output: 50#mention white; skeleton and robot have similar appearence.

Now score the following:
Human: {ground_truth}
Model: {model_output}
Output: """


@dataclass
class CaptionMetricBundle:
    overall_scores: Dict[str, float]
    per_sample: List[Dict[str, Any]]
    unavailable_metrics: List[str]
    metadata: Dict[str, Any] | None = None


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_spaces(text).lower()
    return re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def _bleu_n(reference: Sequence[str], candidate: Sequence[str], n: int, epsilon: float = 0.1) -> float:
    if not candidate:
        return 0.0
    precisions: List[float] = []
    for order in range(1, n + 1):
        cand_ngrams = _ngrams(candidate, order)
        total = sum(cand_ngrams.values())
        if total == 0:
            return 0.0
        ref_ngrams = _ngrams(reference, order)
        clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
        precision = (clipped + epsilon) / (total + epsilon) if clipped == 0 else clipped / total
        precisions.append(precision)
    if not precisions or min(precisions) <= 0:
        return 0.0
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / len(precisions))
    ref_len = len(reference)
    cand_len = len(candidate)
    brevity_penalty = 0.0 if cand_len == 0 else (1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / cand_len)))
    return 100.0 * brevity_penalty * geo_mean


def _f1_from_overlap(reference_items: Counter[Any], candidate_items: Counter[Any]) -> float:
    overlap = sum(min(count, reference_items[item]) for item, count in candidate_items.items())
    reference_total = sum(reference_items.values())
    candidate_total = sum(candidate_items.values())
    if reference_total == 0 or candidate_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / candidate_total
    recall = overlap / reference_total
    return 100.0 * (2.0 * precision * recall) / (precision + recall)


def _rouge_n_f1(reference: Sequence[str], candidate: Sequence[str], n: int) -> float:
    return _f1_from_overlap(_ngrams(reference, n), _ngrams(candidate, n))


def _lcs_length(reference: Sequence[str], candidate: Sequence[str]) -> int:
    if not reference or not candidate:
        return 0
    prev = [0] * (len(candidate) + 1)
    for ref_token in reference:
        curr = [0]
        for idx, cand_token in enumerate(candidate, start=1):
            if ref_token == cand_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_f1(reference: Sequence[str], candidate: Sequence[str]) -> float:
    if not reference or not candidate:
        return 0.0
    lcs = _lcs_length(reference, candidate)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    return 100.0 * (2.0 * precision * recall) / (precision + recall)


def compute_caption_overlap_metrics(rows: Sequence[Dict[str, Any]]) -> CaptionMetricBundle:
    score_names = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "rouge-1", "rouge-2", "rouge-l"]
    collected: Dict[str, List[float]] = {name: [] for name in score_names}
    per_sample: List[Dict[str, Any]] = []

    for row in rows:
        object_id = str(row.get("object_id", ""))
        ground_truth = _normalize_spaces(row.get("ground_truth", ""))
        model_output = _normalize_spaces(row.get("model_output", ""))
        reference_tokens = _tokenize(ground_truth)
        candidate_tokens = _tokenize(model_output)
        sample_scores = {
            "bleu-1": _bleu_n(reference_tokens, candidate_tokens, 1),
            "bleu-2": _bleu_n(reference_tokens, candidate_tokens, 2),
            "bleu-3": _bleu_n(reference_tokens, candidate_tokens, 3),
            "bleu-4": _bleu_n(reference_tokens, candidate_tokens, 4),
            "rouge-1": _rouge_n_f1(reference_tokens, candidate_tokens, 1),
            "rouge-2": _rouge_n_f1(reference_tokens, candidate_tokens, 2),
            "rouge-l": _rouge_l_f1(reference_tokens, candidate_tokens),
        }
        for name, value in sample_scores.items():
            collected[name].append(value)
        per_sample.append(
            {
                "object_id": object_id,
                "ground_truth": ground_truth,
                "model_output": model_output,
                "scores": {key: round(value, 6) for key, value in sample_scores.items()},
            }
        )

    overall_scores = {name: (sum(values) / len(values) if values else 0.0) for name, values in collected.items()}
    return CaptionMetricBundle(
        overall_scores=overall_scores,
        per_sample=per_sample,
        unavailable_metrics=["meteor", "sentence-bert", "simcse", "average_score"],
        metadata={},
    )


def compute_meteor_score(bundle: CaptionMetricBundle) -> CaptionMetricBundle:
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
    except Exception as exc:
        LOGGER.warning("METEOR is unavailable: %s", exc)
        if "meteor" not in bundle.unavailable_metrics:
            bundle.unavailable_metrics.append("meteor")
        return bundle

    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        if not nltk.download("wordnet", quiet=True):
            LOGGER.warning("Failed to download NLTK wordnet; skip METEOR.")
            if "meteor" not in bundle.unavailable_metrics:
                bundle.unavailable_metrics.append("meteor")
            return bundle

    scores: List[float] = []
    for row in bundle.per_sample:
        ground_truth = _normalize_spaces(row.get("ground_truth", ""))
        model_output = _normalize_spaces(row.get("model_output", "")) or "##"
        score = 100.0 * float(meteor_score([ground_truth.split()], model_output.split()))
        row.setdefault("scores", {})["meteor"] = round(score, 6)
        scores.append(score)

    bundle.overall_scores["meteor"] = sum(scores) / len(scores) if scores else 0.0
    bundle.unavailable_metrics = [name for name in bundle.unavailable_metrics if name != "meteor"]
    return bundle


def compute_sentence_bert_similarity(
    bundle: CaptionMetricBundle,
    *,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> CaptionMetricBundle:
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as exc:
        LOGGER.warning("Sentence-BERT is unavailable: %s", exc)
        if "sentence-bert" not in bundle.unavailable_metrics:
            bundle.unavailable_metrics.append("sentence-bert")
        return bundle

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        LOGGER.warning("Sentence-BERT model load failed: %s", exc)
        if "sentence-bert" not in bundle.unavailable_metrics:
            bundle.unavailable_metrics.append("sentence-bert")
        return bundle
    ground_truths = [str(row.get("ground_truth", "")) for row in bundle.per_sample]
    predictions = [str(row.get("model_output", "")) or "##" for row in bundle.per_sample]
    gt_embeddings = model.encode(ground_truths, convert_to_tensor=True, show_progress_bar=False)
    pred_embeddings = model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    similarities = util.cos_sim(gt_embeddings, pred_embeddings).diagonal().tolist()
    scaled = [100.0 * float(value) for value in similarities]
    for row, score in zip(bundle.per_sample, scaled):
        row.setdefault("scores", {})["sentence-bert"] = round(score, 6)
    bundle.overall_scores["sentence-bert"] = sum(scaled) / len(scaled) if scaled else 0.0
    bundle.unavailable_metrics = [name for name in bundle.unavailable_metrics if name != "sentence-bert"]
    metadata = dict(bundle.metadata or {})
    metadata["sentence_bert_model_name"] = model_name
    bundle.metadata = metadata
    return bundle


def compute_simcse_similarity(
    bundle: CaptionMetricBundle,
    *,
    model_name: str = "princeton-nlp/sup-simcse-roberta-large",
    batch_size: int = 32,
    device: str | None = None,
) -> CaptionMetricBundle:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        LOGGER.warning("SimCSE is unavailable: %s", exc)
        if "simcse" not in bundle.unavailable_metrics:
            bundle.unavailable_metrics.append("simcse")
        return bundle

    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(resolved_device)
    except Exception as exc:
        LOGGER.warning("SimCSE model load failed: %s", exc)
        if "simcse" not in bundle.unavailable_metrics:
            bundle.unavailable_metrics.append("simcse")
        return bundle
    model.eval()

    def encode_texts(texts: Sequence[str]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        for start in range(0, len(texts), int(batch_size)):
            batch_texts = [text or "##" for text in texts[start : start + int(batch_size)]]
            encoded = tokenizer(list(batch_texts), padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model(**encoded, return_dict=True)
            pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
            embeddings.append(pooled.detach().cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 1), dtype=torch.float32)

    ground_truths = [str(row.get("ground_truth", "")) for row in bundle.per_sample]
    predictions = [str(row.get("model_output", "")) or "##" for row in bundle.per_sample]
    gt_embeddings = encode_texts(ground_truths)
    pred_embeddings = encode_texts(predictions)
    similarities = F.cosine_similarity(gt_embeddings, pred_embeddings, dim=1).tolist()
    scaled = [100.0 * float(value) for value in similarities]
    for row, score in zip(bundle.per_sample, scaled):
        row.setdefault("scores", {})["simcse"] = round(score, 6)
    bundle.overall_scores["simcse"] = sum(scaled) / len(scaled) if scaled else 0.0
    bundle.unavailable_metrics = [name for name in bundle.unavailable_metrics if name != "simcse"]
    metadata = dict(bundle.metadata or {})
    metadata["simcse_model_name"] = model_name
    metadata["simcse_device"] = str(resolved_device)
    bundle.metadata = metadata
    return bundle


def save_caption_metric_bundle(bundle: CaptionMetricBundle, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall_scores": {key: round(value, 6) for key, value in bundle.overall_scores.items()},
        "unavailable_metrics": list(bundle.unavailable_metrics),
        "results": bundle.per_sample,
    }
    if bundle.metadata:
        payload["metadata"] = bundle.metadata
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _resolve_openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"


def _parse_responses_sse(lines: Iterable[str]) -> tuple[str, Dict[str, int]]:
    text_chunks: List[str] = []
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
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
            text_chunks.append(str(payload.get("delta", "")))
        response_payload = payload.get("response")
        if isinstance(response_payload, dict):
            usage_payload = response_payload.get("usage")
            if isinstance(usage_payload, dict):
                usage["prompt_tokens"] = int(usage_payload.get("input_tokens", usage["prompt_tokens"]) or 0)
                usage["completion_tokens"] = int(usage_payload.get("output_tokens", usage["completion_tokens"]) or 0)
    return "".join(text_chunks).strip(), usage


def request_openai_text_response(
    *,
    prompt: str,
    model_name: str = "",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 256,
) -> tuple[str, Dict[str, int]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    model_name = model_name or os.environ.get("OPENAI_JUDGE_MODEL") or "gpt-4o"
    wire_api = str(os.environ.get("OPENAI_WIRE_API", "responses")).strip().lower()
    base_url = _resolve_openai_base_url()
    request_timeout = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "60"))

    if wire_api == "responses":
        import requests

        response = requests.post(
            base_url.rstrip("/") + "/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": int(max_tokens),
                "stream": True,
            },
            timeout=request_timeout,
            stream=True,
        )
        response.raise_for_status()
        return _parse_responses_sse(response.iter_lines(decode_unicode=True))

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = str(response.choices[0].message.content or "").strip()
    usage = {
        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
    }
    return content, usage


def parse_gpt_caption_score_response(response_text: str) -> tuple[int, str]:
    match = re.search(r"(\d+\s*#.*)", str(response_text or "").strip())
    parsed = match.group(1) if match else str(response_text or "").strip()
    fields = parsed.split("#", 1)
    raw_score = fields[0].strip()
    reason = fields[1].strip() if len(fields) > 1 else ""
    try:
        score = int(raw_score)
    except ValueError:
        return -1, parsed
    if score < 0 or score > 100:
        return -1, parsed
    return score, reason


def _score_gpt_row(index: int, row: Dict[str, Any], *, model_name: str) -> tuple[int, str, Dict[str, int], int | None, str]:
    prompt = GPT_CAPTION_PROMPT.format(
        ground_truth=row.get("ground_truth", ""),
        model_output=row.get("model_output", ""),
    )
    content, usage = request_openai_text_response(
        prompt=prompt,
        model_name=model_name,
        temperature=1,
        top_p=1,
        max_tokens=256,
    )
    score, reason = parse_gpt_caption_score_response(content)
    return index, content, usage, score if score >= 0 else None, reason


def compute_gpt_average_score(
    bundle: CaptionMetricBundle,
    *,
    model_name: str = "",
    workers: int = 1,
) -> CaptionMetricBundle:
    if not os.environ.get("OPENAI_API_KEY"):
        LOGGER.warning("OPENAI_API_KEY is missing; skip GPT average_score.")
        if "average_score" not in bundle.unavailable_metrics:
            bundle.unavailable_metrics.append("average_score")
        return bundle

    model_name = model_name or os.environ.get("OPENAI_JUDGE_MODEL") or "gpt-4o"
    prompt_tokens = 0
    completion_tokens = 0
    if workers <= 1:
        futures_results = []
        for index, row in enumerate(bundle.per_sample):
            try:
                futures_results.append(_score_gpt_row(index, row, model_name=model_name))
            except Exception as exc:
                LOGGER.warning("GPT average_score request failed for object_id=%s: %s", row.get("object_id"), exc)
                futures_results.append((index, "", {"prompt_tokens": 0, "completion_tokens": 0}, None, ""))
    else:
        futures_results = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(_score_gpt_row, index, row, model_name=model_name)
                for index, row in enumerate(bundle.per_sample)
            ]
            for future in as_completed(futures):
                try:
                    futures_results.append(future.result())
                except Exception as exc:
                    LOGGER.warning("GPT average_score request failed: %s", exc)

    for index, content, usage, score, reason in futures_results:
        row = bundle.per_sample[index]
        row["gpt_average_score_raw_response"] = content
        row["gpt_reason"] = reason
        row["gpt_score"] = score
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", 0))

    valid_scores = [float(row["gpt_score"]) for row in bundle.per_sample if row.get("gpt_score") is not None]
    if valid_scores:
        bundle.overall_scores["average_score"] = sum(valid_scores) / len(valid_scores)
        bundle.unavailable_metrics = [name for name in bundle.unavailable_metrics if name != "average_score"]
    else:
        bundle.overall_scores.pop("average_score", None)
        if "average_score" not in bundle.unavailable_metrics:
            bundle.unavailable_metrics.append("average_score")

    metadata = dict(bundle.metadata or {})
    metadata["gpt_model_name"] = model_name
    metadata["gpt_prompt_tokens"] = prompt_tokens
    metadata["gpt_completion_tokens"] = completion_tokens
    metadata["gpt_valid_scores"] = len(valid_scores)
    metadata["gpt_invalid_scores"] = len(bundle.per_sample) - len(valid_scores)
    metadata["gpt_workers"] = workers
    bundle.metadata = metadata
    return bundle

__all__ = [
    "CaptionMetricBundle",
    "GPT_CAPTION_PROMPT",
    "compute_caption_overlap_metrics",
    "compute_gpt_average_score",
    "compute_meteor_score",
    "compute_sentence_bert_similarity",
    "compute_simcse_similarity",
    "parse_gpt_caption_score_response",
    "request_openai_text_response",
    "save_caption_metric_bundle",
]
