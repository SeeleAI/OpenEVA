"""Frozen tokenizer ABI for EVA01 Unified All-Task V2.

The tokenizer rows already exist in the OpenEVA tokenizer and in the original
Qwen embedding matrix.  V2 does not resize the frozen Qwen vocabulary.  Only
the eight start/end rows receive adapter-owned trainable overrides.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch


MESH_START_TOKEN = "<|mesh_start|>"
MESH_END_TOKEN = "<|mesh_end|>"
MESH_PAD_TOKEN = "<|mesh_pad|>"
SS_START_TOKEN = "<|ss_start|>"
SS_END_TOKEN = "<|ss_end|>"
SS_PAD_TOKEN = "<|ss_pad|>"
SHAPE_START_TOKEN = "<|shape_start|>"
SHAPE_END_TOKEN = "<|shape_end|>"
SHAPE_PAD_TOKEN = "<|shape_pad|>"
TEX_START_TOKEN = "<|tex_start|>"
TEX_END_TOKEN = "<|tex_end|>"
TEX_PAD_TOKEN = "<|tex_pad|>"
POINT_PATCH_TOKEN = "<point_patch>"


ALL_TASK_TOKEN_IDS: "OrderedDict[str, int]" = OrderedDict(
    (
        (MESH_START_TOKEN, 151669),
        (MESH_END_TOKEN, 151670),
        (MESH_PAD_TOKEN, 151671),
        (SS_START_TOKEN, 151672),
        (SS_END_TOKEN, 151673),
        (SS_PAD_TOKEN, 151674),
        (SHAPE_START_TOKEN, 151675),
        (SHAPE_END_TOKEN, 151676),
        (SHAPE_PAD_TOKEN, 151677),
        (TEX_START_TOKEN, 151678),
        (TEX_END_TOKEN, 151679),
        (TEX_PAD_TOKEN, 151680),
        (POINT_PATCH_TOKEN, 151684),
    )
)

FAMILY_TOKEN_ABI: Mapping[str, Tuple[str, str, str]] = {
    "ss": (SS_START_TOKEN, SS_END_TOKEN, SS_PAD_TOKEN),
    "shape": (SHAPE_START_TOKEN, SHAPE_END_TOKEN, SHAPE_PAD_TOKEN),
    # Public naming remains Material; tex is the frozen tokenizer/checkpoint ABI.
    "material": (TEX_START_TOKEN, TEX_END_TOKEN, TEX_PAD_TOKEN),
}

FAMILY_TOKEN_IDS: Mapping[str, Tuple[int, int, int]] = {
    family: tuple(ALL_TASK_TOKEN_IDS[token] for token in tokens)
    for family, tokens in FAMILY_TOKEN_ABI.items()
}

SPECIAL_EMBEDDING_TOKEN_IDS: Tuple[int, ...] = (
    ALL_TASK_TOKEN_IDS[MESH_START_TOKEN],
    ALL_TASK_TOKEN_IDS[MESH_END_TOKEN],
    ALL_TASK_TOKEN_IDS[SS_START_TOKEN],
    ALL_TASK_TOKEN_IDS[SS_END_TOKEN],
    ALL_TASK_TOKEN_IDS[SHAPE_START_TOKEN],
    ALL_TASK_TOKEN_IDS[SHAPE_END_TOKEN],
    ALL_TASK_TOKEN_IDS[TEX_START_TOKEN],
    ALL_TASK_TOKEN_IDS[TEX_END_TOKEN],
)

SPECIAL_EMBEDDING_ROW_BY_TOKEN_ID: Mapping[int, int] = {
    token_id: row for row, token_id in enumerate(SPECIAL_EMBEDDING_TOKEN_IDS)
}

POINT_PATCH_TOKEN_ID = ALL_TASK_TOKEN_IDS[POINT_PATCH_TOKEN]
MESH_START_TOKEN_ID = ALL_TASK_TOKEN_IDS[MESH_START_TOKEN]
MESH_END_TOKEN_ID = ALL_TASK_TOKEN_IDS[MESH_END_TOKEN]
SPECIAL_EMBEDDING_ABI = "adapter_override_8x2048_qwen_source_rows_v1"


def validate_all_task_tokenizer(
    tokenizer: Any,
    *,
    reference_tokenizer: Any | None = None,
    ordinary_text_samples: Sequence[str] = (),
) -> Dict[str, Any]:
    """Validate fixed IDs and optionally prove ordinary Qwen token equality."""

    observed: Dict[str, int] = {}
    encoded_singletons: Dict[str, Tuple[int, ...]] = {}
    for token, expected in ALL_TASK_TOKEN_IDS.items():
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tuple(
            int(value)
            for value in tokenizer(
                token,
                add_special_tokens=False,
                truncation=False,
            ).input_ids
        )
        observed[token] = token_id
        encoded_singletons[token] = encoded
        if token_id != int(expected) or encoded != (int(expected),):
            raise RuntimeError(
                f"All-task tokenizer ABI drifted for {token}: "
                f"id={token_id}, encoded={encoded}, expected={expected}."
            )

    samples = tuple(str(value) for value in ordinary_text_samples)
    ordinary_rows = []
    if reference_tokenizer is not None:
        if not samples:
            samples = (
                "A small wooden chair with four straight legs.",
                "Describe the geometry and material of this object.",
                "Turn the round base into a square pedestal.",
                "Image and text conditions must use the same Qwen branch.",
            )
        for text in samples:
            current = tuple(
                int(value)
                for value in tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                ).input_ids
            )
            reference = tuple(
                int(value)
                for value in reference_tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                ).input_ids
            )
            ordinary_rows.append(
                {
                    "text": text,
                    "token_count": len(current),
                    "equal": current == reference,
                }
            )
            if current != reference:
                raise RuntimeError(
                    "OpenEVA tokenizer changed a non-added ordinary token sequence."
                )

    return {
        "schema_version": 1,
        "passed": True,
        "token_ids": observed,
        "singleton_encodings": {
            name: list(values) for name, values in encoded_singletons.items()
        },
        "ordinary_text_alignment": ordinary_rows,
        "embedding_resize_required": False,
    }


def select_special_embedding_rows(weight: torch.Tensor) -> torch.Tensor:
    """Copy the eight frozen Qwen rows used to initialize adapter overrides."""

    if weight.ndim != 2 or int(weight.shape[1]) != 2048:
        raise ValueError("Qwen embedding weight must have shape [vocab,2048].")
    if int(weight.shape[0]) <= max(SPECIAL_EMBEDDING_TOKEN_IDS):
        raise ValueError("Qwen embedding vocabulary does not contain the V2 token ABI.")
    indices = torch.tensor(
        SPECIAL_EMBEDDING_TOKEN_IDS,
        device=weight.device,
        dtype=torch.long,
    )
    return weight.index_select(0, indices).detach().clone()


__all__ = [
    "ALL_TASK_TOKEN_IDS",
    "FAMILY_TOKEN_ABI",
    "FAMILY_TOKEN_IDS",
    "MESH_END_TOKEN",
    "MESH_END_TOKEN_ID",
    "MESH_START_TOKEN",
    "MESH_START_TOKEN_ID",
    "POINT_PATCH_TOKEN",
    "POINT_PATCH_TOKEN_ID",
    "SPECIAL_EMBEDDING_ROW_BY_TOKEN_ID",
    "SPECIAL_EMBEDDING_TOKEN_IDS",
    "SPECIAL_EMBEDDING_ABI",
    "select_special_embedding_rows",
    "validate_all_task_tokenizer",
]
