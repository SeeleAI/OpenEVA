from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from .conversation import DEFAULT_QUESTION, build_mesh_chat_messages
from .mesh_io import POINT_PATCH_TOKEN, SUPPORTED_GEOMETRY_SUFFIXES, build_point_patch_text, load_external_point_cloud
from .model import MODEL_CONFIG_NAME, OpenEVAPointLLMModel, select_generated_tokens


@dataclass
class MeshInput:
    input_path: Path
    object_id: str
    point_cloud: np.ndarray
    pointnum: int


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    value = str(dtype_name or "auto").strip().lower()
    if value == "auto":
        return None
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def load_model_config(checkpoint_dir: str | Path) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_dir).expanduser()
    config_path = checkpoint_path / MODEL_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"OpenEVA model config missing: {config_path}")
    import json

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict in {config_path}")
    return payload


class OpenEVARuntime:
    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        pointnum: int = 8192,
        max_length: int = 2048,
        default_question: str = DEFAULT_QUESTION,
        device_map: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = resolve_torch_dtype(torch_dtype)
        self.pointnum = int(pointnum)
        self.max_length = int(max_length)
        self.default_question = str(default_question or DEFAULT_QUESTION)
        self.model_config = load_model_config(self.checkpoint_dir)
        self.point_patch_token = str(self.model_config.get("point_patch_token") or POINT_PATCH_TOKEN)
        self.point_token_len = int(self.model_config.get("point_token_len") or 513)
        self.use_color = bool(self.model_config.get("point_backbone_use_color", True))
        self.point_patch_text = build_point_patch_text(self.point_patch_token, self.point_token_len)

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir, use_fast=True)
        if self.point_patch_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.point_patch_token], special_tokens=True)
        self.point_patch_token_id = int(self.tokenizer.convert_tokens_to_ids(self.point_patch_token))

        load_device_map = None if not device_map or device_map == "none" else device_map
        self.model = OpenEVAPointLLMModel.from_checkpoint(
            self.checkpoint_dir,
            point_patch_token_id=self.point_patch_token_id,
            torch_dtype=self.dtype,
            device_map=load_device_map,
        )
        self.model.resize_token_embeddings_if_needed(len(self.tokenizer))
        if load_device_map is None:
            self.model.to(device=self.device)
        self.model.eval()

    def load_mesh_or_point_cloud(
        self,
        input_path: str | Path,
        *,
        object_id: str | None = None,
        pointnum: int | None = None,
    ) -> MeshInput:
        path = Path(input_path).expanduser()
        if path.suffix.lower() not in SUPPORTED_GEOMETRY_SUFFIXES:
            raise ValueError(
                f"Unsupported input suffix {path.suffix!r}. Supported suffixes: {', '.join(SUPPORTED_GEOMETRY_SUFFIXES)}"
            )
        count = int(pointnum or self.pointnum)
        point_cloud = load_external_point_cloud(
            path,
            pointnum=count,
            use_color=self.use_color,
            deterministic_key=object_id or path.stem,
        )
        return MeshInput(
            input_path=path,
            object_id=object_id or path.stem,
            point_cloud=point_cloud,
            pointnum=count,
        )

    @torch.inference_mode()
    def answer(
        self,
        *,
        mesh_input: MeshInput,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_history_turns: int = 4,
    ) -> tuple[str, List[Dict[str, str]]]:
        text = str(question or "").strip() or self.default_question
        chat_history = list(history or [])
        chat_history.append({"role": "user", "content": text})
        messages = build_mesh_chat_messages(
            chat_history=chat_history,
            point_patch_text=self.point_patch_text,
            max_history_turns=int(max_history_turns),
        )
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)
        point_clouds = torch.from_numpy(mesh_input.point_cloud.astype(np.float32)).unsqueeze(0).to(self.model.device)
        do_sample = float(temperature) > 0.0
        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        sequences = self.model.generate_from_point_clouds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            **generate_kwargs,
        )
        generated = select_generated_tokens(sequences, prompt_len=input_ids.shape[1])[0]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        chat_history.append({"role": "assistant", "content": response})
        return response, chat_history


__all__ = ["MeshInput", "OpenEVARuntime", "load_model_config", "resolve_torch_dtype"]
