from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging as transformers_logging

from .mesh_und_io import MESH_UND_TOKEN, build_mesh_und_token_text, load_mesh_und_values


PROCESSOR_CONFIG_NAME = "processor_config.json"
LEGACY_PROCESSOR_CONFIG_NAME = "eva01_processor_config.json"


def _load_tokenizer_quietly(source: str | Path, **kwargs: Any):
    verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        return AutoTokenizer.from_pretrained(source, use_fast=True, **kwargs)
    finally:
        transformers_logging.set_verbosity(verbosity)


class EVA01Processor:
    model_input_names = ["input_ids", "attention_mask", "mesh_und_values"]

    def __init__(
        self,
        tokenizer,
        *,
        mesh_und_token: str = MESH_UND_TOKEN,
        mesh_und_token_len: int = 513,
        mesh_und_pointnum: int = 8192,
        mesh_und_use_color: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.mesh_und_token = str(mesh_und_token)
        self.mesh_und_token_len = int(mesh_und_token_len)
        self.mesh_und_pointnum = int(mesh_und_pointnum)
        self.mesh_und_use_color = bool(mesh_und_use_color)
        if self.mesh_und_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.mesh_und_token], special_tokens=True)
        self.mesh_und_token_id = int(self.tokenizer.convert_tokens_to_ids(self.mesh_und_token))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any) -> "EVA01Processor":
        path = Path(pretrained_model_name_or_path).expanduser()
        config: dict[str, Any] = {}
        tokenizer_source: str | Path = pretrained_model_name_or_path
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(pretrained_model_name_or_path), allow_patterns=["*.json", "*.jinja", "vocab.json", "merges.txt"]))
            tokenizer_source = path
        config_path = path / PROCESSOR_CONFIG_NAME
        if not config_path.exists():
            config_path = path / LEGACY_PROCESSOR_CONFIG_NAME
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer_source = path if (path / "tokenizer_config.json").exists() else tokenizer_source
        tokenizer = _load_tokenizer_quietly(tokenizer_source, **kwargs)
        return cls(
            tokenizer,
            mesh_und_token=config.get("mesh_und_token", MESH_UND_TOKEN),
            mesh_und_token_len=int(config.get("mesh_und_token_len", 513)),
            mesh_und_pointnum=int(config.get("mesh_und_pointnum", 8192)),
            mesh_und_use_color=bool(config.get("mesh_und_use_color", True)),
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(target)
        payload = {
            "processor_class": self.__class__.__name__,
            "mesh_und_token": self.mesh_und_token,
            "mesh_und_token_id": self.mesh_und_token_id,
            "mesh_und_token_len": self.mesh_und_token_len,
            "mesh_und_pointnum": self.mesh_und_pointnum,
            "mesh_und_use_color": self.mesh_und_use_color,
        }
        (target / PROCESSOR_CONFIG_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _normalize_content(self, content: Any, mesh_paths: list[str]) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = str(item.get("type", "text"))
            if item_type == "mesh":
                mesh_path = str(item.get("mesh") or item.get("path") or "").strip()
                if not mesh_path:
                    raise ValueError("Mesh content must include a non-empty 'mesh' path.")
                mesh_paths.append(mesh_path)
                parts.append(build_mesh_und_token_text(self.mesh_und_token, self.mesh_und_token_len))
            elif item_type == "text":
                parts.append(str(item.get("text") or ""))
            else:
                raise ValueError(f"Unsupported EVA01 message content type: {item_type!r}")
        return "\n".join(part for part in parts if part.strip())

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
        mesh_paths: list[str] = []
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = self._normalize_content(message.get("content", ""), mesh_paths)
            normalized.append({"role": role, "content": content})
        return normalized, mesh_paths

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        return_dict: bool = False,
        return_tensors: str | None = None,
        **tokenizer_kwargs: Any,
    ) -> str | BatchFeature:
        normalized, mesh_paths = self._normalize_messages(messages)
        rendered = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return rendered
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        data = dict(encoded)
        if mesh_paths:
            values = [
                load_mesh_und_values(
                    mesh_path,
                    pointnum=self.mesh_und_pointnum,
                    use_color=self.mesh_und_use_color,
                    deterministic_key=Path(mesh_path).stem,
                )
                for mesh_path in mesh_paths
            ]
            array = np.stack(values, axis=0).astype(np.float32)
            data["mesh_und_values"] = torch.from_numpy(array) if return_tensors == "pt" else array
        feature = BatchFeature(data=data, tensor_type=return_tensors)
        if return_dict:
            return feature
        return feature["input_ids"]

    def __call__(
        self,
        *,
        text: str,
        meshes: str | Path | list[str | Path],
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> BatchFeature:
        mesh_items = [meshes] if isinstance(meshes, (str, Path)) else list(meshes)
        if len(mesh_items) != 1:
            raise ValueError("EVA01 currently expects one mesh per prompt.")
        messages = [{"role": "user", "content": [{"type": "mesh", "mesh": str(mesh_items[0])}, {"type": "text", "text": text}]}]
        return self.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors=return_tensors,
            **kwargs,
        )

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode(*args, **kwargs)
