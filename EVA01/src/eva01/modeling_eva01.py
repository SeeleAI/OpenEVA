from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, PreTrainedModel
from transformers.utils import logging as transformers_logging

from .configuration_eva01 import EVA01Config
from .mesh_und_encoder import MeshUNDEncoder, build_mesh_und_encoder
from .mesh_und_io import MESH_UND_TOKEN


def _resolve_local_path(pretrained_model_name_or_path: str | Path) -> Path:
    path = Path(pretrained_model_name_or_path).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(pretrained_model_name_or_path)))


def _load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return dict(payload["state_dict"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Unsupported tensor payload: {path}")


def _load_qwen3vl_model_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise ImportError(
            "EVA01 requires a transformers build that provides Qwen3-VL. "
            "Install the EVA01 package with `bash EVA01/install.sh` or use transformers>=4.57.0."
        ) from exc
    return Qwen3VLForConditionalGeneration


def _load_tokenizer_quietly(source: str | Path, **kwargs: Any):
    verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        return AutoTokenizer.from_pretrained(source, use_fast=True, **kwargs)
    finally:
        transformers_logging.set_verbosity(verbosity)


def _resolve_torch_dtype(value: Any) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    text = str(value).lower()
    if text in {"auto", "none"}:
        return None
    if text in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if text in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value!r}")


def _build_connector(*, input_dim: int, hidden_size: int, config: EVA01Config, dtype: torch.dtype | None) -> nn.Module:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in config.mesh_und_connector_hidden_dims:
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.GELU())
        if config.mesh_und_connector_dropout > 0:
            layers.append(nn.Dropout(config.mesh_und_connector_dropout))
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, int(hidden_size)))
    connector = nn.Sequential(*layers)
    if dtype is not None:
        connector.to(dtype=dtype)
    return connector


def _get_qwen_config(qwen3vl: nn.Module):
    if hasattr(qwen3vl, "peft_config"):
        return qwen3vl.base_model.model.config
    return qwen3vl.config


class EVA01ForConditionalGeneration(PreTrainedModel):
    config_class = EVA01Config
    base_model_prefix = "eva01"
    supports_gradient_checkpointing = False

    def __init__(
        self,
        config: EVA01Config,
        *,
        qwen3vl: nn.Module | None = None,
        mesh_und_encoder: MeshUNDEncoder | None = None,
        mesh_und_connector: nn.Module | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config)
        self.qwen3vl = qwen3vl or _load_qwen3vl_model_class()(config.to_qwen3vl_config())
        qwen_config = _get_qwen_config(self.qwen3vl)
        if getattr(qwen_config, "use_cache", True):
            qwen_config.use_cache = False
        self.mesh_und_encoder = mesh_und_encoder or build_mesh_und_encoder(use_color=config.mesh_und_use_color)
        hidden_size = int(qwen_config.text_config.hidden_size)
        self.mesh_und_connector = mesh_und_connector or _build_connector(
            input_dim=self.mesh_und_encoder.output_dim,
            hidden_size=hidden_size,
            config=config,
            dtype=torch_dtype,
        )
        self._output_dtype = torch.float32 if torch_dtype is None else torch_dtype
        self.mesh_und_encoder.to(device=self.device)
        self.mesh_und_connector.to(device=self.device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.qwen3vl.parameters()).device

    def get_input_embeddings(self) -> nn.Module:
        return self.qwen3vl.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.qwen3vl.set_input_embeddings(value)

    def resize_token_embeddings_if_needed(self, tokenizer_length: int) -> None:
        embeddings = self.get_input_embeddings()
        if int(tokenizer_length) > int(embeddings.num_embeddings):
            self.qwen3vl.resize_token_embeddings(int(tokenizer_length))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *model_args: Any, **kwargs: Any) -> "EVA01ForConditionalGeneration":
        torch_dtype = _resolve_torch_dtype(kwargs.pop("torch_dtype", kwargs.pop("dtype", None)))
        device_map = kwargs.pop("device_map", None)
        base_model_name_or_path = kwargs.pop("base_model_name_or_path", None)
        path = _resolve_local_path(pretrained_model_name_or_path)
        config = kwargs.pop("config", None) or EVA01Config.from_pretrained(path)
        if not isinstance(config, EVA01Config):
            config = EVA01Config.from_dict(config.to_dict())

        if config.mesh_und_token_id is None:
            try:
                tokenizer = _load_tokenizer_quietly(path)
                if config.mesh_und_token not in tokenizer.get_vocab():
                    tokenizer.add_tokens([config.mesh_und_token], special_tokens=True)
                config.mesh_und_token_id = int(tokenizer.convert_tokens_to_ids(config.mesh_und_token))
            except Exception:
                raise ValueError("mesh_und_token_id is missing and could not be resolved from the checkpoint tokenizer.")

        if config.is_lora:
            from peft import PeftModel

            base_source = base_model_name_or_path or config.base_model_name_or_path
            if not base_source:
                raise ValueError("LoRA EVA01 checkpoints require base_model_name_or_path.")
            qwen3vl = _load_qwen3vl_model_class().from_pretrained(
                str(base_source),
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            qwen3vl = PeftModel.from_pretrained(qwen3vl, str(path), is_trainable=False)
        else:
            qwen_dir = path / "qwen3vl"
            qwen_source = qwen_dir if qwen_dir.exists() else path
            qwen3vl = _load_qwen3vl_model_class().from_pretrained(
                str(qwen_source),
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

        qwen_config = _get_qwen_config(qwen3vl)
        encoder = build_mesh_und_encoder(use_color=config.mesh_und_use_color)
        encoder_state = _load_tensor_file(path / config.mesh_und_encoder_file)
        encoder.load_pretrained_backbone(encoder_state)
        hidden_size = int(qwen_config.text_config.hidden_size)
        connector = _build_connector(input_dim=encoder.output_dim, hidden_size=hidden_size, config=config, dtype=torch_dtype)
        connector.load_state_dict(_load_tensor_file(path / config.mesh_und_connector_file), strict=True)
        model = cls(
            config,
            qwen3vl=qwen3vl,
            mesh_und_encoder=encoder,
            mesh_und_connector=connector,
            torch_dtype=torch_dtype,
        )
        try:
            tokenizer_length = len(_load_tokenizer_quietly(path))
            model.resize_token_embeddings_if_needed(tokenizer_length)
        except Exception:
            pass
        return model

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(target)
        if self.config.is_lora:
            self.qwen3vl.save_pretrained(target, **kwargs)
        else:
            self.qwen3vl.save_pretrained(target / "qwen3vl", **kwargs)
        save_file(self.mesh_und_encoder.state_dict(), target / self.config.mesh_und_encoder_file)
        save_file(self.mesh_und_connector.state_dict(), target / self.config.mesh_und_connector_file)

    def save_mesh_und_pretrained(self, save_directory: str | Path) -> None:
        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        save_file(self.mesh_und_encoder.state_dict(), target / self.config.mesh_und_encoder_file)
        save_file(self.mesh_und_connector.state_dict(), target / self.config.mesh_und_connector_file)

    def _encode_mesh_und(self, mesh_und_values: torch.Tensor) -> torch.Tensor:
        values = mesh_und_values.to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            tokens = self.mesh_und_encoder(values)
        tokens = tokens.to(device=self.device, dtype=self._output_dtype)
        projected = self.mesh_und_connector(tokens)
        return projected.to(dtype=self._output_dtype)

    def build_inputs_embeds(self, input_ids: torch.Tensor, mesh_und_values: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(device=self.device)
        input_embeds = self.get_input_embeddings()(input_ids)
        mesh_embeds = self._encode_mesh_und(mesh_und_values)
        mesh_mask = input_ids.eq(int(self.config.mesh_und_token_id))
        expected_tokens = mesh_embeds.shape[0] * mesh_embeds.shape[1]
        actual_tokens = int(mesh_mask.sum().item())
        if actual_tokens != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} mesh UND tokens, found {actual_tokens}.")
        input_embeds = input_embeds.clone()
        input_embeds[mesh_mask] = mesh_embeds.reshape(-1, mesh_embeds.shape[-1])
        return input_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mesh_und_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if inputs_embeds is None and mesh_und_values is not None:
            if input_ids is None:
                raise ValueError("input_ids are required when mesh_und_values are provided.")
            inputs_embeds = self.build_inputs_embeds(input_ids=input_ids, mesh_und_values=mesh_und_values)
        qwen_input_ids = None if inputs_embeds is not None else (input_ids.to(self.device) if input_ids is not None else None)
        return self.qwen3vl(
            input_ids=qwen_input_ids,
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mesh_und_values: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        self.eval()
        if mesh_und_values is None:
            return self.qwen3vl.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                **generate_kwargs,
            )
        inputs_embeds = self.build_inputs_embeds(input_ids=input_ids, mesh_und_values=mesh_und_values)
        return self.qwen3vl.generate(
            input_ids=input_ids.to(self.device),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            **generate_kwargs,
        )
