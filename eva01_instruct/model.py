from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

from .mesh_io import POINT_PATCH_TOKEN
from .point_encoder import PointTransformer, build_point_transformer


MODEL_CONFIG_NAME = "pointllm_stage2_model_config.json"
CONNECTOR_STATE_NAME = "pointllm_stage2_connector.bin"
ADAPTER_DIR_NAME = "pointllm_stage2_lora_adapter"
BACKBONE_ONLY_STATE_NAME = "pointllm_point_backbone_only_v1_2.pt"


@dataclass
class OpenEVAModelConfig:
    base_model_path: str
    pointllm_checkpoint_dir: str
    stage1_checkpoint_dir: Optional[str] = None
    connector_hidden_dims: List[int] = field(default_factory=lambda: [1024, 2048])
    connector_dropout: float = 0.0
    point_patch_token: str = POINT_PATCH_TOKEN
    point_token_len: int = 513
    point_backbone_use_color: bool = True
    freeze_llm: bool = True
    freeze_point_encoder: bool = True
    freeze_connector: bool = False
    llm_trainable_scope: str = "language"
    enable_lora: bool = True
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_num_top_layers: int = 8
    lora_target_modules: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.connector_hidden_dims = [int(dim) for dim in self.connector_hidden_dims]
        if not self.connector_hidden_dims:
            raise ValueError("connector_hidden_dims must not be empty.")
        self.point_token_len = int(self.point_token_len)
        self.point_patch_token = str(self.point_patch_token or POINT_PATCH_TOKEN)
        self.point_backbone_use_color = bool(self.point_backbone_use_color)
        self.enable_lora = bool(self.enable_lora)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_dtype(torch_dtype: Optional[torch.dtype]) -> torch.dtype:
    return torch.float32 if torch_dtype is None else torch_dtype


def _resolve_checkpoint_relative(checkpoint_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if text == ".":
        return checkpoint_dir
    if not path.is_absolute():
        return checkpoint_dir / path
    return path


def _load_backbone_only_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported point backbone checkpoint payload: {checkpoint_path}")
    if any(key.startswith(("model.point_backbone.", "point_backbone.")) for key in state_dict):
        return state_dict
    return {f"point_backbone.{key}": value for key, value in state_dict.items()}


def _load_pointllm_partial_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    if checkpoint_path.is_file():
        return _load_backbone_only_state(checkpoint_path)
    backbone_only_path = checkpoint_path / BACKBONE_ONLY_STATE_NAME
    if backbone_only_path.exists():
        return _load_backbone_only_state(backbone_only_path)
    raise FileNotFoundError(f"Point backbone checkpoint not found: {checkpoint_path}")


def _build_connector(*, point_encoder: PointTransformer, hidden_size: int, config: OpenEVAModelConfig, dtype: Optional[torch.dtype]) -> nn.Module:
    layers: List[nn.Module] = []
    input_dim = point_encoder.output_dim
    for hidden_dim in config.connector_hidden_dims:
        layers.append(nn.Linear(input_dim, int(hidden_dim)))
        layers.append(nn.GELU())
        if float(config.connector_dropout) > 0:
            layers.append(nn.Dropout(float(config.connector_dropout)))
        input_dim = int(hidden_dim)
    layers.append(nn.Linear(input_dim, int(hidden_size)))
    connector = nn.Sequential(*layers)
    if dtype is not None:
        connector.to(dtype=dtype)
    return connector


def select_generated_tokens(sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
    if sequences.ndim != 2:
        raise ValueError(f"Expected generation output to be rank-2, got shape={tuple(sequences.shape)}")
    if sequences.shape[1] > int(prompt_len):
        return sequences[:, int(prompt_len) :]
    return sequences


def checkpoint_has_lora_adapter(checkpoint_dir: str | Path) -> bool:
    return (Path(checkpoint_dir) / ADAPTER_DIR_NAME).exists()


def checkpoint_has_full_weights(checkpoint_dir: str | Path) -> bool:
    checkpoint_path = Path(checkpoint_dir)
    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "pytorch_model.bin.index.json",
    )
    return any(any(checkpoint_path.glob(pattern)) for pattern in patterns)


class OpenEVAPointLLMModel(nn.Module):
    def __init__(
        self,
        *,
        qwen3vl: nn.Module,
        point_encoder: PointTransformer,
        connector: nn.Module,
        model_config: OpenEVAModelConfig,
        point_patch_token_id: int,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.qwen3vl = qwen3vl
        self.point_encoder = point_encoder
        self.connector = connector
        self.model_config = model_config
        self.point_patch_token_id = int(point_patch_token_id)
        self._output_dtype = _resolve_dtype(torch_dtype)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.qwen3vl.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        point_patch_token_id: int,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str | Dict[str, Any]] = None,
    ) -> "OpenEVAPointLLMModel":
        checkpoint_path = Path(checkpoint_dir).expanduser()
        config_path = checkpoint_path / MODEL_CONFIG_NAME
        connector_path = checkpoint_path / CONNECTOR_STATE_NAME
        adapter_dir = checkpoint_path / ADAPTER_DIR_NAME
        if not config_path.exists():
            raise FileNotFoundError(f"OpenEVA model config missing: {config_path}")
        if not connector_path.exists():
            raise FileNotFoundError(f"OpenEVA connector state missing: {connector_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            model_config = OpenEVAModelConfig(**json.load(handle))

        base_model_path = _resolve_checkpoint_relative(checkpoint_path, model_config.base_model_path)
        point_backbone_path = _resolve_checkpoint_relative(checkpoint_path, model_config.pointllm_checkpoint_dir)
        if base_model_path is None:
            base_model_path = checkpoint_path
        if point_backbone_path is None:
            point_backbone_path = checkpoint_path / BACKBONE_ONLY_STATE_NAME

        if model_config.enable_lora:
            if not adapter_dir.exists():
                raise FileNotFoundError(f"LoRA adapter directory missing: {adapter_dir}")
            qwen3vl = Qwen3VLForConditionalGeneration.from_pretrained(
                str(base_model_path),
                dtype=torch_dtype,
                device_map=device_map,
            )
            qwen3vl = PeftModel.from_pretrained(qwen3vl, str(adapter_dir), is_trainable=False)
        else:
            if not checkpoint_has_full_weights(checkpoint_path):
                raise FileNotFoundError(f"Full OpenEVA weights missing under {checkpoint_path}")
            qwen3vl = Qwen3VLForConditionalGeneration.from_pretrained(
                str(checkpoint_path),
                dtype=torch_dtype,
                device_map=device_map,
            )

        qwen_config = qwen3vl.base_model.model.config if isinstance(qwen3vl, PeftModel) else qwen3vl.config
        if getattr(qwen_config, "use_cache", True):
            qwen_config.use_cache = False

        point_encoder = build_point_transformer(use_color=model_config.point_backbone_use_color)
        partial_state = _load_pointllm_partial_state(point_backbone_path)
        point_encoder.load_pointllm_checkpoint(partial_state)

        hidden_size = int(qwen_config.text_config.hidden_size)
        connector = _build_connector(
            point_encoder=point_encoder,
            hidden_size=hidden_size,
            config=model_config,
            dtype=torch_dtype,
        )
        connector_state = torch.load(connector_path, map_location="cpu")
        connector.load_state_dict(connector_state, strict=True)
        if torch_dtype is not None:
            connector.to(dtype=torch_dtype)

        return cls(
            qwen3vl=qwen3vl,
            point_encoder=point_encoder,
            connector=connector,
            model_config=model_config,
            point_patch_token_id=point_patch_token_id,
            torch_dtype=torch_dtype,
        )

    def resize_token_embeddings_if_needed(self, tokenizer_length: int) -> None:
        embeddings = self.qwen3vl.get_input_embeddings()
        if int(tokenizer_length) > int(embeddings.num_embeddings):
            self.qwen3vl.resize_token_embeddings(int(tokenizer_length))

    def _encode_point_tokens(self, point_clouds: torch.Tensor) -> torch.Tensor:
        point_clouds = point_clouds.to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            point_tokens = self.point_encoder(point_clouds)
        point_tokens = point_tokens.to(device=self.device, dtype=self._output_dtype)
        projected = self.connector(point_tokens)
        return projected.to(dtype=self._output_dtype)

    def build_inputs_embeds(self, input_ids: torch.Tensor, point_clouds: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(device=self.device)
        input_embeds = self.qwen3vl.get_input_embeddings()(input_ids)
        point_embeds = self._encode_point_tokens(point_clouds)
        point_mask = input_ids.eq(self.point_patch_token_id)
        expected_tokens = point_embeds.shape[0] * point_embeds.shape[1]
        actual_tokens = int(point_mask.sum().item())
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} point patch tokens, but found {actual_tokens}. "
                "Check the point patch token and prompt template."
            )
        input_embeds = input_embeds.clone()
        input_embeds[point_mask] = point_embeds.reshape(-1, point_embeds.shape[-1])
        return input_embeds

    @torch.inference_mode()
    def generate_from_point_clouds(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        point_clouds: torch.Tensor,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        self.eval()
        inputs_embeds = self.build_inputs_embeds(input_ids=input_ids, point_clouds=point_clouds)
        return self.qwen3vl.generate(
            input_ids=input_ids.to(device=self.device),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(device=self.device),
            **generate_kwargs,
        )


__all__ = [
    "ADAPTER_DIR_NAME",
    "CONNECTOR_STATE_NAME",
    "MODEL_CONFIG_NAME",
    "OpenEVAModelConfig",
    "OpenEVAPointLLMModel",
    "checkpoint_has_full_weights",
    "checkpoint_has_lora_adapter",
    "select_generated_tokens",
]
