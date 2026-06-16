from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


def _load_qwen3vl_config_class():
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    except Exception as exc:
        raise ImportError(
            "EVA01 requires a transformers build that provides Qwen3-VL. "
            "Install the EVA01 package with `bash EVA01/install.sh` or use transformers>=4.57.0."
        ) from exc
    return Qwen3VLConfig


class EVA01Config(PretrainedConfig):
    model_type = "eva01"

    def __init__(
        self,
        *,
        qwen3vl_config: dict[str, Any] | "Qwen3VLConfig" | None = None,
        base_model_name_or_path: str | None = None,
        is_lora: bool = False,
        mesh_und_token: str = "<|mesh_und_pad|>",
        mesh_und_token_id: int | None = None,
        mesh_und_token_len: int = 513,
        mesh_und_pointnum: int = 8192,
        mesh_und_use_color: bool = True,
        mesh_und_encoder_type: str = "frozen_mesh_und_transformer",
        mesh_und_encoder_file: str = "mesh_und_encoder.safetensors",
        mesh_und_connector_file: str = "mesh_und_connector.safetensors",
        mesh_und_connector_hidden_dims: list[int] | None = None,
        mesh_und_connector_dropout: float = 0.0,
        training_recipe: str = "alignment_then_instruction_tuning",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if qwen3vl_config is None:
            qwen3vl_config = _load_qwen3vl_config_class()().to_dict()
        elif not isinstance(qwen3vl_config, dict):
            qwen3vl_config = qwen3vl_config.to_dict()
        self.qwen3vl_config = dict(qwen3vl_config)
        self.base_model_name_or_path = base_model_name_or_path
        self.is_lora = bool(is_lora)
        self.mesh_und_token = str(mesh_und_token)
        self.mesh_und_token_id = None if mesh_und_token_id is None else int(mesh_und_token_id)
        self.mesh_und_token_len = int(mesh_und_token_len)
        self.mesh_und_pointnum = int(mesh_und_pointnum)
        self.mesh_und_use_color = bool(mesh_und_use_color)
        self.mesh_und_encoder_type = str(mesh_und_encoder_type)
        self.mesh_und_encoder_file = str(mesh_und_encoder_file)
        self.mesh_und_connector_file = str(mesh_und_connector_file)
        self.mesh_und_connector_hidden_dims = [int(dim) for dim in (mesh_und_connector_hidden_dims or [1024, 2048])]
        self.mesh_und_connector_dropout = float(mesh_und_connector_dropout)
        self.training_recipe = str(training_recipe)

    def to_qwen3vl_config(self) -> "Qwen3VLConfig":
        return _load_qwen3vl_config_class().from_dict(dict(self.qwen3vl_config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any) -> "EVA01Config":
        source = Path(pretrained_model_name_or_path).expanduser()
        if source.is_file():
            config_path = source
        elif source.exists():
            config_path = source / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            download_kwargs = {
                key: kwargs[key]
                for key in ("cache_dir", "force_download", "local_files_only", "revision", "token")
                if key in kwargs
            }
            config_path = Path(hf_hub_download(str(pretrained_model_name_or_path), "config.json", **download_kwargs))
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(**config_dict)
