from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    batch, num_src, _ = src.shape
    _, num_dst, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, dim=-1).view(batch, num_src, 1)
    dist += torch.sum(dst**2, dim=-1).view(batch, 1, num_dst)
    return dist


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def _deterministic_fps_seed(xyz: torch.Tensor) -> torch.Tensor:
    centroid = xyz.mean(dim=1, keepdim=True)
    distance_to_centroid = torch.sum((xyz - centroid) ** 2, dim=-1)
    return torch.max(distance_to_centroid, dim=-1)[1]


def fps(xyz: torch.Tensor, npoint: int, *, deterministic: bool = False) -> torch.Tensor:
    device = xyz.device
    batch, num_points, channels = xyz.shape
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch, num_points, device=device) * 1e10
    farthest = _deterministic_fps_seed(xyz) if deterministic else torch.randint(0, num_points, (batch,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch, dtype=torch.long, device=device)
    for idx in range(npoint):
        centroids[:, idx] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch, 1, channels)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return xyz[batch_indices[:, None], centroids]


class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int, *, deterministic_fps: bool = False) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.deterministic_fps = bool(deterministic_fps)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points_xyz = xyz[:, :, :3].contiguous()
        center = fps(points_xyz, self.num_group, deterministic=self.deterministic_fps)
        idx = knn_point(self.group_size, points_xyz, center)
        idx_base = torch.arange(0, xyz.size(0), device=xyz.device).view(-1, 1, 1) * xyz.size(1)
        idx = (idx + idx_base).view(-1)
        grouped = xyz.view(xyz.size(0) * xyz.size(1), -1)[idx, :]
        grouped = grouped.view(xyz.size(0), self.num_group, self.group_size, -1).contiguous()
        grouped[:, :, :, :3] = grouped[:, :, :, :3] - center.unsqueeze(2)
        return grouped, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel: int, point_input_dims: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        batch, groups, samples, dims = point_groups.shape
        point_groups = point_groups.reshape(batch * groups, samples, dims).transpose(2, 1)
        features = self.first_conv(point_groups)
        feature_global = torch.max(features, dim=2, keepdim=True)[0]
        features = torch.cat([feature_global.expand(-1, -1, samples), features], dim=1)
        features = self.second_conv(features)
        feature_global = torch.max(features, dim=2, keepdim=False)[0]
        return feature_global.reshape(batch, groups, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = int(dim) // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_hidden_dim, dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        drop_path_rate: Iterable[float],
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    drop_path=rate,
                )
                for rate in drop_path_rate
            ]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


@dataclass
class MeshUNDEncoderConfig:
    trans_dim: int = 384
    depth: int = 12
    drop_path_rate: float = 0.1
    num_heads: int = 6
    group_size: int = 32
    num_group: int = 512
    encoder_dims: int = 256
    point_dims: int = 6
    use_max_pool: bool = False
    deterministic_fps: bool = False


class MeshUNDEncoder(nn.Module):
    def __init__(self, config: MeshUNDEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.use_max_pool = bool(config.use_max_pool)
        self.group_divider = Group(
            num_group=config.num_group,
            group_size=config.group_size,
            deterministic_fps=config.deterministic_fps,
        )
        self.encoder = Encoder(encoder_channel=config.encoder_dims, point_input_dims=config.point_dims)
        self.reduce_dim = nn.Linear(config.encoder_dims, config.trans_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, config.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, config.trans_dim),
        )
        drop_path = [item.item() for item in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=config.trans_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=drop_path,
        )
        self.norm = nn.LayerNorm(config.trans_dim)

    @property
    def output_token_len(self) -> int:
        return 1 if self.use_max_pool else self.config.num_group + 1

    @property
    def output_dim(self) -> int:
        return self.config.trans_dim * 2 if self.use_max_pool else self.config.trans_dim

    def forward_with_centers(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grouped, center = self.group_divider(pts)
        patch_tokens = self.encoder(grouped)
        patch_tokens = self.reduce_dim(patch_tokens)
        cls_tokens = self.cls_token.expand(patch_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(patch_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, patch_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        if not self.use_max_pool:
            return x, center
        pooled = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1)
        return pooled, center

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.forward_with_centers(pts)
        return tokens

    def load_pretrained_backbone(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        own_state = self.state_dict()
        filtered = dict(state_dict)
        missing = [key for key in own_state.keys() if key not in filtered]
        self.load_state_dict(filtered, strict=False)
        return missing


def build_mesh_und_encoder(use_color: bool, config: MeshUNDEncoderConfig | None = None) -> MeshUNDEncoder:
    resolved = config or MeshUNDEncoderConfig()
    resolved = replace(resolved, point_dims=6 if use_color else 3)
    return MeshUNDEncoder(resolved)


__all__ = ["MeshUNDEncoder", "MeshUNDEncoderConfig", "build_mesh_und_encoder"]
