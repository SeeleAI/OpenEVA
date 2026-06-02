from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PointLLM200CaptionDataset(Dataset):
    def __init__(
        self,
        *,
        annotations: Sequence[Dict[str, Any]],
        pointcloud_dir: str | Path,
        pointnum: int = 8192,
    ) -> None:
        self.annotations = [dict(item) for item in annotations]
        self.pointcloud_dir = Path(pointcloud_dir).expanduser()
        self.pointnum = int(pointnum)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.annotations[index]
        object_id = str(item.get("object_id", "")).strip()
        conversations = list(item.get("conversations") or [])
        if len(conversations) < 2:
            raise ValueError(f"Invalid eval annotation for {object_id}: expected at least two turns.")
        answer_text = str(conversations[1].get("value", "")).strip()
        pointcloud_path = self.pointcloud_dir / f"{object_id}_{self.pointnum}.npy"
        if not pointcloud_path.exists():
            pointcloud_path = self.pointcloud_dir / f"{object_id}.npy"
        if not pointcloud_path.exists():
            raise FileNotFoundError(pointcloud_path)
        point_cloud = np.load(pointcloud_path).astype(np.float32)
        return {
            "object_id": object_id,
            "ground_truth": answer_text,
            "point_clouds": torch.from_numpy(point_cloud),
        }


def collate_pointllm200(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "object_ids": [item["object_id"] for item in items],
        "ground_truths": [item["ground_truth"] for item in items],
        "point_clouds": torch.stack([item["point_clouds"] for item in items], dim=0).to(torch.float32),
    }


__all__ = ["PointLLM200CaptionDataset", "collate_pointllm200"]
