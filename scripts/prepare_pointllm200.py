#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local folders for PointLLM-200 caption evaluation.")
    parser.add_argument("--data-root", default="data/pointllm200")
    parser.add_argument("--hf-repo", default="YOUR_ORG/PointLLM-200")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    annotation_dir = data_root / "annotations"
    pointcloud_dir = data_root / "pointclouds" / "8192_npy"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    pointcloud_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        if args.hf_repo.startswith("YOUR_ORG/"):
            raise RuntimeError("The default hf repo is a placeholder. Replace --hf-repo with the released dataset repo.")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=args.hf_repo, local_dir=data_root, local_dir_use_symlinks=False, token=args.token)

    print("PointLLM-200 local layout:")
    print(f"  annotations: {annotation_dir}")
    print(f"  point clouds: {pointcloud_dir}")
    print("Expected annotation file: PointLLM_brief_description_val_200_GT.json")
    print("Expected point cloud files: <object_id>_8192.npy")


if __name__ == "__main__":
    main()
