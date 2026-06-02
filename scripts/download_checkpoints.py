#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPOS = {
    "lora": "YOUR_ORG/OpenEVA-Instruct-LoRA",
    "full": "YOUR_ORG/OpenEVA-Instruct-Full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenEVA-Instruct checkpoint bundles from Hugging Face.")
    parser.add_argument("--variant", choices=["lora", "full", "all"], default="lora")
    parser.add_argument("--lora-repo", default=DEFAULT_REPOS["lora"])
    parser.add_argument("--full-repo", default=DEFAULT_REPOS["full"])
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def download(repo_id: str, target_dir: Path, token: str | None) -> None:
    if repo_id.startswith("YOUR_ORG/"):
        raise RuntimeError(
            f"{repo_id} is a placeholder. Replace it with the released Hugging Face repository before downloading."
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.checkpoint_root)
    jobs = []
    if args.variant in {"lora", "all"}:
        jobs.append((args.lora_repo, root / "openeva-instruct-lora"))
    if args.variant in {"full", "all"}:
        jobs.append((args.full_repo, root / "openeva-instruct-full"))
    for repo_id, target in jobs:
        print(f"Downloading {repo_id} -> {target}")
        download(repo_id, target, args.token)


if __name__ == "__main__":
    main()
