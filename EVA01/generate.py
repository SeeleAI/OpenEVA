#!/usr/bin/env python3
"""Run the public EVA01 Unified MoT text-to-3D or image-to-3D cascade."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eva01.mesh_runtime import load_decoder
from eva01.unified_generation import EVA01UnifiedGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an EVA01 3D asset and optionally export a textured GLB."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Text condition for text-to-3D generation.")
    source.add_argument("--image", type=Path, help="Input image for image-to-3D generation.")
    parser.add_argument("--checkpoint", required=True, help="Generation-only adapter or LoRA export.")
    parser.add_argument("--tokenizer-root", required=True, help="Unified All-Task tokenizer directory.")
    parser.add_argument("--processor-root", default=None, help="Qwen image processor directory; defaults to the Qwen base directory.")
    parser.add_argument("--qwen-root", required=True, help="Local Qwen3-VL-2B base directory.")
    parser.add_argument("--ss-xl-stem", required=True, help="TRELLIS1 SS flow base directory.")
    parser.add_argument("--shape-stem", required=True, help="TRELLIS2 Shape-512 flow base directory.")
    parser.add_argument("--material-stem", required=True, help="TRELLIS2 Material-512 flow base directory.")
    parser.add_argument("--normalization-config", required=True, help="TRELLIS2 pipeline.json used for latent normalization.")
    parser.add_argument("--ss-decoder-stem", required=True, help="TRELLIS1 SS decoder stem without a file extension.")
    parser.add_argument(
        "--shape-decoder-stem",
        default=None,
        help="External TRELLIS.2 Shape decoder stem; required with --output-glb.",
    )
    parser.add_argument(
        "--material-decoder-stem",
        default=None,
        help="External TRELLIS.2 Material decoder stem; required with --output-glb.",
    )
    parser.add_argument(
        "--mesh-resolution",
        type=int,
        default=512,
        choices=(512, 1024, 1536),
        help="TRELLIS.2 decoder resolution used for GLB export.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file for native generation latents.")
    parser.add_argument("--output-glb", type=Path, default=None, help="Optional textured GLB output path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_ss_decoder(stem: str, device: torch.device) -> torch.nn.Module:
    return load_decoder(stem, label="SS", device=device)


def main() -> None:
    args = parse_args()
    if args.output_glb is not None and (args.shape_decoder_stem is None or args.material_decoder_stem is None):
        raise ValueError("--output-glb requires both --shape-decoder-stem and --material-decoder-stem.")
    device = torch.device(args.device)
    runtime = EVA01UnifiedGeneration.from_pretrained(
        args.checkpoint,
        tokenizer_root=args.tokenizer_root,
        processor_root=args.processor_root,
        qwen_root=args.qwen_root,
        ss_xl_stem=args.ss_xl_stem,
        shape_stem=args.shape_stem,
        material_stem=args.material_stem,
        normalization_config=args.normalization_config,
        device=device,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
    )
    decoder = _load_ss_decoder(args.ss_decoder_stem, device)
    result = runtime.generate(
        prompt=args.prompt or "",
        image=args.image,
        ss_decoder=decoder,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ss=result.ss.detach().cpu().numpy(),
        coords=result.coords.detach().cpu().numpy(),
        shape_coords=result.shape.coords[:, 1:].detach().cpu().numpy(),
        shape_feats=result.shape.feats.detach().cpu().numpy(),
        material_coords=result.material.coords[:, 1:].detach().cpu().numpy(),
        material_feats=result.material.feats.detach().cpu().numpy(),
        metadata=json.dumps(
            {
                "mode": result.mode,
                "cascade": ["ss", "shape", "material"],
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "traces": {
                    family: asdict(trace)
                    for family, trace in result.traces.items()
                },
            },
            sort_keys=True,
        ),
    )
    if args.output_glb is not None:
        from eva01.mesh_runtime import decode_mesh_with_material, export_mesh_glb, load_decoder

        # The Unified MoT weights are no longer needed while the external
        # TRELLIS decoders are resident, which keeps the public CLI bounded.
        del decoder
        runtime.model.cpu()
        torch.cuda.empty_cache()
        shape_decoder = load_decoder(
            args.shape_decoder_stem,
            label="Shape",
            device=device,
        )
        material_decoder = load_decoder(
            args.material_decoder_stem,
            label="Material",
            device=device,
        )
        mesh = decode_mesh_with_material(
            result.shape,
            result.material,
            shape_decoder=shape_decoder,
            material_decoder=material_decoder,
            resolution=args.mesh_resolution,
        )
        vertices = mesh.vertices
        faces = mesh.faces
        if (
            vertices.ndim != 2
            or vertices.shape[-1] != 3
            or faces.ndim != 2
            or faces.shape[-1] != 3
            or vertices.shape[0] == 0
            or faces.shape[0] == 0
            or not torch.isfinite(vertices).all()
            or not torch.isfinite(faces.to(dtype=torch.float32)).all()
            or int(faces.min()) < 0
            or int(faces.max()) >= int(vertices.shape[0])
        ):
            raise RuntimeError("Decoded mesh failed finite, non-empty, or index validation.")
        export_mesh_glb(mesh, args.output_glb)
    print(json.dumps({"output": str(args.output.resolve()), "mode": result.mode}, indent=2))


if __name__ == "__main__":
    main()
