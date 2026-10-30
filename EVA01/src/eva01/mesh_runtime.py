"""Inference-only TRELLIS.2 decoding and GLB export for EVA01."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


PBR_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


def _require_checkpoint_stem(stem: str | Path, label: str) -> Path:
    """Resolve a decoder stem and fail closed when either sidecar is absent."""
    path = Path(stem).expanduser().resolve()
    if not Path(f"{path}.json").is_file() or not Path(f"{path}.safetensors").is_file():
        raise FileNotFoundError(
            f"{label} decoder requires both {path}.json and {path}.safetensors; "
            "decoder weights are external TRELLIS assets and are not bundled with EVA01."
        )
    return path


def load_decoder(stem: str | Path, *, label: str, device: torch.device) -> torch.nn.Module:
    """Load one external TRELLIS decoder without an implicit fallback."""
    decoder_stem = _require_checkpoint_stem(stem, label)
    from diffusionstudio.pipelines.eva01.qwen3_mot.trellis2.models import from_pretrained

    decoder = from_pretrained(str(decoder_stem))
    return decoder.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def decode_mesh_with_material(
    shape_latent: Any,
    material_latent: Any,
    *,
    shape_decoder: torch.nn.Module,
    material_decoder: torch.nn.Module,
    resolution: int = 512,
):
    """Decode native Shape/Material latents into a textured MeshWithVoxel."""
    shape_decoder.set_resolution(int(resolution))
    meshes, subdivision = shape_decoder(shape_latent, return_subs=True)
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one Shape mesh, received {len(meshes)}.")

    runtime_param = next(material_decoder.parameters())
    material_input = material_latent.to(
        device=runtime_param.device,
        dtype=runtime_param.dtype if torch.is_floating_point(runtime_param) else material_latent.feats.dtype,
    )
    material_voxels = material_decoder(material_input, guide_subs=subdivision) * 0.5 + 0.5
    if len(material_voxels) != 1:
        raise RuntimeError(f"Expected one Material voxel field, received {len(material_voxels)}.")
    voxel = material_voxels[0]
    if voxel.feats.ndim != 2 or voxel.feats.shape[-1] < 6:
        raise RuntimeError("Material decoder must return six PBR channels.")
    from diffusionstudio.pipelines.eva01.qwen3_mot.trellis2.representations import MeshWithVoxel

    mesh = meshes[0]
    return MeshWithVoxel(
        mesh.vertices,
        mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / float(resolution),
        coords=voxel.coords[:, 1:],
        attrs=voxel.feats,
        voxel_shape=torch.Size([*voxel.shape, *voxel.spatial_shape]),
        layout=PBR_ATTR_LAYOUT,
    )


def export_mesh_glb(mesh: Any, output_path: str | Path) -> None:
    """Export a decoded MeshWithVoxel as a valid geometry-and-color GLB.

    The export deliberately avoids topology repair and remeshing.  Those
    operations are optional presentation postprocessing and can fail on very
    large sparse meshes; the public API must preserve the decoded geometry.
    Material base-color attributes are written as vertex colors when the
    decoder returns one attribute row per vertex, otherwise their mean color
    is used as a deterministic preview color.
    """
    import numpy as np
    import trimesh

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise RuntimeError("Decoded mesh vertices must have shape [N, 3].")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise RuntimeError("Decoded mesh faces must have shape [M, 3].")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise RuntimeError("Decoded mesh is empty.")
    if not np.isfinite(vertices).all():
        raise RuntimeError("Decoded mesh vertices contain non-finite values.")
    if int(faces.min()) < 0 or int(faces.max()) >= int(vertices.shape[0]):
        raise RuntimeError("Decoded mesh faces contain an out-of-range vertex index.")

    color = torch.ones((vertices.shape[0], 3), dtype=torch.float32)
    attrs = mesh.attrs.detach().to(device="cpu", dtype=torch.float32)
    base_color = attrs[:, mesh.layout["base_color"]]
    if int(base_color.shape[0]) == int(vertices.shape[0]):
        color = base_color[:, :3]
    elif int(base_color.shape[0]) > 0:
        color = base_color[:, :3].mean(dim=0, keepdim=True).expand(vertices.shape[0], -1)
    colors = (color.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    colors = np.concatenate([colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False).export(output)


__all__ = ["decode_mesh_with_material", "export_mesh_glb", "load_decoder"]
