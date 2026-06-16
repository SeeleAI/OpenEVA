from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np


MESH_UND_TOKEN = "<|mesh_und_pad|>"
SUPPORTED_GEOMETRY_SUFFIXES = (".glb", ".npy")
NEUTRAL_RGB_VALUE = 0.5


def build_mesh_und_token_text(mesh_und_token: str = MESH_UND_TOKEN, mesh_und_token_len: int = 513) -> str:
    return " ".join([mesh_und_token] * int(mesh_und_token_len))


def _seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if rgb.size == 0:
        return rgb
    finite_max = float(np.nanmax(rgb)) if np.isfinite(rgb).any() else 0.0
    if finite_max > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def normalize_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    xyz = point_cloud[:, :3].astype(np.float32)
    extras = point_cloud[:, 3:].astype(np.float32)
    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    radius = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    radius = max(float(radius), 1e-6)
    xyz = xyz / radius
    if extras.size == 0:
        return xyz.astype(np.float32)
    return np.concatenate([xyz, extras], axis=1).astype(np.float32)


def deterministic_downsample(point_cloud: np.ndarray, *, pointnum: int, object_id: str) -> np.ndarray:
    if point_cloud.shape[0] == pointnum:
        return point_cloud.astype(np.float32)
    if point_cloud.shape[0] < pointnum:
        repeat = int(np.ceil(pointnum / point_cloud.shape[0]))
        tiled = np.tile(point_cloud, (repeat, 1))
        return tiled[:pointnum].astype(np.float32)
    rng = np.random.default_rng(_seed_from_key(object_id))
    indices = rng.choice(point_cloud.shape[0], size=int(pointnum), replace=False)
    return point_cloud[np.sort(indices)].astype(np.float32)


def _ensure_point_feature_shape(array: np.ndarray, *, use_color: bool) -> np.ndarray:
    point_cloud = np.asarray(array, dtype=np.float32)
    if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
        raise ValueError(f"Expected point cloud array with shape [N, >=3], got {point_cloud.shape!r}")
    xyz = point_cloud[:, :3].astype(np.float32)
    if not use_color:
        return xyz

    extras = point_cloud[:, 3:]
    if extras.shape[1] >= 3:
        rgb = extras[:, :3]
    elif extras.shape[1] > 0:
        pad = np.full((point_cloud.shape[0], 3 - extras.shape[1]), NEUTRAL_RGB_VALUE, dtype=np.float32)
        rgb = np.concatenate([extras, pad], axis=1)
    else:
        rgb = np.full((point_cloud.shape[0], 3), NEUTRAL_RGB_VALUE, dtype=np.float32)
    rgb = _normalize_rgb(rgb)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def _load_ply_points_from_bytes(raw_bytes: bytes, *, use_color: bool) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(io.BytesIO(raw_bytes))
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if use_color and all(name in vertex.dtype.names for name in ("red", "green", "blue")):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
        return np.concatenate([xyz, rgb], axis=1)
    return xyz


def _load_point_cloud_from_npz(path: Path) -> np.ndarray:
    payload = np.load(path)
    candidate_keys = ["points", "point_cloud", "xyz", "coords", "vertices", "pc"]
    array = None
    for key in candidate_keys:
        if key in payload:
            array = payload[key]
            break
    if array is None:
        if len(payload.files) != 1:
            raise ValueError(f"Unable to infer point cloud key from {path}; keys={payload.files}")
        array = payload[payload.files[0]]
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Resolved payload from {path} is not an ndarray.")

    if array.ndim == 2 and array.shape[1] >= 3:
        if any(key in payload for key in ["colors", "rgb"]):
            color_key = "colors" if "colors" in payload else "rgb"
            colors = np.asarray(payload[color_key], dtype=np.float32)
            if colors.ndim != 2 or colors.shape[0] != array.shape[0]:
                raise ValueError(f"Color array shape mismatch in {path}: points={array.shape}, colors={colors.shape}")
            return np.concatenate([array[:, :3], colors[:, :3]], axis=1).astype(np.float32)
        return array.astype(np.float32)
    raise ValueError(f"Unsupported npz point cloud shape in {path}: {array.shape!r}")


def _scene_to_geometry(loaded: Any) -> Any:
    import trimesh

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geom
            for geom in loaded.dump(concatenate=False)
            if isinstance(geom, trimesh.Trimesh) and len(getattr(geom, "vertices", [])) > 0
        ]
        if len(geometries) == 1:
            return geometries[0]
        geometry = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
        if geometry is None:
            raise ValueError("trimesh.Scene did not contain concatenable geometry.")
        return geometry
    return loaded


def _get_texture_image(material: Any) -> Any:
    if material is None:
        return None
    for attr_name in ("baseColorTexture", "image"):
        texture = getattr(material, attr_name, None)
        if texture is not None:
            return texture
    data = getattr(material, "_data", None)
    if isinstance(data, dict):
        for key in ("baseColorTexture", "image"):
            texture = data.get(key)
            if texture is not None:
                return texture
    return None


def _sample_texture_colors(mesh: Any, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    texture = _get_texture_image(material)
    if uv is None or texture is None or len(face_indices) == 0:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv_array = np.asarray(uv, dtype=np.float64)
    if vertices.ndim != 2 or faces.ndim != 2 or uv_array.ndim != 2:
        return None
    if vertices.shape[1] < 3 or faces.shape[1] < 3 or uv_array.shape[1] < 2:
        return None
    if len(uv_array) < len(vertices):
        return None

    safe_face_indices = np.clip(face_indices.astype(np.int64), 0, len(faces) - 1)
    face_vertices = faces[safe_face_indices, :3]
    triangles = vertices[np.clip(face_vertices, 0, len(vertices) - 1)]
    sample_points = np.asarray(points, dtype=np.float64)

    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    rel = sample_points - triangles[:, 0]
    d00 = np.einsum("ij,ij->i", edge0, edge0)
    d01 = np.einsum("ij,ij->i", edge0, edge1)
    d11 = np.einsum("ij,ij->i", edge1, edge1)
    d20 = np.einsum("ij,ij->i", rel, edge0)
    d21 = np.einsum("ij,ij->i", rel, edge1)
    denom = d00 * d11 - d01 * d01
    valid = np.abs(denom) > 1e-12
    bary = np.full((len(sample_points), 3), 1.0 / 3.0, dtype=np.float64)
    bary[valid, 1] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    bary[valid, 2] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    bary[valid, 0] = 1.0 - bary[valid, 1] - bary[valid, 2]
    bary = np.clip(bary, 0.0, 1.0)
    bary = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-8)

    face_uv = uv_array[np.clip(face_vertices, 0, len(uv_array) - 1), :2]
    sample_uv = np.einsum("ij,ijk->ik", bary, face_uv)
    sample_uv = sample_uv - np.floor(sample_uv)

    image = np.asarray(texture.convert("RGBA") if hasattr(texture, "convert") else texture)
    if image.ndim < 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    xs = np.clip(np.rint(sample_uv[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
    ys = np.clip(np.rint((1.0 - sample_uv[:, 1]) * (height - 1)).astype(np.int64), 0, height - 1)
    return _normalize_rgb(image[ys, xs, :3])


def _extract_mesh_sample_colors(mesh: Any, points: np.ndarray, face_indices: np.ndarray, num_points: int) -> np.ndarray:
    texture_colors = _sample_texture_colors(mesh, points, face_indices)
    if texture_colors is not None and texture_colors.shape == (num_points, 3):
        return texture_colors

    face_colors = getattr(mesh.visual, "face_colors", None)
    if face_colors is not None and len(face_colors) >= len(mesh.faces):
        rgb = np.asarray(face_colors, dtype=np.float32)[face_indices, :3]
        return _normalize_rgb(rgb)

    vertex_colors = getattr(mesh.visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) >= len(mesh.vertices):
        face_vertices = np.asarray(mesh.faces, dtype=np.int64)[face_indices]
        rgb = np.asarray(vertex_colors, dtype=np.float32)[face_vertices, :3].mean(axis=1)
        return _normalize_rgb(rgb)

    return np.full((num_points, 3), NEUTRAL_RGB_VALUE, dtype=np.float32)


def _load_with_trimesh(path: Path, *, pointnum: int, use_color: bool, sample_seed_key: str | None = None) -> np.ndarray:
    import trimesh

    loaded = _scene_to_geometry(trimesh.load(path, force="scene"))
    if isinstance(loaded, trimesh.points.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float32)
        colors = np.asarray(getattr(loaded, "colors", None), dtype=np.float32) if getattr(loaded, "colors", None) is not None else None
        if use_color and colors is not None and colors.ndim == 2 and colors.shape[0] == points.shape[0]:
            return np.concatenate([points, colors[:, :3]], axis=1).astype(np.float32)
        return points.astype(np.float32)

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported geometry type from {path}: {type(loaded)!r}")
    if len(loaded.vertices) == 0:
        raise ValueError(f"Mesh {path} does not contain vertices.")

    sample_count = max(int(pointnum), len(loaded.vertices))
    state = np.random.get_state()
    np.random.seed(_seed_from_key(sample_seed_key or str(path)))
    try:
        points, face_indices = trimesh.sample.sample_surface(loaded, sample_count)
    finally:
        np.random.set_state(state)

    if use_color:
        colors = _extract_mesh_sample_colors(loaded, points, np.asarray(face_indices), sample_count)
        return np.concatenate([points.astype(np.float32), colors.astype(np.float32)], axis=1)
    return np.asarray(points, dtype=np.float32)


def load_mesh_und_values(
    path: str | Path,
    *,
    pointnum: int,
    use_color: bool,
    deterministic_key: str | None = None,
) -> np.ndarray:
    point_path = Path(path).expanduser()
    if not point_path.exists():
        raise FileNotFoundError(f"Input geometry file not found: {point_path}")

    suffix = point_path.suffix.lower()
    if suffix == ".npy":
        raw_point_cloud = np.load(point_path)
    elif suffix == ".glb":
        raw_point_cloud = _load_with_trimesh(point_path, pointnum=pointnum, use_color=use_color)
    else:
        raise ValueError(f"Unsupported geometry suffix {suffix!r}. Supported suffixes: .glb, .npy.")

    point_cloud = _ensure_point_feature_shape(raw_point_cloud, use_color=use_color)
    sampling_key = deterministic_key or str(point_path)
    point_cloud = deterministic_downsample(point_cloud, pointnum=int(pointnum), object_id=sampling_key)
    return normalize_point_cloud(point_cloud).astype(np.float32)


__all__ = [
    "MESH_UND_TOKEN",
    "SUPPORTED_GEOMETRY_SUFFIXES",
    "build_mesh_und_token_text",
    "deterministic_downsample",
    "load_mesh_und_values",
    "normalize_point_cloud",
]
