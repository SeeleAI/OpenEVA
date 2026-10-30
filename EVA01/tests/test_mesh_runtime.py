from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from eva01.mesh_runtime import export_mesh_glb, load_decoder


def test_decoder_loader_fails_closed_without_external_sidecars(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="external TRELLIS assets"):
        load_decoder(tmp_path / "missing_decoder", label="Shape", device=torch.device("cpu"))


def test_public_glb_writer_preserves_geometry_and_vertex_colors(tmp_path) -> None:
    mesh = SimpleNamespace(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        attrs=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
             [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]],
            dtype=torch.float32,
        ),
        layout={"base_color": slice(0, 3)},
    )
    output = tmp_path / "triangle.glb"
    export_mesh_glb(mesh, output)

    loaded = trimesh.load(output, force="scene")
    geometry = next(iter(loaded.geometry.values()))
    assert geometry.vertices.shape == (3, 3)
    assert geometry.faces.shape == (1, 3)
    assert np.asarray(geometry.visual.vertex_colors).shape[0] == 3
