"""Minimal O-Voxel bridge used by the public mesh exporter."""

from __future__ import annotations

import importlib
import sys

try:
    from . import _C  # type: ignore[attr-defined]
except ImportError:
    _C = importlib.import_module("o_voxel._C")

from . import convert

sys.modules.setdefault("o_voxel", sys.modules[__name__])

__all__ = ["_C", "convert"]
