"""Versioned semantic primitives for general structural scene geometry.

This module deliberately depends only on the Python standard library.  The
semantic layout can therefore be validated before starting Blender, Drake, or
any generative model service.  Mesh compilation lives in separate modules.
"""

from __future__ import annotations

SCHEMA_VERSION = 2
GEOMETRY_TOLERANCE = 1e-9

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Loop2 = tuple[Point2, ...]
