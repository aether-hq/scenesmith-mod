"""Geometry generation server components.

This module contains the complete geometry generation server implementation,
including server infrastructure and both Hunyuan3D and SAM3D backends.

Local execution is selected through an injectable geometry provider. CUDA uses
one isolated worker per visible NVIDIA device; MLX uses one Metal worker.

IMPORTANT: The server-related imports (GeometryGenerationServer, GeometryGenerationClient)
do not initialize an accelerator runtime. Model implementations are accessed via
lazy loading so provider initialization remains isolated to workers.
"""

# Safe imports that do not initialize an accelerator runtime.
from .client import GeometryGenerationClient
from .dataclasses import (
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
)
from .server.server_manager import GeometryGenerationServer

# Lazy imports for model-specific modules.
# These should only be imported in provider workers or when explicitly needed.


def __getattr__(name: str):
    """Lazy load model implementations.

    This prevents accelerator initialization in the parent process.
    """
    if name == "generate_geometry_from_image":
        from .geometry_generation import generate_geometry_from_image

        return generate_geometry_from_image

    if name == "Hunyuan3DPipelineManager":
        from .pipelines.hunyuan3d_pipeline_manager import Hunyuan3DPipelineManager

        return Hunyuan3DPipelineManager

    if name == "SAM3DPipelineManager":
        from .pipelines.sam3d_pipeline_manager import SAM3DPipelineManager

        return SAM3DPipelineManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Safe imports (no accelerator initialization).
    "GeometryGenerationClient",
    "GeometryGenerationServer",
    "GeometryGenerationServerRequest",
    "GeometryGenerationServerResponse",
    # Lazy model implementation imports.
    "generate_geometry_from_image",
    "Hunyuan3DPipelineManager",
    "SAM3DPipelineManager",
]
