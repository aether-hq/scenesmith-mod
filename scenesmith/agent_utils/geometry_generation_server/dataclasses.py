"""Dataclasses for geometry generation server API contracts.

This module contains serializable Data Transfer Objects (DTOs) used for
communication with the geometry generation server. These classes define the
HTTP API contract and use primitive types for JSON serialization.
"""

import json
import math

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_MAX_PATH_CHARS = 4096
_MAX_PROMPT_CHARS = 8192
_MAX_SCENE_ID_CHARS = 256
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_CONFIG_DEPTH = 12
_MAX_CONFIG_ITEMS = 512


@dataclass(frozen=True)
class GeometryGenerationServerRequest:
    """Request payload for geometry generation server.

    This DTO defines the contract for geometry generation requests sent
    to the geometry generation server via HTTP. Contains all information
    needed to generate 3D geometry from a 2D image.
    """

    image_path: str
    """Absolute path to the input image file (PNG/JPG format)."""

    output_dir: str
    """Absolute path to directory where generated assets will be saved."""

    prompt: str
    """Text description of the asset to generate (e.g., 'Modern wooden chair')."""

    debug_folder: str | None = None
    """Optional absolute path to directory where debug images will be saved."""

    output_filename: str | None = None
    """Optional filename for the generated geometry file. If not provided, will be
    generated from prompt."""

    backend: str = "hunyuan3d"
    """3D generation backend to use. Either "hunyuan3d" or "sam3d"."""

    sam3d_config: dict | None = None
    """Configuration for SAM3D backend. Required if backend="sam3d". Should contain:
    - provider (str): "auto", "mlx", or "cuda"
    - sam3_checkpoint / sam3d_checkpoint: CUDA provider checkpoints
    - mlx_repo_path / mlx_python_path / mlx_checkpoint_dir: MLX runtime paths
    - mode (str): Segmentation mode ("foreground" or "object_description")
    - object_description (str | None): Object description (if mode="object_description")
    - threshold (float): Confidence threshold for mask generation
    """

    scene_id: str | None = None
    """Optional scene identifier for fair round-robin scheduling.

    When multiple scenes submit requests concurrently, the server uses this ID to
    group requests from the same scene together for fair GPU time allocation.
    All requests with the same scene_id are treated as a single "client" in the
    round-robin scheduler. If not provided, each HTTP request is treated as a
    separate client.
    """

    def __post_init__(self) -> None:
        _validate_string("image_path", self.image_path, max_chars=_MAX_PATH_CHARS)
        _validate_string("output_dir", self.output_dir, max_chars=_MAX_PATH_CHARS)
        _validate_string("prompt", self.prompt, max_chars=_MAX_PROMPT_CHARS)
        for name, value in (
            ("debug_folder", self.debug_folder),
            ("output_filename", self.output_filename),
            ("scene_id", self.scene_id),
        ):
            if value is not None:
                limit = _MAX_SCENE_ID_CHARS if name == "scene_id" else _MAX_PATH_CHARS
                _validate_string(name, value, max_chars=limit)
        if (
            self.output_filename is not None
            and Path(self.output_filename).name != self.output_filename
        ):
            raise ValueError("output_filename must be a safe basename")
        if type(self.backend) is not str or self.backend not in {"hunyuan3d", "sam3d"}:
            raise ValueError("backend must be 'hunyuan3d' or 'sam3d'")
        if self.sam3d_config is not None:
            if type(self.sam3d_config) is not dict:
                raise TypeError("sam3d_config must be a JSON object")
            item_count = [0]
            _validate_json_value(
                self.sam3d_config,
                path="sam3d_config",
                depth=0,
                item_count=item_count,
            )
            try:
                encoded = json.dumps(
                    self.sam3d_config,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "sam3d_config must contain finite JSON values"
                ) from exc
            if len(encoded) > _MAX_CONFIG_BYTES:
                raise ValueError("sam3d_config exceeds its byte budget")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for HTTP requests.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for HTTP request body.

        Returns:
            JSON string representation of the request.
        """
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class GeometryGenerationServerResponse:
    """Response payload from geometry generation server.

    This DTO defines the contract for responses from the geometry
    generation server after successful 3D geometry generation.
    """

    geometry_path: str
    """Absolute path to the generated 3D geometry file (GLB format)."""

    def __post_init__(self) -> None:
        _validate_string("geometry_path", self.geometry_path, max_chars=_MAX_PATH_CHARS)


@dataclass(frozen=True)
class GeometryGenerationError:
    """Error information for a failed geometry generation request.

    This DTO represents a geometry generation failure that occurred on the
    server. Used to communicate errors back to the client without stopping
    the entire batch.
    """

    index: int
    """Index of the failed request within the original batch."""

    error_message: str
    """Description of what went wrong during geometry generation."""

    def __post_init__(self) -> None:
        _validate_result_index(self.index)
        _validate_string(
            "error_message", self.error_message, max_chars=_MAX_PROMPT_CHARS
        )


@dataclass(frozen=True)
class StreamedResult:
    """Single result in a streaming batch response.

    This DTO represents one completed request result in a streaming
    NDJSON response from the server.
    """

    index: int
    """Index of this result within the original batch request."""

    status: str
    """Result status: either "success" or "error"."""

    data: dict | None = None
    """Response data for successful requests (contains geometry_path, etc.)."""

    error: str | None = None
    """Error message for failed requests."""

    def __post_init__(self) -> None:
        _validate_result_index(self.index)
        if type(self.status) is not str or self.status not in {"success", "error"}:
            raise ValueError("status must be 'success' or 'error'")
        if self.status == "success":
            if type(self.data) is not dict:
                raise ValueError("success result must contain object data")
            if self.error is not None:
                raise ValueError("success result must not contain an error message")
            _validate_json_value(
                self.data,
                path="data",
                depth=0,
                item_count=[0],
            )
        else:
            if self.data is not None:
                raise ValueError("error result must not contain success data")
            if self.error is None:
                raise ValueError("error result must contain an error message")
            _validate_string("error", self.error, max_chars=_MAX_PROMPT_CHARS)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for streaming response."""
        return json.dumps(self.to_dict())


def _validate_string(name: str, value: Any, *, max_chars: int) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or len(value) > max_chars:
        raise ValueError(f"{name} must contain 1 to {max_chars} characters")


def _validate_result_index(index: Any) -> None:
    if type(index) is not int:
        raise TypeError("index must be an integer")
    if index < 0:
        raise ValueError("index must be non-negative")


def _validate_json_value(
    value: Any, *, path: str, depth: int, item_count: list[int]
) -> None:
    if depth > _MAX_CONFIG_DEPTH:
        raise ValueError(f"{path} exceeds the nesting budget")
    item_count[0] += 1
    if item_count[0] > _MAX_CONFIG_ITEMS:
        raise ValueError("sam3d_config exceeds its item budget")
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numeric values")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                item_count=item_count,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            _validate_json_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                item_count=item_count,
            )
        return
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")
