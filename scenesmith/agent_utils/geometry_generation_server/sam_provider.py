"""Provider facade for SAM 3D geometry generation.

The CUDA implementation runs in the SceneSmith process.  The Apple Silicon
implementation is intentionally invoked as a subprocess so its Metal-specific
dependencies can live in an isolated virtual environment.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess

from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from scenesmith.agent_utils.execution_providers import (
    CudaVisibilityToken,
    ProviderUnavailableError,
    detect_cuda_device_ids,
)

SamProviderName = Literal["cuda", "mlx"]

console_logger = logging.getLogger(__name__)


def _get(config: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    """Read a value from a dict or OmegaConf DictConfig."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def resolve_sam_provider(
    config: Mapping[str, Any] | Any,
    *,
    system: str | None = None,
    machine: str | None = None,
    cuda_detector: Callable[
        [], tuple[CudaVisibilityToken, ...]
    ] = detect_cuda_device_ids,
    requested: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> SamProviderName:
    """Resolve ``auto`` to the platform-appropriate SAM provider.

    ``SCENESMITH_SAM_PROVIDER`` takes precedence over configuration, making the
    same checked-in config usable on a Mac workstation and a CUDA cloud host.
    """
    current_environ = os.environ if environ is None else environ
    effective = (
        requested
        or current_environ.get("SCENESMITH_SAM_PROVIDER")
        or _get(config, "provider", "auto")
    )
    requested = str(effective).strip().lower()

    if requested in {"metal", "mps", "apple"}:
        requested = "mlx"
    if requested not in {"auto", "cuda", "mlx"}:
        raise ValueError(
            f"Unknown SAM provider '{requested}'. Expected auto, mlx, or cuda."
        )
    if requested != "auto":
        return requested  # type: ignore[return-value]

    current_system = (system or platform.system()).lower()
    current_machine = (machine or platform.machine()).lower()
    if current_system == "darwin" and current_machine in {"arm64", "aarch64"}:
        return "mlx"
    if cuda_detector():
        return "cuda"
    raise ProviderUnavailableError(
        "SAM3D auto-selection found no supported local provider. Install the MLX "
        "provider on Apple Silicon, expose a CUDA device, or use a remote geometry "
        "service."
    )


def sam_provider_config_from_mapping(
    config: Mapping[str, Any] | Any,
    *,
    object_description: str | None = None,
) -> dict[str, Any]:
    """Create a primitive, process-safe provider config from an agent config."""
    watermark = _mlx_memory_watermark(config)
    result: dict[str, Any] = {
        "provider": str(_get(config, "provider", "auto")),
        "sam3_checkpoint": str(
            _get(config, "sam3_checkpoint", "external/checkpoints/sam3.pt")
        ),
        "sam3d_checkpoint": str(
            _get(
                config,
                "sam3d_checkpoint",
                "external/checkpoints/pipeline.yaml",
            )
        ),
        "mode": str(_get(config, "mode", "foreground")),
        "threshold": float(_get(config, "threshold", 0.5)),
        "mlx_repo_path": str(
            _get(config, "mlx_repo_path", "external/Sam3D-Objects-MLX")
        ),
        "mlx_python_path": str(
            _get(
                config,
                "mlx_python_path",
                "external/Sam3D-Objects-MLX/.venv/bin/python",
            )
        ),
        "mlx_checkpoint_dir": str(
            _get(
                config,
                "mlx_checkpoint_dir",
                "external/Sam3D-Objects-MLX/checkpoints/hf",
            )
        ),
        "mlx_steps": int(_get(config, "mlx_steps", 12)),
        "mlx_seed": int(_get(config, "mlx_seed", 42)),
        "mlx_simplify_ratio": float(_get(config, "mlx_simplify_ratio", 0.0)),
        "mlx_mask_color_threshold": float(
            _get(config, "mlx_mask_color_threshold", 24.0)
        ),
        "mlx_timeout_seconds": int(_get(config, "mlx_timeout_seconds", 7200)),
        "mlx_mps_high_watermark_ratio": watermark,
    }
    if object_description:
        result["object_description"] = object_description
    return result


def validate_sam_provider_config(
    config: Mapping[str, Any] | Any,
) -> SamProviderName:
    """Validate provider-specific runtime files and return the resolved provider."""
    provider = resolve_sam_provider(config)
    if provider == "cuda":
        sam3_checkpoint = Path(_get(config, "sam3_checkpoint"))
        sam3d_checkpoint = Path(_get(config, "sam3d_checkpoint"))
        if not sam3_checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {sam3_checkpoint}. "
                "Run 'bash scripts/install_sam3d.sh' after Hugging Face access is approved."
            )
        if not sam3d_checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM 3D Objects checkpoint not found: {sam3d_checkpoint}. "
                "Run 'bash scripts/install_sam3d.sh' after Hugging Face access is approved."
            )
        return provider

    repo_path = Path(_get(config, "mlx_repo_path")).expanduser().resolve()
    # Keep the venv launcher path intact. On uv-managed environments it is a
    # symlink; resolving it selects the base interpreter and silently drops the
    # venv's site-packages (including torch).
    python_path = Path(_get(config, "mlx_python_path")).expanduser().absolute()
    checkpoint_dir = Path(_get(config, "mlx_checkpoint_dir")).expanduser().resolve()
    if not (repo_path / "main.py").is_file():
        raise FileNotFoundError(
            f"SAM3D MLX checkout not found: {repo_path}. "
            "Run 'bash scripts/install_sam3d_mlx.sh'."
        )
    if not python_path.is_file():
        raise FileNotFoundError(
            f"SAM3D MLX Python environment not found: {python_path}. "
            "Run 'bash scripts/install_sam3d_mlx.sh'."
        )
    if not (checkpoint_dir / "pipeline.yaml").is_file():
        raise FileNotFoundError(
            f"SAM3D MLX checkpoints not found in {checkpoint_dir}. "
            "Accept the facebook/sam-3d-objects license on Hugging Face, then run "
            "'bash scripts/install_sam3d_mlx.sh --download-checkpoints'."
        )
    weights = list(checkpoint_dir.rglob("*.pt")) + list(
        checkpoint_dir.rglob("*.safetensors")
    )
    if not weights:
        raise FileNotFoundError(
            f"No SAM3D MLX weight files found in {checkpoint_dir}. "
            "Run 'bash scripts/install_sam3d_mlx.sh --download-checkpoints'."
        )
    return provider


def preload_sam_provider(config: Mapping[str, Any] | Any) -> None:
    """Preload CUDA models or cheaply validate the isolated MLX runtime."""
    provider = validate_sam_provider_config(config)
    if provider == "mlx":
        console_logger.info("SAM3D MLX provider ready (models load per subprocess)")
        return

    # Importing the CUDA manager configures CUDA and must stay behind dispatch.
    from scenesmith.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
        SAM3DPipelineManager,
    )

    SAM3DPipelineManager.get_pipelines(
        sam3_checkpoint=Path(_get(config, "sam3_checkpoint")),
        sam3d_checkpoint=Path(_get(config, "sam3d_checkpoint")),
    )


def generate_with_sam_provider(
    *,
    image_path: Path,
    output_path: Path,
    config: Mapping[str, Any] | Any,
    debug_folder: Path | None = None,
    use_pipeline_caching: bool = True,
) -> None:
    """Generate a GLB using the resolved SAM provider."""
    provider = resolve_sam_provider(config)
    console_logger.info("Generating %s with SAM provider '%s'", output_path, provider)
    if provider == "mlx":
        _generate_with_mlx(
            image_path=image_path,
            output_path=output_path,
            config=config,
            debug_folder=debug_folder,
        )
        return

    from scenesmith.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
        generate_with_sam3d,
    )

    generate_with_sam3d(
        image_path=image_path,
        output_path=output_path,
        sam3_checkpoint=Path(_get(config, "sam3_checkpoint")),
        sam3d_checkpoint=Path(_get(config, "sam3d_checkpoint")),
        mode=_get(config, "mode", "foreground"),
        object_description=_get(config, "object_description"),
        threshold=float(_get(config, "threshold", 0.5)),
        debug_folder=debug_folder,
        use_pipeline_caching=use_pipeline_caching,
    )


def _create_foreground_mask(
    image_path: Path, mask_path: Path, color_threshold: float
) -> None:
    """Extract the main foreground object from SceneSmith's uniform background."""
    import numpy as np

    from PIL import Image
    from scipy import ndimage

    rgba = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    if alpha.min() < 250:
        mask = alpha > 16
    else:
        rgb = rgba[..., :3].astype(np.float32)
        border = np.concatenate(
            [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(rgb - background, axis=2)
        mask = distance >= color_threshold

    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)

    coverage = float(mask.mean())
    if coverage < 0.005 or coverage > 0.98:
        raise RuntimeError(
            f"Could not isolate a foreground object from {image_path} "
            f"(mask coverage {coverage:.1%}). Use a uniform contrasting background."
        )

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)


def _generate_with_mlx(
    *,
    image_path: Path,
    output_path: Path,
    config: Mapping[str, Any] | Any,
    debug_folder: Path | None,
) -> None:
    validate_sam_provider_config(config)
    repo_path = Path(_get(config, "mlx_repo_path")).expanduser().resolve()
    # Do not resolve this symlink: uv venv launchers rely on their own path to
    # select the environment's site-packages.
    python_path = Path(_get(config, "mlx_python_path")).expanduser().absolute()
    checkpoint_dir = Path(_get(config, "mlx_checkpoint_dir")).expanduser().resolve()

    # The upstream MLX script currently resolves this path relative to its checkout.
    expected_checkpoint_dir = (repo_path / "checkpoints" / "hf").resolve()
    if checkpoint_dir != expected_checkpoint_dir:
        raise ValueError(
            "Sam3D-Objects-MLX currently requires mlx_checkpoint_dir to be "
            f"{expected_checkpoint_dir}; got {checkpoint_dir}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_root = debug_folder or output_path.parent / ".sam3d_mlx_masks"
    mask_path = mask_root / f"{image_path.stem}_foreground.png"
    _create_foreground_mask(
        image_path,
        mask_path,
        float(_get(config, "mlx_mask_color_threshold", 24.0)),
    )

    if _get(config, "mode", "foreground") == "object_description":
        console_logger.warning(
            "The MLX provider does not include text-prompted SAM3 segmentation; "
            "using SceneSmith's uniform-background foreground mask instead."
        )

    command = [
        str(python_path),
        str(repo_path / "main.py"),
        "--image",
        str(image_path.resolve()),
        "--mask",
        str(mask_path.resolve()),
        "--mesh",
        "--output",
        str(output_path.resolve()),
        "--steps",
        str(int(_get(config, "mlx_steps", 12))),
        "--seed",
        str(int(_get(config, "mlx_seed", 42))),
        "--simplify",
        str(float(_get(config, "mlx_simplify_ratio", 0.0))),
        "--cache-dir",
        str((output_path.parent / ".sam3d_mlx_cache").resolve()),
    ]
    env = os.environ.copy()
    # macOS injects this launcher hint into Python child processes. If it is
    # inherited from SceneSmith's venv, executing the MLX venv's python still
    # resolves packages against the parent interpreter and cannot import its
    # pinned torch build.
    env.pop("__PYVENV_LAUNCHER__", None)
    high_watermark = _mlx_memory_watermark(config)
    low_watermark = min(0.7, high_watermark * 0.875)
    env.update(
        {
            "VIRTUAL_ENV": str(python_path.parent.parent),
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": str(high_watermark),
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": str(low_watermark),
            "LIDRA_SKIP_INIT": "1",
            # Use the port's compatibility path. Its experimental native Metal
            # sparse kernels can segfault in mesh-decoder upsampling on current
            # Apple GPUs, while PyTorch MPS + SDPA completes reliably.
            "SPARSE_BACKEND": "mps",
            "SPARSE_ATTN_BACKEND": "sdpa",
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(repo_path), env.get("PYTHONPATH")])
            ),
        }
    )
    result = subprocess.run(
        command,
        cwd=repo_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(_get(config, "mlx_timeout_seconds", 7200)),
    )
    if result.stdout:
        console_logger.info("SAM3D MLX output:\n%s", result.stdout.rstrip())
    if result.returncode != 0:
        raise RuntimeError(
            f"SAM3D MLX exited with status {result.returncode}. See worker logs above."
        )
    if not output_path.is_file():
        raise RuntimeError(f"SAM3D MLX did not create expected GLB: {output_path}")


def _mlx_memory_watermark(config: Mapping[str, Any] | Any) -> float:
    watermark = float(_get(config, "mlx_mps_high_watermark_ratio", 0.8))
    if not 0.0 < watermark <= 1.0:
        raise ValueError(
            "mlx_mps_high_watermark_ratio must be greater than 0 and at most 1"
        )
    return watermark
