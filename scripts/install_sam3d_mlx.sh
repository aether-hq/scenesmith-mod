#!/usr/bin/env bash

# Install the Apple Silicon SAM 3D Objects provider used by SceneSmith's
# sam3d.provider=mlx facade. Model access is gated by Meta on Hugging Face.

set -euo pipefail

REPOSITORY_URL="${SAM3D_MLX_REPOSITORY_URL:-https://github.com/ZimengXiong/Sam3D-Objects-MLX.git}"
REPOSITORY_COMMIT="${SAM3D_MLX_COMMIT:-c6f3701e4c9d45281afe0f022d2ba499cd60b39d}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROVIDER_ROOT="${PROJECT_ROOT}/external/Sam3D-Objects-MLX"
DOWNLOAD_CHECKPOINTS=false

if [[ "${1:-}" == "--download-checkpoints" ]]; then
  DOWNLOAD_CHECKPOINTS=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: bash scripts/install_sam3d_mlx.sh [--download-checkpoints]" >&2
  exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "The MLX provider requires macOS on Apple Silicon." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/external"
if [[ ! -d "${PROVIDER_ROOT}/.git" ]]; then
  git clone "${REPOSITORY_URL}" "${PROVIDER_ROOT}"
fi
git -C "${PROVIDER_ROOT}" fetch origin
git -C "${PROVIDER_ROOT}" checkout --detach "${REPOSITORY_COMMIT}"

# Open3D publishes macOS arm64 wheels for CPython 3.12, but not 3.13.
env MAX_JOBS="${MAX_JOBS:-8}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}" \
  uv sync --project "${PROVIDER_ROOT}" --python 3.12

if [[ "${DOWNLOAD_CHECKPOINTS}" == true ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required to download checkpoints." >&2
    exit 1
  fi
  echo "Downloading gated facebook/sam-3d-objects checkpoints..."
  DOWNLOAD_ROOT="${PROVIDER_ROOT}/.hf-checkpoint-download"
  hf download facebook/sam-3d-objects \
    --repo-type model \
    --include "checkpoints/*" \
    --local-dir "${DOWNLOAD_ROOT}"
  mkdir -p "${PROVIDER_ROOT}/checkpoints/hf"
  cp -R "${DOWNLOAD_ROOT}/checkpoints/." "${PROVIDER_ROOT}/checkpoints/hf/"
fi

"${PROVIDER_ROOT}/.venv/bin/python" -c \
  'import torch; assert torch.backends.mps.is_available(), "PyTorch MPS is unavailable"'

echo "SAM3D MLX provider installed at ${PROVIDER_ROOT}"
if [[ ! -f "${PROVIDER_ROOT}/checkpoints/hf/pipeline.yaml" ]]; then
  echo "Next: accept the model license at https://huggingface.co/facebook/sam-3d-objects"
  echo "Then run: bash scripts/install_sam3d_mlx.sh --download-checkpoints"
fi
