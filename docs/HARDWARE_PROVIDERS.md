# Hardware provider architecture

SceneSmith separates *what a subsystem needs* from *which vendor runtime supplies
it*. The five-agent scene workflow is unchanged: providers affect execution and
asset generation, not planning, iteration, scene state, or the semantic geometry
contract.

## Capability matrix

| Subsystem | Portable providers | Automatic behavior |
| --- | --- | --- |
| Torch/CLIP retrieval | `cuda`, `mps`, `cpu` | `balanced` prefers inexpensive local acceleration; `performance` prefers CUDA; `cost` prefers MPS/CPU |
| Blender Cycles | `optix`, `cuda`, `hip`, `oneapi`, `metal`, `cpu` | Probes supported Cycles devices and falls back to CPU only in `auto` mode |
| Blender process isolation | `shared`, `nvidia-bwrap` | Uses Linux/NVIDIA isolation only when a numeric CUDA slot and bubblewrap are available |
| SAM3D local generation | `cuda`, `mlx` | `mlx` on Apple Silicon; `cuda` on CUDA hosts |
| Hunyuan3D local generation | `cuda` | Fails clearly on non-CUDA hosts; it never fabricates GPU 0 |
| Geometry service | `local`, `external` | Local by default; external connects to a shared accelerator service without owning its lifecycle |

CPU-only and non-NVIDIA systems can run the agent workflow, CLIP retrieval, and
Blender. Open-set generated geometry needs SAM3D/MLX, a CUDA provider, or an
external geometry service. HSSD and Objaverse retrieval remain alternatives when
their datasets are installed.

## Configuration

The base experiment exposes checked-in defaults:

```yaml
geometry_generation_server:
  host: "127.0.0.1"
  port: 7005
  provider: "local"  # local or external

execution_providers:
  compute: "auto"     # auto, cuda, mps, cpu
  policy: "balanced"  # balanced, performance, cost
  render: "auto"      # auto, optix, cuda, hip, oneapi, metal, cpu
  geometry: "auto"    # auto, cuda, mlx
```

Environment variables override configuration so the same experiment can move
between a workstation and a cloud host:

| Variable | Values |
| --- | --- |
| `SCENESMITH_COMPUTE_PROVIDER` | `auto`, `cuda`, `cuda:N`, `mps`, `cpu` |
| `SCENESMITH_RENDER_PROVIDER` | `auto`, `optix`, `cuda`, `hip`, `oneapi`, `metal`, `cpu` |
| `SCENESMITH_RENDER_PROCESS_PROVIDER` | `auto`, `shared`, `nvidia-bwrap` |
| `SCENESMITH_GEOMETRY_PROVIDER` | `auto`, `cuda`, `mlx` |
| `SCENESMITH_GEOMETRY_SERVICE_PROVIDER` | `local`, `external` |
| `SCENESMITH_SAM_PROVIDER` | `auto`, `cuda`, `mlx` |

Explicit unavailable providers are errors. Only `auto` may fall through to a
different implementation. This prevents a requested accelerator from silently
running a much slower or lower-fidelity path.

### Examples

Use Apple Silicon locally:

```sh
SCENESMITH_COMPUTE_PROVIDER=mps \
SCENESMITH_RENDER_PROVIDER=metal \
SCENESMITH_GEOMETRY_PROVIDER=mlx \
uv run python main.py +name=my_experiment
```

Use a CPU workstation with a shared geometry host:

```sh
SCENESMITH_COMPUTE_PROVIDER=cpu \
SCENESMITH_RENDER_PROVIDER=cpu \
SCENESMITH_GEOMETRY_SERVICE_PROVIDER=external \
uv run python main.py +name=my_experiment \
  experiment.geometry_generation_server.host=geometry.example.net \
  experiment.geometry_generation_server.port=7005
```

Use the fastest available NVIDIA path:

```sh
SCENESMITH_COMPUTE_PROVIDER=cuda \
SCENESMITH_RENDER_PROVIDER=optix \
SCENESMITH_GEOMETRY_PROVIDER=cuda \
uv run python main.py +name=my_experiment
```

## Platform installation

`uv sync --locked` is the canonical host install. The lock contains both:

- Drake 1.40 for the Python 3.11 Apple Silicon host, because it is the last
  compatible macOS wheel for this runtime; and
- current Drake 1.49+ for Linux x86-64.

Blender 4.5.4, Shapely, Torch, and the rest of the base runtime resolve natively
on Apple Silicon. SAM3D/MLX uses its isolated environment:

```sh
bash scripts/install_sam3d_mlx.sh --download-checkpoints
```

CUDA model dependencies remain isolated behind their installers:

```sh
bash scripts/install_sam3d.sh
# or
bash scripts/install_hunyuan3d.sh
```

PyTorch documents MPS device availability and optional operation fallback in its
[MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html). Blender
documents its supported CUDA, OptiX, HIP, oneAPI, and Metal Cycles providers in
the [Cycles GPU rendering manual](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html).

## Injection boundaries

Provider selection is dependency-light and does not import accelerator SDKs in a
parent process that will fork workers. Downstream code can inject:

- an `ExecutionProvider` into `ExecutionProviderRegistry`;
- a `GeometryExecutionProvider` into `GeometryWorkerPool`;
- a `GeometryServiceProvider` at the orchestration boundary; and
- a `RenderProcessProvider` into `BlenderServer`.

Model-specific CUDA initialization is still necessary inside the CUDA SAM3D and
Hunyuan implementations. It is an implementation detail, not an orchestration
decision. `test_hardware_provider_boundaries.py` scans production code and fails
if direct runtime probes appear outside the approved provider modules.

### CUDA audit result

Every executable CUDA reference falls into one of these owned boundaries:

| Owner | Remaining vendor-specific responsibility |
| --- | --- |
| `agent_utils/execution_providers.py` | CUDA discovery, Torch target construction, cache release, and MPS/CPU alternatives |
| `geometry_generation_server/execution_provider.py` | Physical CUDA worker targets and their process environment |
| `geometry_generation_server/cuda_env_setup.py` | CUDA toolkit discovery needed by the CUDA SAM3D implementation |
| `geometry_generation_server/sam3d_pipeline_manager.py` | nvdiffrast/Warp CUDA context ordering inside the CUDA provider only |
| `blender/render_provider.py` | Cycles vendor API probing, including non-CUDA alternatives |
| `blender/process_provider.py` | Optional Linux/NVIDIA device-file isolation |
| `geometry_generation_server/worker_pool.py` | Deprecated detector wrapper that delegates to the shared registry; retained for API compatibility |

The SAM3D and Hunyuan install scripts are provider-specific deployment tools,
not runtime selection logic. Docker's NVIDIA configuration is likewise an
explicit CUDA deployment profile. References in tests and documentation do not
make hardware decisions.

## Verification

```sh
uv lock --check
uv run pytest -q \
  tests/unit/runtime/test_execution_providers.py \
  tests/unit/blender/test_blender_render_provider.py \
  tests/unit/blender/test_blender_process_provider.py \
  tests/unit/geometry/test_geometry_execution_provider.py \
  tests/unit/geometry/test_geometry_service_provider.py \
  tests/unit/architecture/test_hardware_provider_boundaries.py
```

GitHub Actions runs the unit suite on Linux and an Apple Silicon `macos-15`
runner. A labelled self-hosted CUDA gate can be enabled with the repository
variable `SCENESMITH_CUDA_RUNNER=enabled`; it is opt-in so ordinary pull requests
do not pay for a GPU runner.
