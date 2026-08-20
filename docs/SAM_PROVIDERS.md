# SAM provider facade

SceneSmith exposes one `sam3d` geometry backend with two runtime providers:

| Provider | Platform | Implementation | Output contract |
| --- | --- | --- | --- |
| `mlx` | macOS on Apple Silicon | `ZimengXiong/Sam3D-Objects-MLX` in an isolated Python 3.12 environment | `.glb` mesh |
| `cuda` | Linux with NVIDIA CUDA | SceneSmith's original SAM3 + SAM 3D Objects pipeline | textured `.glb` |

Set `asset_manager.sam3d.provider` to `auto`, `mlx`, or `cuda`. `auto` chooses
`mlx` on Darwin/arm64 and `cuda` everywhere else. The environment variable
`SCENESMITH_SAM_PROVIDER` overrides the configuration, which lets one checked-in
configuration run on a Mac workstation and a CUDA cloud host.

## Apple Silicon setup

1. Request and accept access to
   [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects).
2. Authenticate the Hugging Face CLI with an account that has access.
3. Install the provider and weights:

   ```sh
   bash scripts/install_sam3d_mlx.sh --download-checkpoints
   ```

The installer pins the tested upstream revision and Python 3.12, installs its
Metal dependencies in `external/Sam3D-Objects-MLX/.venv`, and places weights in
`external/Sam3D-Objects-MLX/checkpoints/hf`.

The MLX port consumes an image plus a mask. SceneSmith produces asset images on a
uniform background, so the facade derives a foreground mask from the image border
and keeps the largest connected component. `object_description` mode currently
falls back to this foreground mask because the port does not include the separate
text-prompted SAM3 segmentation model.

The upstream MLX port does not support Gaussian splatting or color baking, so its
GLB is suitable for geometry composition but will not have the CUDA provider's
texture fidelity. It also describes its low-memory pipeline as targeting a 48 GB
Apple Silicon machine.

## CUDA setup

On a CUDA host, retain the existing setup:

```sh
bash scripts/install_sam3d.sh
```

The facade imports the CUDA implementation only after provider dispatch. The MLX
path therefore never initializes CUDA, while CUDA deployments preserve pipeline
caching and one worker per visible GPU.

## Host runtime

The complete Python 3.11 SceneSmith host now resolves natively on Apple Silicon.
The lock selects Drake 1.40 on macOS—the last compatible ARM wheel for this
Python runtime—and Drake 1.49+ on Linux x86-64. `bpy==4.5.4`, Shapely, Torch MPS,
and the semantic geometry compiler are installed in the main environment; the
SAM3D/MLX model remains in its isolated Python 3.12 environment.

See [Hardware provider architecture](HARDWARE_PROVIDERS.md) for compute, render,
geometry-worker, remote-service, dependency, and CI contracts.
