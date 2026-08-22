#!/usr/bin/env python3
"""Export an existing scene to self-contained MuJoCo MJCF format.

Takes a scene directory (e.g., outputs/2025-12-05/13-39-27/scene_039) and exports
it to a self-contained MuJoCo directory with the scene.xml and all referenced
mesh assets.

Can also export a single Drake SDF file to MuJoCo MJCF format.

Usage:
    python scripts/export_scene_to_mujoco.py <scene_path> [--output <output_path>]

Example:
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039 \
        --output /tmp/mujoco_scene
"""

import argparse
import logging
import sys

from pathlib import Path

import mujoco

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scripts.mujoco_export.mesh_conversion import load_house_from_directory
from scripts.mujoco_export.scene_export import (
    export_dmd_scene_to_mujoco,
    export_scene_to_mujoco,
)
from scripts.mujoco_export.sdf_export import export_sdf_to_mujoco
from scripts.mujoco_export.usd_export import export_to_usd, validate_mujoco_export

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def main():
    parser = argparse.ArgumentParser(
        description="Export an existing scene to self-contained MuJoCo MJCF format"
    )
    parser.add_argument(
        "scene_path",
        type=Path,
        nargs="?",
        help="Path to scene directory (e.g., outputs/2025-12-05/13-39-27/scene_039)",
    )
    parser.add_argument(
        "--sdf",
        type=Path,
        default=None,
        help="Convert a single SDF file directly (for testing articulated models)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for MuJoCo export (default: scene_path/mujoco)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Treat model as static (no freejoint at base link). Only used with --sdf.",
    )
    parser.add_argument(
        "--no-floor-plan",
        action="store_true",
        help="Exclude floor plan (floor, walls) from export",
    )
    parser.add_argument(
        "--weld-furniture",
        action="store_true",
        help="Make furniture static (no freejoint). Default: furniture has freejoint.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip MuJoCo validation after export",
    )
    parser.add_argument(
        "--usd",
        action="store_true",
        help="Also export to USD format (OpenUSD/Universal Scene Description)",
    )
    parser.add_argument(
        "--skip-isaac-sim-fix",
        action="store_true",
        help=(
            "When exporting USD, skip the Isaac Sim compatibility fixer and "
            "leave the raw mujoco-usd-converter output untouched"
        ),
    )

    args = parser.parse_args()

    if args.scene_path is None and args.sdf is None:
        parser.error("Either scene_path or --sdf must be specified")

    # Handle standalone SDF conversion mode.
    if args.sdf:
        sdf_path = args.sdf.resolve()
        if not sdf_path.exists():
            console_logger.error(f"SDF file does not exist: {sdf_path}")
            sys.exit(1)

        output_dir = args.output or Path(f"/tmp/mujoco_{sdf_path.stem}")
        console_logger.info(f"Converting SDF to MuJoCo: {sdf_path}")

        output_path = export_sdf_to_mujoco(
            sdf_path=sdf_path, output_dir=output_dir, is_static=args.static
        )

        if not args.skip_validation:
            console_logger.info("Validating export...")
            if not validate_mujoco_export(output_path):
                console_logger.error("Export validation failed")
                sys.exit(1)

        console_logger.info(f"\nExport complete!")
        console_logger.info(f"  Scene file: {output_path}")
        console_logger.info(f"  Meshes dir: {output_dir / 'meshes'}")
        console_logger.info(f"\nTo view in MuJoCo:")
        console_logger.info(f"  python -m mujoco.viewer --mjcf={output_path}")

        # Export to USD if requested.
        if args.usd:
            export_to_usd(
                output_path,
                output_dir,
                apply_isaac_sim_fix=not args.skip_isaac_sim_fix,
            )

        return

    # Regular scene export mode.
    if not args.scene_path:
        parser.error("scene_path is required unless --sdf is specified")

    scene_path = args.scene_path.resolve()
    if not scene_path.exists():
        console_logger.error(f"Scene path does not exist: {scene_path}")
        sys.exit(1)

    output_dir = args.output or scene_path / "mujoco"

    console_logger.info(f"Loading scene from: {scene_path}")

    house_state_path = scene_path / "combined_house" / "house_state.json"
    dmd_path = scene_path / "combined_house" / "house.dmd.yaml"

    console_logger.info(f"Exporting to: {output_dir}")
    if house_state_path.exists():
        house = load_house_from_directory(scene_path)
        output_path = export_scene_to_mujoco(
            house=house,
            output_dir=output_dir,
            include_floor_plan=not args.no_floor_plan,
            weld_furniture=args.weld_furniture,
        )
    elif dmd_path.exists():
        console_logger.info(
            "house_state.json missing; exporting directly from house.dmd.yaml"
        )
        output_path = export_dmd_scene_to_mujoco(
            scene_dir=scene_path,
            dmd_path=dmd_path,
            output_dir=output_dir,
            include_floor_plan=not args.no_floor_plan,
            weld_furniture=args.weld_furniture,
        )
    else:
        console_logger.error(
            f"Missing scene metadata: expected {house_state_path} or {dmd_path}"
        )
        sys.exit(1)

    # Validate export.
    if not args.skip_validation:
        console_logger.info("Validating export...")
        if not validate_mujoco_export(output_path):
            console_logger.error("Export validation failed")
            sys.exit(1)

    console_logger.info(f"\nExport complete!")
    console_logger.info(f"  Scene file: {output_path}")
    console_logger.info(f"  Meshes dir: {output_dir / 'meshes'}")
    console_logger.info(f"\nTo load in MuJoCo:")
    console_logger.info(f"  import mujoco")
    console_logger.info(f"  model = mujoco.MjModel.from_xml_path('{output_path}')")

    # Export to USD if requested.
    if args.usd:
        export_to_usd(
            output_path,
            output_dir,
            apply_isaac_sim_fix=not args.skip_isaac_sim_fix,
        )


if __name__ == "__main__":
    main()
