"""Fix USD physics for Isaac Sim compatibility.

The mujoco-usd-converter (v0.1.0a3) generates PhysicsFixedJoint prims that
connect objects to the root Xform, but the root has no PhysicsRigidBodyAPI.
PhysX requires valid physics bodies on both sides of a joint, so the
constraint solver pulls everything to (0,0,0).

This script post-processes Physics.usda files to fix three object categories:

1. **Static objects** (walls, desks, beds): Remove all physics body APIs and
   joints, leaving only collision geometry. Isaac Sim treats these as static
   colliders.

2. **Dynamic objects** (mugs, books): Flatten nested rigid bodies by moving
   MassAPI from base_link to wrapper, removing inner RigidBodyAPI, and
   deleting the internal FixedJoint.

3. **Articulated objects** (wardrobes with doors, fridges): Promote invalid
   base-body Xforms to real rigid bodies, reparent articulated links as
   siblings when needed, preserve authored collision geometry by default, and
   recreate self-collision filters (mirroring MuJoCo's ``<contact><exclude>``
   pairs). Optionally, articulated collision can be regenerated from visual
   meshes using Isaac-compatible mesh approximations.

Usage:
    # Fix single scene USD directory.
    python scripts/fix_usd_isaac_sim.py /path/to/scene/mujoco/usd

    # Fix all scenes recursively with parallel workers.
    python scripts/fix_usd_isaac_sim.py /path/to/SceneAgent_Cleaned \\
        --recursive --workers 16
"""

import argparse
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

console_logger = logging.getLogger(__name__)

from scripts.usd_physics.workflow import _fix_single_scene, find_usd_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix USD physics for Isaac Sim compatibility"
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a single USD directory (containing Payload/), "
            "or a parent directory when using --recursive"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for all USD scenes under the given path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for recursive mode (default: 1)",
    )
    parser.add_argument(
        "--articulated-collision-mode",
        choices=["preserve", "convex-hull", "convex-decomposition"],
        default="preserve",
        help=(
            "How to handle collision on articulated objects only. "
            "'preserve' keeps authored collision meshes, while the other "
            "modes deactivate authored *collision* meshes and regenerate "
            "collision from *visual* meshes."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    usd_dirs = find_usd_dirs(base_path=args.path, recursive=args.recursive)
    if not usd_dirs:
        console_logger.error(f"No USD scenes found at {args.path}")
        return

    console_logger.info(f"Found {len(usd_dirs)} USD scene(s) to fix")

    total_counts: dict[str, int] = {
        "static": 0,
        "dynamic": 0,
        "articulated": 0,
    }
    errors = 0

    if args.workers > 1 and len(usd_dirs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _fix_single_scene,
                    d,
                    args.articulated_collision_mode,
                ): d
                for d in usd_dirs
            }
            for future in as_completed(futures):
                path, result = future.result()
                if isinstance(result, str):
                    console_logger.warning(f"{path}: {result}")
                    errors += 1
                else:
                    for k, v in result.items():
                        total_counts[k] += v
    else:
        for usd_dir in usd_dirs:
            path, result = _fix_single_scene(
                usd_dir,
                args.articulated_collision_mode,
            )
            if isinstance(result, str):
                console_logger.warning(f"{path}: {result}")
                errors += 1
            else:
                for k, v in result.items():
                    total_counts[k] += v

    console_logger.info(
        f"Done. Fixed {len(usd_dirs) - errors}/{len(usd_dirs)} scenes: "
        f"{total_counts['static']} static, "
        f"{total_counts['dynamic']} dynamic, "
        f"{total_counts['articulated']} articulated objects total"
    )
    if errors:
        console_logger.warning(f"{errors} scene(s) had errors")


if __name__ == "__main__":
    main()
