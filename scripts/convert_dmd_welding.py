#!/usr/bin/env python3
"""Convert DMD file welding configurations.

This script converts Drake Model Directive (DMD) files between different welding
configurations. It supports three modes:

- nothing: Only wall/ceiling-mounted objects welded (furniture FREE, composites FREE)
- furniture: Furniture welded, composites FREE
- all: Everything welded (furniture + manipulands)

The script requires house_state.json metadata to determine object types. It will
fail with an error if the metadata is not found or if a model is not found in
the metadata (no fallback heuristics).

Example usage:
    python scripts/convert_dmd_welding.py combined_house/house.dmd.yaml -m furniture
    python scripts/convert_dmd_welding.py house.dmd.yaml -m nothing -o house_free.dmd.yaml
"""

import argparse
import logging

from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
console_logger = logging.getLogger(__name__)

from scripts.scene_conversion.conversion import convert_dmd
from scripts.scene_conversion.dmd_io import (
    build_object_registry,
    load_house_state,
    parse_dmd_yaml,
    write_dmd_yaml,
)

# Object types that are always welded (regardless of mode).
ALWAYS_WELDED_TYPES = {"wall_mounted", "ceiling_mounted"}

# Object types that are free in all modes.
ALWAYS_FREE_TYPES = {"manipuland"}

# Asset sources that are always welded (regardless of mode).
# Thin coverings (rugs, carpets, tablecloths) have no collision geometry,
# so they must remain welded to avoid unrealistic physics behavior.
ALWAYS_WELDED_ASSET_SOURCES = {"thin_covering"}


def main():
    parser = argparse.ArgumentParser(
        description="Convert DMD file welding configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Path to input DMD YAML file")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["nothing", "furniture", "all"],
        default="nothing",
        help="Welding mode (default: nothing)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: INPUT with _<mode> suffix)",
    )
    parser.add_argument(
        "--scene-state",
        type=Path,
        default=None,
        help="Path to house_state.json (auto-detected if in same dir)",
    )

    args = parser.parse_args()

    # Determine house_state.json path.
    if args.scene_state:
        state_path = args.scene_state
    else:
        # Auto-detect: look in same directory as input.
        state_path = args.input.parent / "house_state.json"

    # Determine output path.
    if args.output:
        output_path = args.output
    else:
        stem = args.input.stem
        if stem.endswith(f"_{args.mode}"):
            # Already has suffix, don't add another.
            output_path = args.input.parent / f"{stem}.dmd.yaml"
        else:
            # Remove existing mode suffix if present.
            for mode_suffix in ["_nothing", "_furniture", "_all"]:
                if stem.endswith(mode_suffix):
                    stem = stem[: -len(mode_suffix)]
                    break
            output_path = args.input.parent / f"{stem}_{args.mode}.dmd.yaml"

    console_logger.info(f"Loading house state from {state_path}")
    house_state = load_house_state(state_path)

    console_logger.info("Building object registry from metadata")
    object_registry = build_object_registry(house_state)
    console_logger.info(f"Found {len(object_registry)} objects in registry")

    console_logger.info(f"Parsing DMD file {args.input}")
    directives = parse_dmd_yaml(args.input)
    console_logger.info(f"Parsed {len(directives)} directives")

    console_logger.info(f"Converting to mode '{args.mode}'")
    converted = convert_dmd(directives, object_registry, args.mode)
    console_logger.info(f"Converted to {len(converted)} directives")

    console_logger.info(f"Writing output to {output_path}")
    write_dmd_yaml(converted, output_path)

    console_logger.info("Done!")


if __name__ == "__main__":
    main()
