#!/usr/bin/env python3
"""Render a measured Cycles frame and write an honest CPU device receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import bpy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame_path = arguments.output.with_suffix(".png")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.resolution_x = 32
    scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(frame_path)

    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "CPU"
    active_devices = [
        {"name": device.name, "type": device.type}
        for device in preferences.devices
        if device.use
    ]
    if not active_devices or any(device["type"] != "CPU" for device in active_devices):
        raise RuntimeError(f"Cycles did not expose an exclusive CPU device: {active_devices}")

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    bpy.ops.object.light_add(type="AREA", location=(2.0, -2.0, 3.0))
    bpy.context.object.data.energy = 300.0
    bpy.ops.object.camera_add(location=(3.5, -3.5, 2.5))
    camera = bpy.context.object
    scene.camera = camera
    direction = bpy.data.objects["Cube"].location - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    started = time.monotonic()
    bpy.ops.render.render(write_still=True)
    elapsed = time.monotonic() - started
    if not frame_path.is_file() or frame_path.stat().st_size == 0:
        raise RuntimeError("Cycles CPU qualification produced no rendered frame")
    payload = {
        "contractVersion": 1,
        "activeBackend": "CPU",
        "activeDevices": active_devices,
        "renderEngine": scene.render.engine,
        "cyclesDevice": scene.cycles.device,
        "blenderVersion": bpy.app.version_string,
        "renderSeconds": round(elapsed, 6),
        "renderSha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "renderFile": frame_path.name,
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
