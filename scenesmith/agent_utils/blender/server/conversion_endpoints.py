"""Mesh canonicalization and format-conversion HTTP endpoints."""

import logging
import time

from pathlib import Path

import flask

from scenesmith.agent_utils.blender.geometry.canonicalization import (
    canonicalize_mesh_impl,
)
from scenesmith.agent_utils.blender.geometry.mesh_conversion import (
    convert_glb_to_gltf_impl,
)

console_logger = logging.getLogger(__name__)


class BlenderConversionEndpointsMixin:
    def _canonicalize_endpoint(self) -> flask.Response:
        """Canonicalize mesh orientation and placement.

        This endpoint receives mesh parameters and canonicalizes the mesh to
        standard orientation using Blender. Unlike rendering endpoints, this
        is self-contained and doesn't require pre-set configuration.

        JSON body:
            input_path: Path to input GLTF file.
            output_path: Path where canonicalized GLTF will be saved.
            up_axis: Up axis in Blender coordinates (e.g., "+Z", "-Y").
            front_axis: Front axis in Blender coordinates (e.g., "+Y", "+X").
            object_type: Type of object (determines placement strategy).
                One of: "furniture", "manipuland", "wall_mounted", "ceiling_mounted".

        Returns:
            JSON response with status and output path.
        """
        request_start = time.time()
        console_logger.info("Canonicalize request received")

        try:
            # Parse JSON body.
            data = flask.request.get_json()
            if not data:
                flask.abort(400, description="Missing JSON body")

            # Validate required fields.
            required_fields = ["input_path", "output_path", "up_axis", "front_axis"]
            for field in required_fields:
                if field not in data:
                    flask.abort(400, description=f"Missing required field: {field}")

            input_path = Path(data["input_path"])
            output_path = Path(data["output_path"])
            up_axis = data["up_axis"]
            front_axis = data["front_axis"]
            object_type = data.get("object_type", "furniture")

            canonicalize_mesh_impl(
                input_path=input_path,
                output_path=output_path,
                up_axis=up_axis,
                front_axis=front_axis,
                object_type=object_type,
            )

            total_time = time.time() - request_start
            console_logger.info(
                f"Canonicalization completed in {total_time:.2f}s: {output_path}"
            )

            return flask.jsonify(
                {
                    "status": "success",
                    "output_path": str(output_path),
                }
            )

        except FileNotFoundError as e:
            console_logger.error(f"Canonicalization failed: {e}")
            flask.abort(404, description=str(e))
        except Exception as e:
            console_logger.error(f"Canonicalization failed: {e}")
            flask.abort(500, description=f"Canonicalization failed: {e}")

    def _convert_glb_to_gltf_endpoint(self) -> flask.Response:
        """Convert GLB file to GLTF with separate textures using Blender.

        This endpoint receives mesh parameters and converts the GLB to GLTF format
        with separate texture files. Running in the BlenderServer subprocess ensures
        bpy crashes don't kill the main scene worker process.

        JSON body:
            input_path: Path to input GLB or GLTF file.
            output_path: Path where converted GLTF will be saved.
            export_yup: If True, converts to Y-up GLTF standard. Default True.

        Returns:
            JSON response with status and output path.
        """
        request_start = time.time()
        console_logger.info("GLB to GLTF conversion request received")

        try:
            # Parse JSON body.
            data = flask.request.get_json()
            if not data:
                flask.abort(400, description="Missing JSON body")

            # Validate required fields.
            required_fields = ["input_path", "output_path"]
            for field in required_fields:
                if field not in data:
                    flask.abort(400, description=f"Missing required field: {field}")

            input_path = Path(data["input_path"])
            output_path = Path(data["output_path"])
            export_yup = data.get("export_yup", True)

            # Import and run bpy conversion (runs in this BlenderServer subprocess).
            convert_glb_to_gltf_impl(
                input_path=input_path,
                output_path=output_path,
                export_yup=export_yup,
            )

            total_time = time.time() - request_start
            console_logger.info(
                f"GLB to GLTF conversion completed in {total_time:.2f}s: {output_path}"
            )

            return flask.jsonify(
                {
                    "status": "success",
                    "output_path": str(output_path),
                }
            )

        except FileNotFoundError as e:
            console_logger.error(f"GLB to GLTF conversion failed: {e}")
            flask.abort(404, description=str(e))
        except Exception as e:
            console_logger.error(f"GLB to GLTF conversion failed: {e}")
            flask.abort(500, description=f"GLB to GLTF conversion failed: {e}")
