"""Tests for content-addressed remote geometry artifact transport."""

import hashlib
import io
import threading
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from scenesmith.agent_utils.geometry_generation_server.artifact_store import (
    ArtifactStore,
)
from scenesmith.agent_utils.geometry_generation_server.client import (
    GeometryGenerationClient,
)
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.geometry_generation_server.execution_provider import (
    GeometryWorkerTarget,
)
from scenesmith.agent_utils.geometry_generation_server.server.server_app import (
    GeometryGenerationApp,
)


class _PortableProvider:
    key = "portable-test"

    def targets(self):
        return (
            GeometryWorkerTarget(
                worker_id="worker-0",
                provider=self.key,
                device_id=None,
                label="portable/test",
                environment=(),
            ),
        )

    def process_start_method(self) -> str:
        return "spawn"


class TestArtifactStore(unittest.TestCase):
    def test_publish_is_content_addressed_and_idempotent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.glb"
            source.write_bytes(b"glTF fixture")
            store = ArtifactStore(root / "store")

            first = store.publish_path(source, filename="dragon.glb")
            second = store.publish_path(source, filename="dragon.glb")

            expected = hashlib.sha256(b"glTF fixture").hexdigest()
            self.assertEqual(first.artifact_id, expected)
            self.assertEqual(second.artifact_id, expected)
            self.assertEqual(store.resolve(expected).read_bytes(), b"glTF fixture")

    def test_invalid_artifact_identifier_cannot_escape_store(self) -> None:
        with TemporaryDirectory() as temporary:
            store = ArtifactStore(Path(temporary))

            with self.assertRaisesRegex(ValueError, "artifact identifier"):
                store.resolve("../../etc/passwd")

    def test_upload_budget_is_enforced_before_publish(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "large.bin"
            source.write_bytes(b"12345")
            store = ArtifactStore(root / "store", max_artifact_bytes=4)

            with self.assertRaisesRegex(ValueError, "budget"):
                store.publish_path(source)

            self.assertEqual(list((root / "store" / "objects").iterdir()), [])


class TestRemoteGeometryClient(unittest.TestCase):
    def test_non_loopback_transport_requires_https(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            GeometryGenerationClient(
                host="geometry.example",
                port=7443,
                scheme="http",
                transport="artifact",
                auth_token="secret",
            )

    def test_non_loopback_transport_requires_authentication(self) -> None:
        with self.assertRaisesRegex(ValueError, "auth token"):
            GeometryGenerationClient(
                host="geometry.example",
                port=7443,
                scheme="https",
                transport="artifact",
            )

    def test_artifact_transport_uploads_and_downloads_content(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            image.write_bytes(b"PNG fixture")
            output_dir = root / "output"
            session = MagicMock()

            upload_response = MagicMock()
            upload_response.raise_for_status.return_value = None
            upload_response.json.return_value = {"artifact_id": "a" * 64}
            generation_response = MagicMock()
            generation_response.raise_for_status.return_value = None
            generation_response.iter_lines.return_value = [
                (
                    b'{"index": 0, "status": "success", "data": '
                    b'{"artifact_id": "' + b"b" * 64 + b'", '
                    b'"filename": "chair.glb"}}'
                )
            ]
            session.post.side_effect = [upload_response, generation_response]
            download_response = MagicMock()
            download_response.raise_for_status.return_value = None
            download_response.iter_content.return_value = [b"glTF", b" result"]
            session.get.return_value = download_response
            client = GeometryGenerationClient(
                host="geometry.example",
                port=7443,
                scheme="https",
                transport="artifact",
                auth_token="secret",
                session=session,
            )
            request = GeometryGenerationServerRequest(
                image_path=str(image),
                output_dir=str(output_dir),
                output_filename="chair.glb",
                prompt="chair",
            )

            results = list(client.generate_geometries([request]))

            result_path = Path(results[0][1].geometry_path)
            self.assertEqual(result_path, (output_dir / "chair.glb").resolve())
            self.assertEqual(result_path.read_bytes(), b"glTF result")
            upload_call = session.post.call_args_list[0]
            self.assertEqual(
                upload_call.args[0], "https://geometry.example:7443/v1/artifacts"
            )
            self.assertEqual(
                upload_call.kwargs["headers"]["Authorization"], "Bearer secret"
            )
            generation_payload = session.post.call_args_list[1].kwargs["json"]
            self.assertEqual(generation_payload[0]["input_artifact"], "a" * 64)
            self.assertNotIn("image_path", generation_payload[0])
            self.assertNotIn("output_dir", generation_payload[0])

    def test_download_budget_removes_partial_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            image.write_bytes(b"PNG fixture")
            output_dir = root / "output"
            session = MagicMock()
            upload_response = MagicMock()
            upload_response.raise_for_status.return_value = None
            upload_response.json.return_value = {"artifact_id": "a" * 64}
            generation_response = MagicMock()
            generation_response.raise_for_status.return_value = None
            generation_response.iter_lines.return_value = [
                (
                    b'{"index": 0, "status": "success", "data": '
                    b'{"artifact_id": "' + b"b" * 64 + b'", '
                    b'"filename": "chair.glb"}}'
                )
            ]
            session.post.side_effect = [upload_response, generation_response]
            download_response = MagicMock()
            download_response.raise_for_status.return_value = None
            download_response.headers = {}
            download_response.iter_content.return_value = [b"1234", b"5"]
            session.get.return_value = download_response
            client = GeometryGenerationClient(
                host="geometry.example",
                port=7443,
                scheme="https",
                transport="artifact",
                auth_token="secret",
                session=session,
                max_download_bytes=4,
            )
            request = GeometryGenerationServerRequest(
                image_path=str(image),
                output_dir=str(output_dir),
                output_filename="chair.glb",
                prompt="chair",
            )

            with self.assertRaisesRegex(ValueError, "download budget"):
                list(client.generate_geometries([request]))

            self.assertFalse((output_dir / "chair.glb").exists())
            self.assertEqual(list(output_dir.iterdir()), [])


class TestArtifactServerContract(unittest.TestCase):
    def test_invalid_request_does_not_stop_coordinator(self) -> None:
        app = GeometryGenerationApp(
            execution_provider=_PortableProvider(),
            preload_pipeline=False,
        )
        request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="chair",
        )
        completed = threading.Event()
        results = []
        calls = 0

        def submit_request(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("invalid runtime override")
            kwargs["callback"](
                kwargs["request_index"],
                ("success", {"geometry_path": "/test/output/chair.glb"}),
            )
            app._processing_active = False
            completed.set()

        app._worker_pool.submit_request = submit_request
        app._worker_pool._running = True
        app._scheduler.add_batch(
            "first", [request], lambda i, r: results.append((i, r)), 0
        )
        app._scheduler.add_batch(
            "second", [request], lambda i, r: results.append((i, r)), 0
        )
        app._processing_active = True
        coordinator = threading.Thread(target=app._process_queue)
        coordinator.start()

        self.assertTrue(completed.wait(timeout=2))
        coordinator.join(timeout=2)
        self.assertFalse(coordinator.is_alive())
        self.assertEqual(calls, 2)
        self.assertEqual(results[0][1][0], "error")
        self.assertEqual(results[1][1][0], "success")

    def test_artifact_routes_require_auth_and_stage_only_server_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            store = ArtifactStore(Path(temporary) / "store")
            app = GeometryGenerationApp(
                execution_provider=_PortableProvider(),
                preload_pipeline=False,
                artifact_store=store,
                auth_token="secret",
            )
            client = app.test_client()

            unauthorized = client.post(
                "/v1/artifacts",
                data={"file": (io.BytesIO(b"image"), "input.png")},
            )
            uploaded = client.post(
                "/v1/artifacts",
                data={"file": (io.BytesIO(b"image"), "input.png")},
                headers={"Authorization": "Bearer secret"},
            )

            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(uploaded.status_code, 201)
            artifact_id = uploaded.get_json()["artifact_id"]
            app._stream_requests = MagicMock(return_value=("queued", 202))

            response = client.post(
                "/v1/generate_geometries",
                json=[
                    {
                        "input_artifact": artifact_id,
                        "prompt": "chair",
                        "output_filename": "chair.glb",
                    }
                ],
                headers={"Authorization": "Bearer secret"},
            )

            self.assertEqual(response.status_code, 202)
            queued_request = app._stream_requests.call_args.args[0][0]
            self.assertEqual(
                Path(queued_request.image_path), store.resolve(artifact_id)
            )
            self.assertTrue(
                Path(queued_request.output_dir).is_relative_to(store.root / "jobs")
            )

            generated = Path(queued_request.output_dir) / "chair.glb"
            generated.write_bytes(b"mesh")
            transform = app._stream_requests.call_args.args[1]
            transformed = transform(0, {"geometry_path": str(generated)})
            self.assertEqual(
                store.resolve(transformed["artifact_id"]).read_bytes(), b"mesh"
            )
            self.assertFalse(Path(queued_request.output_dir).exists())

    def test_artifact_result_must_be_contained_by_its_server_job(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ArtifactStore(root / "store")
            source = root / "input.png"
            source.write_bytes(b"image")
            artifact_id = store.publish_path(source).artifact_id
            app = GeometryGenerationApp(
                execution_provider=_PortableProvider(),
                preload_pipeline=False,
                artifact_store=store,
            )
            app._stream_requests = MagicMock(return_value=("queued", 202))
            response = app.test_client().post(
                "/v1/generate_geometries",
                json=[{"input_artifact": artifact_id, "prompt": "chair"}],
            )
            outside = root / "outside.glb"
            outside.write_bytes(b"mesh")
            transform = app._stream_requests.call_args.args[1]

            with self.assertRaisesRegex(ValueError, "outside its server job"):
                transform(0, {"geometry_path": str(outside)})

            self.assertTrue(outside.exists())

    def test_artifact_request_rejects_client_filesystem_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            store = ArtifactStore(Path(temporary) / "store")
            source = Path(temporary) / "input.png"
            source.write_bytes(b"image")
            artifact_id = store.publish_path(source).artifact_id
            app = GeometryGenerationApp(
                execution_provider=_PortableProvider(),
                preload_pipeline=False,
                artifact_store=store,
            )

            response = app.test_client().post(
                "/v1/generate_geometries",
                json=[
                    {
                        "input_artifact": artifact_id,
                        "image_path": "/etc/passwd",
                        "prompt": "chair",
                    }
                ],
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("unknown fields", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
