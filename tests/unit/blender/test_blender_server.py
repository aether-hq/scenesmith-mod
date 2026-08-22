import io
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import flask
import requests

from scenesmith.agent_utils.blender import (
    BlenderRenderApp,
    BlenderRenderer,
    BlenderServer,
    RenderParams,
)
from scenesmith.agent_utils.blender.server.server_manager import (
    find_available_port,
    is_port_available,
)


class TestBlenderRenderApp(unittest.TestCase):
    """Test cases for BlenderRenderApp Flask application."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_blender_render_app_init(self):
        """Test BlenderRenderApp initialization."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)
        self.assertEqual(app.name, "scenesmith_blender_render")
        self.assertEqual(app._temp_dir, self.temp_dir)
        self.assertIsInstance(app._blender, BlenderRenderer)

    def test_blender_render_app_init_with_files(self):
        """Test BlenderRenderApp initialization with blend and settings files."""
        blend_file = Path("/tmp/test.blend")
        settings_file = Path("/tmp/settings.py")

        app = BlenderRenderApp(
            temp_dir=self.temp_dir,
            blend_file=blend_file,
            bpy_settings_file=settings_file,
        )
        self.assertEqual(app._blender._blend_file, blend_file)
        self.assertEqual(app._blender._bpy_settings_file, settings_file)

    def test_root_endpoint_returns_banner(self):
        """Test that _root_endpoint returns HTML banner page."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)
        response = app._root_endpoint()

        self.assertIsInstance(response, str)
        self.assertIn("<!doctype html>", response.lower())
        self.assertIn("blender", response.lower())

    def test_render_endpoint_handles_post_request(self):
        """Test that _render_endpoint handles POST requests and returns image."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        with app.test_request_context("/render", method="POST"):
            with patch.object(app, "_parse_params") as mock_parse:
                with patch.object(app, "_render") as mock_render:
                    mock_buffer = io.BytesIO(b"fake_png_data")
                    mock_render.return_value = mock_buffer

                    response = app._render_endpoint()

                    mock_parse.assert_called_once()
                    mock_render.assert_called_once()
                    self.assertIsInstance(response, flask.Response)

    def test_parse_params_converts_form_data(self):
        """Test that _parse_params correctly parses Flask request form data."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        # Mock request with form data.
        mock_request = Mock(spec=flask.Request)
        mock_request.form = {
            "scene_sha256": "abc123",
            "image_type": "color",
            "width": "640",
            "height": "480",
            "near": "0.1",
            "far": "100.0",
            "focal_x": "320.0",
            "focal_y": "320.0",
            "fov_x": "1.047",
            "fov_y": "0.785",
            "center_x": "320.0",
            "center_y": "240.0",
        }
        mock_files = {"scene": Mock()}

        # Mock save method to create actual file for stat() call.
        def mock_save(path):
            Path(path).touch()

        mock_files["scene"].save = mock_save
        mock_request.files = mock_files

        params = app._parse_params(mock_request)

        self.assertIsInstance(params, RenderParams)
        self.assertEqual(params.scene_sha256, "abc123")
        self.assertEqual(params.image_type, "color")
        self.assertEqual(params.width, 640)
        self.assertEqual(params.height, 480)

    def test_render_returns_png_buffer(self):
        """Test that _render calls BlenderRenderer and returns PNG buffer."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)
        params = RenderParams(
            scene=Path("/tmp/test.gltf"),
            scene_sha256="abc123",
            image_type="color",
            width=640,
            height=480,
            near=0.1,
            far=100.0,
            focal_x=320.0,
            focal_y=320.0,
            fov_x=1.047,
            fov_y=0.785,
            center_x=320.0,
            center_y=240.0,
        )

        with patch.object(app._blender, "render_image") as mock_render:
            # Mock the temporary file creation and PNG data.
            mock_png_data = b"fake_png_data"

            with (
                patch("tempfile.NamedTemporaryFile") as mock_tempfile,
                patch.object(Path, "read_bytes", return_value=mock_png_data),
                patch.object(Path, "exists", return_value=True),
                patch.object(Path, "unlink"),
            ):

                # Setup the mock temp file.
                mock_temp = MagicMock()
                mock_temp.name = "/tmp/tmpfile.png"
                mock_tempfile.return_value.__enter__.return_value = mock_temp

                buffer = app._render(params)

                self.assertIsInstance(buffer, io.BytesIO)
                # Verify render_image was called with params and some temp path.
                self.assertEqual(mock_render.call_count, 1)
                call_args = mock_render.call_args[0]
                self.assertEqual(call_args[0], params)
                self.assertIsInstance(call_args[1], Path)

    def test_url_rules_configured(self):
        """Test that URL rules are properly configured."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        # Check that routes are registered.
        rule_endpoints = [rule.endpoint for rule in app.url_map.iter_rules()]
        self.assertIn("/render", rule_endpoints)

        # Check that POST method is allowed for render endpoint.
        render_rule = None
        for rule in app.url_map.iter_rules():
            if rule.endpoint == "/render":
                render_rule = rule
                break

        self.assertIsNotNone(render_rule)
        self.assertIn("POST", render_rule.methods)


class TestBlenderServer(unittest.TestCase):
    """Test cases for BlenderServer lifecycle manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.blend_file = self.temp_dir / "test.blend"
        self.settings_file = self.temp_dir / "settings.py"

    def test_blender_server_init(self):
        """Test BlenderServer initialization with default parameters."""
        server = BlenderServer()
        self.assertEqual(server._host, "127.0.0.1")
        self.assertIsNone(server._port)
        self.assertEqual(server._port_range, (8000, 8050))
        self.assertIsNone(server._actual_port)
        self.assertIsNone(server._blend_file)
        self.assertIsNone(server._bpy_settings_file)
        self.assertEqual(server._server_startup_delay, 3.0)
        self.assertEqual(server._port_cleanup_delay, 2.0)
        self.assertIsNone(server._server_process)
        self.assertIsNone(server._temp_dir)
        self.assertFalse(server._running)

    def test_blender_server_init_custom_params(self):
        """Test BlenderServer initialization with custom parameters."""
        # Create the files so they exist for validation.
        self.blend_file.touch()
        self.settings_file.touch()

        server = BlenderServer(
            host="192.168.1.100",
            port=9000,
            blend_file=self.blend_file,
            bpy_settings_file=self.settings_file,
        )
        self.assertEqual(server._host, "192.168.1.100")
        self.assertEqual(server._port, 9000)
        self.assertIsNone(server._port_range)
        self.assertIsNone(server._actual_port)
        self.assertEqual(server._blend_file, self.blend_file)
        self.assertEqual(server._bpy_settings_file, self.settings_file)

    def test_blender_server_init_invalid_files(self):
        """Test BlenderServer initialization with non-existent files raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BlenderServer(blend_file=Path("/nonexistent/file.blend"))
        self.assertIn("Blend file not found", str(context.exception))

        with self.assertRaises(ValueError) as context:
            BlenderServer(bpy_settings_file=Path("/nonexistent/settings.py"))
        self.assertIn("Bpy settings file not found", str(context.exception))

    def test_blender_server_init_with_port_range(self):
        """Test BlenderServer initialization with port range."""
        server = BlenderServer(port_range=(9000, 9005))
        self.assertEqual(server._host, "127.0.0.1")
        self.assertIsNone(server._port)
        self.assertEqual(server._port_range, (9000, 9005))
        self.assertIsNone(server._actual_port)

    def test_blender_server_init_both_port_and_range_raises_error(self):
        """Test BlenderServer initialization with both port and port_range raises
        ValueError."""
        with self.assertRaises(ValueError) as context:
            BlenderServer(port=8000, port_range=(9000, 9005))
        self.assertIn("Cannot specify both port and port_range", str(context.exception))

    def test_is_running_initial_state(self):
        """Test that server is not running initially."""
        server = BlenderServer()
        self.assertFalse(server.is_running())

    def test_get_url_when_not_running(self):
        """Test that get_url raises RuntimeError when server is not running."""
        server = BlenderServer()
        with self.assertRaises(RuntimeError) as context:
            server.get_url()
        self.assertIn("Server is not running", str(context.exception))
        self.assertIn("status:", str(context.exception))

    def test_get_url_when_running(self):
        """Test get_url returns correct URL when server is marked as running."""
        server = BlenderServer(host="localhost", port=8080)
        server._running = True  # Simulate running state
        server._actual_port = 8080  # Set the actual port for URL generation
        url = server.get_url()
        self.assertEqual(url, "http://localhost:8080")

    @patch(
        "scenesmith.agent_utils.blender.server.server_manager.is_port_available",
        return_value=True,
    )
    @patch("tempfile.TemporaryDirectory")
    @patch("subprocess.Popen")
    @patch.object(Path, "exists", return_value=True)  # Mock standalone script exists
    def test_start_creates_process(
        self, mock_exists, mock_popen, mock_temp_dir, mock_port_available
    ):
        """Test that start creates temporary directory and process."""
        mock_temp_dir_instance = Mock()
        mock_temp_dir_instance.name = "/tmp/test"
        mock_temp_dir.return_value = mock_temp_dir_instance

        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process is still running
        mock_process.communicate.return_value = ("", "")  # Mock stdout/stderr
        mock_popen.return_value = mock_process

        server = BlenderServer(
            port=8000, server_startup_delay=0.0, port_cleanup_delay=0.0
        )
        server.start()

        # Should create temporary directory.
        mock_temp_dir.assert_called_once()

        # Should create and start process.
        mock_popen.assert_called_once()

        # Verify command contains expected parts.
        call_args = mock_popen.call_args[0][0]  # First positional arg
        self.assertIn("python", call_args[0])  # sys.executable contains "python"
        self.assertIn("standalone_server.py", call_args[1])
        self.assertIn("--host", call_args)
        self.assertIn("--port", call_args)
        self.assertIn("8000", call_args)  # Check specific port is used

        # Should mark as running.
        self.assertTrue(server.is_running())
        # Should set actual port.
        self.assertEqual(server._actual_port, 8000)

    @patch("tempfile.TemporaryDirectory")
    def test_stop_cleans_up_resources(self, mock_temp_dir):
        """Test that stop cleans up process and temporary directory."""
        mock_temp_dir_instance = Mock()
        mock_temp_dir.return_value = mock_temp_dir_instance

        server = BlenderServer(server_startup_delay=0.0, port_cleanup_delay=0.0)

        # Simulate running state.
        server._running = True
        server._temp_dir = mock_temp_dir_instance
        server._server_process = Mock()
        server._server_process.pid = 12345
        server._server_process.wait.return_value = 0

        # Capture the mock process before calling stop (since stop sets it to None).
        mock_process = server._server_process

        server.stop()

        # Should terminate process and cleanup.
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5.0)
        mock_temp_dir_instance.cleanup.assert_called_once()

        # Should mark as not running and clear references.
        self.assertFalse(server.is_running())
        self.assertIsNone(server._server_process)
        self.assertIsNone(server._temp_dir)

    def test_get_process_status_no_process(self):
        """Test get_process_status when no process exists."""
        server = BlenderServer()
        status = server.get_process_status()
        self.assertEqual(status, "No process")

    def test_get_process_status_running(self):
        """Test get_process_status when process is running."""
        server = BlenderServer()
        server._server_process = Mock()
        server._server_process.pid = 12345
        server._server_process.poll.return_value = None  # Still running

        status = server.get_process_status()
        self.assertEqual(status, "Running (PID 12345)")

    def test_get_process_status_exited(self):
        """Test get_process_status when process has exited."""
        server = BlenderServer()
        server._server_process = Mock()
        server._server_process.poll.return_value = 0  # Exited with code 0

        status = server.get_process_status()
        self.assertEqual(status, "Exited with code 0")

    def test_readiness_timeout_uses_monotonic_deadline_and_cleans_child(self):
        """A wall-clock rollback must not extend startup or orphan the child."""
        server = BlenderServer(port=8026, port_cleanup_delay=0.0)
        server._actual_port = 8026
        server._running = True
        server._server_process = Mock()
        server._server_process.pid = 42
        server._server_process.poll.return_value = None
        server._server_process.wait.return_value = 0
        ticks = iter((10.0, 10.0, 12.1))

        with (
            patch(
                "scenesmith.agent_utils.blender.server.server_manager.requests.get",
                side_effect=requests.ConnectionError("connection refused"),
            ) as get,
            patch(
                "scenesmith.agent_utils.blender.server.server_manager.time.monotonic",
                side_effect=lambda: next(ticks, 12.1),
            ),
            patch(
                "scenesmith.agent_utils.blender.server.server_manager.time.time",
                side_effect=(100.0, 40.0, 40.0),
            ),
            patch("scenesmith.agent_utils.blender.server.server_manager.time.sleep"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "failed to become ready within 2.0s.*connection refused"
            ):
                server.wait_until_ready(timeout=2.0)

        get.assert_called_once()
        self.assertFalse(server.is_running())
        self.assertIsNone(server._server_process)


class TestPortUtilities(unittest.TestCase):
    """Test cases for port availability utility functions."""

    def test_is_port_available_free_port(self):
        """Test is_port_available returns True for free ports."""

        # Test with a very high port that should be available.
        self.assertTrue(is_port_available("127.0.0.1", 65432))

    def test_find_available_port_finds_port(self):
        """Test find_available_port finds an available port in range."""
        # Use a high port range that should have available ports.
        port = find_available_port("127.0.0.1", (65400, 65410))
        self.assertIsNotNone(port)
        self.assertGreaterEqual(port, 65400)
        self.assertLessEqual(port, 65410)

    def test_find_available_port_returns_none_when_no_ports(self):
        """Test find_available_port returns None when no ports available."""
        # Mock socket.socket to always raise OSError (port unavailable).
        with patch("socket.socket") as mock_socket:
            mock_context = mock_socket.return_value.__enter__
            mock_context.return_value.bind.side_effect = OSError(
                "Address already in use"
            )

            port = find_available_port("127.0.0.1", (8000, 8002))
            self.assertIsNone(port)
