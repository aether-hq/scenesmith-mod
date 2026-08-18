#!/usr/bin/env python3
"""Serve and optionally open the prison escape WebGL walkthrough."""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import webbrowser

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DemoRequestHandler(SimpleHTTPRequestHandler):
    """Static handler rooted at the example with quiet, useful logging."""

    def __init__(self, *args, directory: str, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, message: str, *args) -> None:
        print(f"[viewer] {self.address_string()} {message % args}")


def _port_available(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        return probe.connect_ex((host, port)) != 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manifest_path = root / "generated" / "manifest.json"
    required = (
        root / "viewer.html",
        manifest_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Generated scene assets are missing. Run generate_scene.py first:\n"
            ".venv/bin/python -m examples.prison_escape.generate_scene\n"
            f"Missing:\n{formatted}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated_root = manifest_path.parent.resolve()
        artifact_paths = tuple(
            (generated_root / manifest[product][key]).resolve()
            for product in ("architecture", "tunnel")
            for key in ("mesh_path", "sdf_path")
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Generated manifest is invalid: {manifest_path}") from exc
    if any(
        generated_root not in path.parents or not path.is_file()
        for path in artifact_paths
    ):
        raise SystemExit(
            "Generated manifest references missing or unsafe architecture artifacts. "
            "Run generate_scene.py again."
        )
    if not _port_available(args.host, args.port):
        raise SystemExit(f"Port {args.port} is already in use on {args.host}")

    handler = lambda *handler_args, **handler_kwargs: DemoRequestHandler(  # noqa: E731
        *handler_args, directory=str(root), **handler_kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/viewer.html"
    print(f"The Long Way Out is available at:\n  {url}")
    print("Press Ctrl-C to stop the server.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
