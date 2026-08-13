#!/usr/bin/env python3
"""Generate and serve the auto-discovered semantic scene gallery."""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import sys
import webbrowser

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Keep the documented direct-script entry point independent of installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.semantic_gallery.generate_gallery import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_TRIAL_DIRECTORY,
    generate_gallery,
)


class GalleryRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, message: str, *args) -> None:
        print(f"[gallery] {self.address_string()} {message % args}")


def manifest_asset_paths(
    generated_root: Path | str, manifest: dict[str, object]
) -> tuple[Path, ...]:
    """Resolve and validate every browser-loaded artifact in a gallery manifest."""

    root = Path(generated_root).resolve()
    paths: list[Path] = []
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("gallery manifest scenes must be an array")
    for scene in scenes:
        try:
            relative_paths = [
                scene["shell"]["mesh_path"],
                *(detail["mesh_path"] for detail in scene["details"]),
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError("gallery manifest has an invalid scene product") from exc
        for relative in relative_paths:
            if not isinstance(relative, str):
                raise ValueError("gallery asset paths must be strings")
            path = (root / relative).resolve()
            if root not in path.parents:
                raise ValueError(f"gallery asset escapes generated root: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"gallery asset is missing: {path}")
            paths.append(path)
    return tuple(paths)


def _port_available(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        return probe.connect_ex((host, port)) != 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--trials-dir", type=Path, default=DEFAULT_TRIAL_DIRECTORY)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if not args.no_generate:
        print("Discovering and compiling retained semantic scenes…")
        generate_gallery(args.output_dir, trial_directory=args.trials_dir)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Gallery manifest is missing: {manifest_path}\n"
            "Run without --no-generate to build it."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_asset_paths(args.output_dir, manifest)
    if args.output_dir.resolve() != (root / "generated").resolve():
        raise SystemExit(
            "The browser viewer serves examples/semantic_gallery/generated; "
            "use the default --output-dir when serving."
        )
    if not _port_available(args.host, args.port):
        raise SystemExit(f"Port {args.port} is already in use on {args.host}")

    handler = (
        lambda *handler_args, **handler_kwargs: GalleryRequestHandler(  # noqa: E731
            *handler_args, directory=str(root), **handler_kwargs
        )
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/viewer.html"
    print(f"Semantic Scene Gallery is available at:\n  {url}")
    print("Press Ctrl-C to stop the server.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping gallery.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
