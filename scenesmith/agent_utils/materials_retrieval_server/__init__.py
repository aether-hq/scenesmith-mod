"""Materials retrieval server for semantic search over PBR material library.

This package provides a Flask-based server for CLIP-based semantic retrieval
over the AmbientCG material library. The server pattern avoids CUDA
multiprocessing issues by keeping the CLIP model in a single process.

Usage:
    # Programmatic usage (recommended for experiments)
    from scenesmith.agent_utils.materials_retrieval_server import (
        MaterialsRetrievalServer,
        MaterialsRetrievalClient,
        MaterialsRetrievalServerRequest,
    )

    server = MaterialsRetrievalServer(host="127.0.0.1", port=7018)
    server.start()
    server.wait_until_ready()

    client = MaterialsRetrievalClient(host="127.0.0.1", port=7018)
    requests = [MaterialsRetrievalServerRequest(
        material_description="warm hardwood floor",
        output_dir="/tmp/output",
    )]
    for index, response in client.retrieve_materials(requests):
        print(f"Retrieved: {response.results[0].material_path}")

    server.stop()

    # Standalone usage (for testing/debugging/microservice deployment)
    # python -m scenesmith.agent_utils.materials_retrieval_server.standalone_server
"""

__all__ = [
    "MaterialsRetrievalClient",
    "MaterialRetrievalResult",
    "MaterialsRetrievalServer",
    "MaterialsRetrievalServerRequest",
    "MaterialsRetrievalServerResponse",
]


def __getattr__(name: str):
    """Lazily load the CLIP/Flask server only when it is actually requested."""

    if name == "MaterialsRetrievalClient":
        from .client import MaterialsRetrievalClient

        return MaterialsRetrievalClient
    if name == "MaterialsRetrievalServer":
        from .server_manager import MaterialsRetrievalServer

        return MaterialsRetrievalServer
    if name in {
        "MaterialRetrievalResult",
        "MaterialsRetrievalServerRequest",
        "MaterialsRetrievalServerResponse",
    }:
        from .dataclasses import (
            MaterialRetrievalResult,
            MaterialsRetrievalServerRequest,
            MaterialsRetrievalServerResponse,
        )

        return {
            "MaterialRetrievalResult": MaterialRetrievalResult,
            "MaterialsRetrievalServerRequest": MaterialsRetrievalServerRequest,
            "MaterialsRetrievalServerResponse": MaterialsRetrievalServerResponse,
        }[name]
    raise AttributeError(name)
