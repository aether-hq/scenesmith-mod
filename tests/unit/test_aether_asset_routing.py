"""Static contract checks for per-brief deterministic asset routing."""

from __future__ import annotations

import ast
from pathlib import Path

from scenesmith.aether.assets import compile_asset_spec
from scenesmith.aether.runtime import SceneSmithCompletionRuntime


ROOT = Path(__file__).parents[2]


def _function_source(path: Path, class_name: str, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name:
                    return ast.get_source_segment(source, child) or ""
    raise AssertionError(f"{class_name}.{function_name} not found in {path}")


def test_asset_manager_exposes_typed_items_without_llm_analysis():
    source = _function_source(
        ROOT / "scenesmith/agent_utils/asset_manager.py",
        "AssetManager",
        "generate_assets_from_typed_items",
    )
    assert "_generate_assets_with_router" not in source
    assert "_generate_items_with_validation" in source
    assert "item.object_type" in source


def test_router_honors_each_typed_source_in_order():
    source = _function_source(
        ROOT / "scenesmith/agent_utils/asset_router/router.py",
        "AssetRouter",
        "generate_with_validation",
    )
    assert "for strategy in item.strategies" in source
    assert '{"generated", "hssd", "objaverse", "sam3d"}' in source
    assert "asset_source_override" in source
    assert "backend" in source


def test_completion_brief_compiles_to_concrete_scene_smith_sources_without_an_llm():
    spec = compile_asset_spec(
        {"operation": "populate-surfaces"},
        {
            "variant_id": "amber-bottle",
            "short_name": "amber-bottle",
            "description": "A scuffed amber service bottle with a paper label",
            "dimensions_m": [0.08, 0.08, 0.28],
            "source_order": ["curated", "hssd", "objaverse", "sam3d"],
        },
    )
    assert spec.object_type == "manipuland"
    assert spec.strategies == ("hssd", "objaverse", "sam3d")


def test_router_enabled_manager_constructs_every_ordered_source_client():
    source = _function_source(
        ROOT / "scenesmith/agent_utils/asset_manager.py",
        "AssetManager",
        "__init__",
    )
    assert 'self.general_asset_source == "generated" or router_enabled' in source
    assert 'self.general_asset_source == "hssd" or router_enabled' in source
    assert 'self.general_asset_source == "objaverse" or router_enabled' in source


def test_experiment_starts_every_router_enabled_source_server():
    path = ROOT / "scenesmith/experiments/indoor_scene_generation.py"
    geometry = _function_source(path, "IndoorSceneGenerationExperiment", "_start_geometry_server")
    hssd = _function_source(path, "IndoorSceneGenerationExperiment", "_start_hssd_server")
    objaverse = _function_source(path, "IndoorSceneGenerationExperiment", "_start_objaverse_server")
    for source in (geometry, hssd, objaverse):
        assert "router.enabled" in source


def test_scene_object_metadata_accepts_structured_semantic_provenance():
    source = (ROOT / "scenesmith/agent_utils/room.py").read_text()
    assert "metadata: dict[str, Any]" in source


def test_concrete_runtime_routes_one_template_then_uses_deterministic_adapter(monkeypatch):
    class Result:
        has_failures = False
        failed_assets = []
        successful_assets = [object()]

    calls = []

    def acquire(manager, operation, brief, **context):
        calls.append((manager, operation, brief, context))
        return Result()

    monkeypatch.setattr("scenesmith.aether.runtime.acquire_completion_assets", acquire)

    class Scene:
        scene_dir = Path("fixture-scene")

        def to_state_dict(self):
            return {"objects": {}}

        def restore_from_state_dict(self, state):
            self.restored = state

    class Adapter:
        def place(self, scene, asset, operation, brief, *, instance_index, round_index):
            return None if instance_index == 1 else f"bottle_{instance_index}"

    runtime = SceneSmithCompletionRuntime(
        scene=Scene(),
        asset_managers={"populate-surfaces": "manipuland-manager"},
        placement_adapters={"populate-surfaces": Adapter()},
        evidence_provider=lambda: {"measured": True},
        style_context="grungy cyberpunk bar",
    )
    operation = {"operation": "populate-surfaces"}
    brief = {"variant_id": "amber-bottle", "instance_count": 3}
    assert runtime.place_asset_brief(operation, brief, round_index=1) == (
        "bottle_0",
        "bottle_2",
    )
    assert len(calls) == 1
    assert calls[0][3]["style_context"] == "grungy cyberpunk bar"
