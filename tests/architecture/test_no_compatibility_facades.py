"""Keep the v3 package graph free of behaviorless compatibility modules."""

import ast

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPATIBILITY_FACADES = {
    "scenesmith.agent_utils.blender.overlays.annotations": (
        "scenesmith/agent_utils/blender/overlays/annotations.py"
    ),
    "scenesmith.agent_utils.geometry.support_surface_extraction": (
        "scenesmith/agent_utils/geometry/support_surface_extraction.py"
    ),
    "scenesmith.agent_utils.rendering.rendering": (
        "scenesmith/agent_utils/rendering/rendering.py"
    ),
    "scenesmith.agent_utils.semantics.environment.semantic_environments": (
        "scenesmith/agent_utils/semantics/environment/semantic_environments.py"
    ),
    "scenesmith.agent_utils.structure.structural_compiler": (
        "scenesmith/agent_utils/structure/structural_compiler.py"
    ),
    "scenesmith.agent_utils.structure.structural_geometry": (
        "scenesmith/agent_utils/structure/structural_geometry.py"
    ),
    "scenesmith.floor_plan_agents.tools.submission.room_placement": (
        "scenesmith/floor_plan_agents/tools/submission/room_placement.py"
    ),
    "scenesmith.manipuland_agents.tools.fill_container": (
        "scenesmith/manipuland_agents/tools/fill_container.py"
    ),
}


def _maintained_python_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for root_name in ("scenesmith", "tests", "scripts", "examples"):
        files.extend((REPOSITORY_ROOT / root_name).rglob("*.py"))
    return tuple(
        path
        for path in files
        if path != Path(__file__).resolve() and "__pycache__" not in path.parts
    )


def test_removed_compatibility_facades_do_not_return() -> None:
    for relative_path in COMPATIBILITY_FACADES.values():
        assert not (REPOSITORY_ROOT / relative_path).exists()

    module_names = frozenset(COMPATIBILITY_FACADES)
    stale_references: list[str] = []
    for source_path in _maintained_python_files():
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in module_names:
                stale_references.append(
                    f"{source_path.relative_to(REPOSITORY_ROOT)}:{node.lineno}"
                )
            elif isinstance(node, ast.Import) and any(
                alias.name in module_names for alias in node.names
            ):
                stale_references.append(
                    f"{source_path.relative_to(REPOSITORY_ROOT)}:{node.lineno}"
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(
                    node.value == module_name
                    or node.value.startswith(f"{module_name}.")
                    for module_name in module_names
                ):
                    stale_references.append(
                        f"{source_path.relative_to(REPOSITORY_ROOT)}:{node.lineno}"
                    )

    assert not stale_references, "stale compatibility references: " + ", ".join(
        sorted(set(stale_references))
    )
