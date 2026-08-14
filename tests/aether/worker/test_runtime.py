from __future__ import annotations

from types import SimpleNamespace

from scenesmith.aether.worker.runtime import (
    OperationResources,
    SwitchingCompletionRuntime,
)


class _Scene:
    scene_dir = SimpleNamespace(name="room-one")

    def to_state_dict(self):
        return {"objects": {}}

    def restore_from_state_dict(self, _state):
        return None


class _Adapter:
    def __init__(self, kind: str):
        self.kind = kind

    def place(self, _scene, _asset, _operation, _brief, **_kwargs):
        return f"{self.kind}-instance"


def test_switching_runtime_closes_previous_category() -> None:
    opened: list[str] = []
    closed: list[str] = []

    def factory(kind: str) -> OperationResources:
        opened.append(kind)
        return OperationResources(
            SimpleNamespace(kind=kind),
            _Adapter(kind),
            lambda: closed.append(kind),
        )

    result = SimpleNamespace(
        has_failures=False,
        successful_assets=(SimpleNamespace(object_id="asset"),),
    )
    runtime = SwitchingCompletionRuntime(
        scene=_Scene(),
        operation_factory=factory,
        evidence_provider=lambda: {},
        style_context="test",
        asset_acquirer=lambda *_args, **_kwargs: result,
    )
    brief = {"variant_id": "one", "instance_count": 1}
    floor = {"operation": "place-floor-group"}
    wall = {"operation": "place-wall-group"}

    assert runtime.place_asset_brief(floor, brief, round_index=0) == (
        "place-floor-group-instance",
    )
    runtime.place_asset_brief(floor, brief, round_index=0)
    runtime.place_asset_brief(wall, brief, round_index=0)
    runtime.close()

    assert opened == ["place-floor-group", "place-wall-group"]
    assert closed == ["place-floor-group", "place-wall-group"]
