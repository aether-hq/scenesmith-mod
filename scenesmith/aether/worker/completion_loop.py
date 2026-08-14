"""Bounded, immutable SceneSmith contextual-completion round ledger."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ..completion_bridge import execute_completion_patch
from ..scene_census import CensusError, canonical_digest


class CompletionLoopError(RuntimeError):
    """The real scene could not satisfy its approved completion contract."""


def derive_deficit_report(
    stage_input: dict[str, Any], census: dict[str, Any]
) -> dict[str, Any]:
    request = stage_input["request"]
    density = request["realization"]["density"]
    width, _, depth = request["shell"]["dimensions_m"]
    target = math.ceil(
        float(width) * float(depth) / 50 * float(density["target_objects_per_50_m2"])
    )
    observed = len(census["objects"])
    deficits: list[dict[str, Any]] = []
    if observed < target:
        deficits.append(
            {
                "deficit_id": "total-density",
                "kind": "total-density",
                "role_id": None,
                "functional_zone_ids": [],
                "observed_count": observed,
                "target_count": target,
                "shortfall_count": target - observed,
                "permitted_operations": [
                    "populate-surfaces",
                    "place-floor-group",
                    "place-wall-group",
                    "place-ceiling-group",
                ],
                "support_role_ids": [],
                "arrangement": "contextual",
                "rationale": (
                    "The measured scene is below the approved semantic density target; "
                    "additions must serve its zones and theme rather than act as count filler."
                ),
            }
        )
    for role in density.get("role_targets", ()):
        required_zones = set(role.get("functional_zone_ids", ()))
        role_observed = sum(
            1
            for item in census["objects"]
            if item["role_id"] == role["role_id"]
            and (
                not required_zones
                or required_zones.intersection(item["functional_zone_ids"])
            )
        )
        if role_observed > int(role["maximum_count"]):
            raise CompletionLoopError(
                f"role {role['role_id']} exceeds its approved maximum; "
                "add-only completion cannot repair overpopulation"
            )
        if role_observed >= int(role["target_count"]):
            continue
        arrangement = role["arrangement"]
        operation = {
            "surface-dressing": "populate-surfaces",
            "wall-array": "place-wall-group",
            "ceiling-array": "place-ceiling-group",
        }.get(arrangement, "place-floor-group")
        deficits.append(
            {
                "deficit_id": f"role-{role['role_id']}",
                "kind": "role-composition",
                "role_id": role["role_id"],
                "functional_zone_ids": list(role.get("functional_zone_ids", ())),
                "observed_count": role_observed,
                "target_count": int(role["target_count"]),
                "shortfall_count": int(role["target_count"]) - role_observed,
                "permitted_operations": [operation],
                "support_role_ids": list(role.get("support_role_ids", ())),
                "arrangement": arrangement,
                "rationale": role["rationale"],
            }
        )
    role_shortfall = sum(
        item["shortfall_count"]
        for item in deficits
        if item["kind"] == "role-composition"
    )
    return {
        "contract_version": 1,
        "job_id": stage_input["job_id"],
        "round_index": census["round_index"],
        "census_sha256": canonical_digest(census),
        "approved_target_count": target,
        "minimum_acceptable_count": math.ceil(
            target * float(density["minimum_completion_ratio"])
        ),
        "observed_count": observed,
        "maximum_addition_count": max(target - observed, role_shortfall, 0),
        "available_support_role_ids": sorted(
            {item["role_id"] for item in census["objects"]}
        ),
        "deficits": deficits,
        "target_change_allowed": False,
        "satisfied": not deficits,
    }


def build_authoring_brief(
    stage_input: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    routing = stage_input["request"]["realization"]["asset_routing"]
    sources = list(
        dict.fromkeys(
            (
                *routing["conventional_sources"],
                routing["articulated_source"],
                routing["distinctive_source"],
                routing["generated_fallback_source"],
            )
        )
    )
    return {
        "contract_version": 1,
        "job_id": stage_input["job_id"],
        "room_prompt": stage_input["room_prompt"],
        "report": report,
        "permitted_functional_zone_ids": [
            item["zone_id"] for item in stage_input["request"]["functional_zones"]
        ],
        "permitted_asset_sources": sources,
        "instruction": (
            "Return only a typed, add-only CompletionPatch. Select contextually meaningful "
            "roles, zones, supports, arrangements, and varied asset sources that resolve the "
            "listed deficits. Do not change density, architecture, portals, circulation, story "
            "positions, accepted placements, lighting, or theme. Do not add people. The "
            "deterministic placer owns transforms, collisions, support, and clearance."
        ),
    }


def validate_patch(
    stage_input: dict[str, Any], report: dict[str, Any], patch: dict[str, Any]
) -> None:
    problems: list[str] = []
    if patch.get("job_id") != stage_input["job_id"]:
        problems.append("patch identifies a different immutable job")
    if patch.get("round_index") != report["round_index"]:
        problems.append("patch round does not match the deficit report")
    if patch.get("base_census_sha256") != report["census_sha256"]:
        problems.append("patch was authored against a stale census")
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        problems.append("patch has no typed completion operations")
        operations = []
    if (
        sum(int(item.get("count", 0)) for item in operations)
        > report["maximum_addition_count"]
    ):
        problems.append("patch exceeds the bounded addition count")
    deficits = {item["deficit_id"]: item for item in report["deficits"]}
    zones = {item["zone_id"] for item in stage_input["request"]["functional_zones"]}
    resolved: Counter[str] = Counter()
    for operation in operations:
        if set(operation.get("functional_zone_ids", ())) - zones:
            problems.append(
                f"operation {operation.get('operation_id')} cites unknown zones"
            )
        for deficit_id in operation.get("resolves_deficit_ids", ()):
            deficit = deficits.get(deficit_id)
            if deficit is None:
                problems.append(
                    f"operation {operation.get('operation_id')} cites unknown deficit"
                )
                continue
            resolved[deficit_id] += int(operation.get("count", 0))
            if operation.get("operation") not in deficit["permitted_operations"]:
                problems.append(
                    f"operation {operation.get('operation_id')} is invalid for {deficit_id}"
                )
            if deficit["role_id"] and operation.get("role_id") != deficit["role_id"]:
                problems.append(
                    f"operation {operation.get('operation_id')} changes a required role"
                )
    if any(resolved[key] > value["shortfall_count"] for key, value in deficits.items()):
        problems.append("patch overfills a measured deficit")
    if problems:
        raise CompletionLoopError("invalid completion patch: " + "; ".join(problems))


class ArtifactLedger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, round_index: int, value: dict[str, Any]) -> str:
        digest = canonical_digest(value)
        path = self.root / f"{round_index:02d}-{kind}-{digest}.json"
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != payload:
            raise CompletionLoopError(
                f"immutable completion artifact changed: {path.name}"
            )
        if not path.exists():
            path.write_text(payload)
        return digest


def run_completion_loop(
    stage_input: dict[str, Any],
    initial_census: dict[str, Any],
    *,
    author_patch,
    runtime,
    artifact_root: Path,
    scene_root: Path | None = None,
) -> dict[str, Any]:
    ledger = ArtifactLedger(artifact_root)
    census = initial_census
    initial_digest = ledger.record("scene-census", census["round_index"], census)
    rounds: list[dict[str, Any]] = []
    maximum_rounds = int(
        stage_input["request"]["realization"]["inspection"]["maximum_repair_rounds"]
    )
    try:
        for _ in range(maximum_rounds + 1):
            report = derive_deficit_report(stage_input, census)
            report_digest = ledger.record(
                "deficit-report", census["round_index"], report
            )
            if report["satisfied"]:
                receipt = {
                    "contract_version": 1,
                    "job_id": stage_input["job_id"],
                    "initial_census_sha256": initial_digest,
                    "final_census_sha256": canonical_digest(census),
                    "final_object_count": len(census["objects"]),
                    "rounds": rounds,
                }
                ledger.record("completion-receipt", census["round_index"], receipt)
                return receipt
            if len(rounds) == maximum_rounds:
                break
            brief = build_authoring_brief(stage_input, report)
            ledger.record("authoring-brief", census["round_index"], brief)
            patch = author_patch(brief)
            patch_digest = ledger.record(
                "completion-patch", census["round_index"], patch
            )
            validate_patch(stage_input, report, patch)
            execution = execute_completion_patch(
                stage_input, patch, census, runtime, scene_root=scene_root
            )
            ledger.record("completion-execution", census["round_index"], execution)
            if execution["placed_count"] == 0:
                raise CompletionLoopError("completion repair made zero progress")
            next_census = execution["next_census"]
            round_receipt = {
                "round_index": census["round_index"],
                "deficit_report_sha256": report_digest,
                "patch_sha256": patch_digest,
                "input_census_sha256": canonical_digest(census),
                "output_census_sha256": canonical_digest(next_census),
                "requested_count": execution["requested_count"],
                "placed_count": execution["placed_count"],
                "rejected_count": execution["rejected_count"],
                "diagnostics": execution["diagnostics"],
            }
            rounds.append(round_receipt)
            ledger.record("completion-round", census["round_index"], round_receipt)
            census = next_census
            ledger.record("scene-census", census["round_index"], census)
    except (CensusError, CompletionLoopError):
        raise
    remaining = derive_deficit_report(stage_input, census)
    failure = {
        "contract_version": 1,
        "job_id": stage_input["job_id"],
        "code": "repair-rounds-exhausted",
        "final_census_sha256": canonical_digest(census),
        "completed_rounds": rounds,
        "remaining_deficits": {
            item["deficit_id"]: item["shortfall_count"]
            for item in remaining["deficits"]
        },
        "diagnostics": [f"bounded repair limit: {maximum_rounds}"],
    }
    ledger.record("completion-failure", census["round_index"], failure)
    raise CompletionLoopError(
        f"completion repair exhausted {maximum_rounds} rounds with deficits remaining"
    )
