"""Large corpus proving literal preservation is independent of domain nouns."""

import json
import re

from itertools import product
from pathlib import Path

import pytest

from scenesmith.agent_utils.scene_requirements import (
    CompositionPlan,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementQuantity,
    SceneCompositionOpinion,
    TopologyOpinion,
    VerificationPolicy,
    literal_candidates_from_prompt,
    merge_requirement_interpretations,
)


MATRIX_PATH = (
    Path(__file__).parents[1]
    / "test_data"
    / "semantic_obligations"
    / "generic_grammar_matrix.json"
)
SOURCE_PATH = (
    Path(__file__).parents[2] / "scenesmith" / "agent_utils" / "scene_requirements.py"
)
MATRIX = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
EXPANDED_CASES = list(product(MATRIX["domains"], MATRIX["templates"]))
DISCOURSE_WORDS = {"and", "so", "on", "etc", "etcetera"}


def _source_words(text):
    return [match for match in re.finditer(r"[A-Za-z0-9]+", text)]


def _assert_literal_coverage(prompt):
    candidates = literal_candidates_from_prompt(prompt)
    assert candidates
    covered = [False] * len(prompt)
    for candidate in candidates:
        evidence = candidate.evidence
        assert prompt[evidence.start : evidence.end] == evidence.text
        for index in range(evidence.start, evidence.end):
            covered[index] = True
        for quantity in candidate.explicit_quantities:
            q_evidence = quantity.evidence
            assert prompt[q_evidence.start : q_evidence.end] == q_evidence.text

    uncovered = [
        match.group(0)
        for match in _source_words(prompt)
        if match.group(0).casefold() not in DISCOURSE_WORDS
        and not all(covered[index] for index in range(match.start(), match.end()))
    ]
    assert uncovered == [], (prompt, uncovered, [c.evidence.text for c in candidates])
    return candidates


def _semantic_placeholder(candidate, quantity=None):
    if quantity is None and candidate.explicit_quantities:
        explicit = candidate.explicit_quantities[0]
        quantity = RequirementQuantity(
            mode=explicit.mode,
            value=explicit.value,
            label=explicit.label,
            source_quantity_id=explicit.quantity_id,
            interpreted_minimum=(5 if explicit.mode == "qualitative" else None),
        )
        source_quantity_id = explicit.quantity_id
    elif quantity is None:
        quantity = RequirementQuantity(mode="qualitative", label="present")
        source_quantity_id = None
    else:
        source_quantity_id = quantity.source_quantity_id
    return RequirementInterpretationProposal(
        candidate_id=candidate.candidate_id,
        subject=candidate.evidence.text,
        kind="object_group",
        source_quantity_id=source_quantity_id,
        quantity=quantity,
        topology=TopologyOpinion(
            role="LLM-classified role",
            enclosure="LLM-classified enclosure",
            circulation="Keep a usable route.",
            rationale="The production model owns semantic interpretation.",
        ),
        composition=CompositionPlan(
            recommended_strategy="procedural",
            reusable_parts=("generic reusable component",),
            procedural_geometry="Generate geometry from the interpreted envelope.",
            arrangement="Arrange parts to fulfill the interpreted whole.",
            rationale="This fixture tests the merge, not semantic quality.",
        ),
        verification=VerificationPolicy(
            stage="semantic",
            method="source_bound_fixture",
            measurable_criteria=("The interpreted obligation remains represented",),
        ),
        interpretation_rationale="Fixture interpretation for an arbitrary domain.",
    )


def _complete_fake_batch(candidates):
    proposals = []
    for candidate in candidates:
        if not candidate.explicit_quantities:
            proposals.append(_semantic_placeholder(candidate))
            continue
        for explicit in candidate.explicit_quantities:
            proposals.append(
                _semantic_placeholder(
                    candidate,
                    RequirementQuantity(
                        mode=explicit.mode,
                        value=explicit.value,
                        label=explicit.label,
                        source_quantity_id=explicit.quantity_id,
                        interpreted_minimum=(
                            5 if explicit.mode == "qualitative" else None
                        ),
                    ),
                )
            )
    return RequirementInterpretationBatch(
        composition=SceneCompositionOpinion(
            scene_type="fixture domain",
            overall_scale="model judged",
            preferred_dimensions_m=(12.0, 12.0, 6.0),
            composition_summary="Model-owned composition opinion.",
            topology_summary="Model-owned topology opinion.",
            circulation_summary="Model-owned circulation opinion.",
            density="model judged",
            focal_hierarchy=("model judged",),
        ),
        requirements=tuple(proposals),
        analysis_summary="Every arbitrary literal candidate was interpreted.",
    )


@pytest.mark.parametrize(
    ("domain", "template"),
    EXPANDED_CASES,
    ids=[
        f"{index:03d}-{template['template_id']}-{domain['entity'].replace(' ', '-')}"
        for index, (domain, template) in enumerate(EXPANDED_CASES)
    ],
)
def test_combinatorial_holdout_preserves_source_then_accepts_fake_llm(domain, template):
    prompt = template["prompt"].format(**domain)
    candidates = _assert_literal_coverage(prompt)
    graph = merge_requirement_interpretations(
        prompt,
        candidates,
        _complete_fake_batch(candidates),
        analysis_model="fixture-model",
        allow_partial=False,
    )

    assert graph.analysis_status == "complete"
    assert {candidate.candidate_id for candidate in candidates} == {
        requirement.source_candidate_id for requirement in graph.requirements
    }
    expected_quantity_ids = {
        quantity.quantity_id
        for candidate in candidates
        for quantity in candidate.explicit_quantities
    }
    observed_quantity_ids = {
        requirement.quantity.source_quantity_id
        for requirement in graph.requirements
        if requirement.quantity.source_quantity_id
    }
    assert observed_quantity_ids == expected_quantity_ids


def test_corpus_expands_to_at_least_250_cases_across_twenty_domains():
    assert len(MATRIX["domains"]) >= 20
    assert len(EXPANDED_CASES) >= 250
    assert len({domain["scene"] for domain in MATRIX["domains"]}) == len(
        MATRIX["domains"]
    )


def test_holdout_vocabulary_and_semantic_head_sets_do_not_leak_into_extractor():
    source = SOURCE_PATH.read_text(encoding="utf-8").casefold()
    leaked = [
        phrase
        for domain in MATRIX["domains"]
        for phrase in (domain["entity"], domain["destination"])
        if phrase.casefold() in source
    ]

    assert leaked == []
    assert "_zone_heads" not in source
    assert "_opening_heads" not in source
    assert "_connector_heads" not in source


@pytest.mark.parametrize(
    "case",
    MATRIX["curated"],
    ids=[case["case_id"] for case in MATRIX["curated"]],
)
def test_curated_edge_cases_preserve_text_quantities_and_modality(case):
    prompt = case["prompt"]
    candidates = _assert_literal_coverage(prompt)
    folded = prompt.casefold()

    if case["case_id"].startswith("negated"):
        assert any(candidate.modality == "forbidden" for candidate in candidates)
    if case["case_id"].startswith("optional"):
        assert any(candidate.modality == "optional" for candidate in candidates)
    if case["case_id"] == "contradictory-counts":
        values = sorted(
            quantity.value
            for candidate in candidates
            for quantity in candidate.explicit_quantities
            if quantity.mode == "exact" and quantity.value in {2, 3}
        )
        assert values == [2, 3]
    if case["case_id"] == "mixed-casing":
        assert any(
            quantity.value == 10
            for candidate in candidates
            for quantity in candidate.explicit_quantities
        )
    if "archted" in folded:
        assert any("archted" in candidate.evidence.text for candidate in candidates)

    graph = merge_requirement_interpretations(
        prompt,
        candidates,
        _complete_fake_batch(candidates),
        allow_partial=False,
    )
    assert graph.analysis_status == "complete"
