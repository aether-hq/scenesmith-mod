"""Grammar-only extraction of immutable literal prompt obligations."""

from __future__ import annotations

import hashlib
import re

from typing import Literal

from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    CandidateModality,
    ExplicitQuantity,
    LiteralObligationCandidate,
    PromptEvidence,
    RequirementRelationWire,
)

_NUMBER_VALUES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_QUALITATIVE_QUANTIFIERS = {
    "all",
    "many",
    "multiple",
    "several",
    "hundreds",
    "thousands",
    "a bunch of",
}
_NUMBER_PATTERN = "|".join(sorted(_NUMBER_VALUES, key=len, reverse=True))
_QUALITATIVE_PATTERN = "|".join(
    re.escape(value)
    for value in sorted(_QUALITATIVE_QUANTIFIERS, key=len, reverse=True)
)
_QUANTITY_PATTERN = re.compile(
    rf"\b(?:(?P<minimum>at\s+least)\s+)?"
    rf"(?P<token>\d+|{_NUMBER_PATTERN}|{_QUALITATIVE_PATTERN}|an?|both|a\s+couple\s+of)\b",
    flags=re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(
    r",+|\b(?:and|but|while|with|featuring|containing|having|where|without)\b",
    flags=re.IGNORECASE,
)
_FORBIDDEN_MODALITY = re.compile(
    r"\b(?:no|not|without|avoid|avoiding|exclude|excluding|forbid|forbidden)\b",
    flags=re.IGNORECASE,
)
_OPTIONAL_MODALITY = re.compile(
    r"\b(?:could|might|may|optional|optionally|perhaps)\b",
    flags=re.IGNORECASE,
)
_DISCOURSE_ONLY = re.compile(
    r"^(?:(?:and|then|also)\s+)?(?:so\s+on|etc|etcetera)\.?$",
    flags=re.IGNORECASE,
)

_RELATION_STOP_WORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with"}
)


def _relation_word(token: str) -> str:
    """Return a small grammar-only stem for source relationship validation."""

    word = token.casefold()
    if word in {
        "floor",
        "floors",
        "level",
        "levels",
        "storey",
        "storeys",
        "story",
        "stories",
    }:
        return "vertical-level"
    if word.endswith("ies") and len(word) > 4:
        return f"{word[:-3]}y"
    if word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if word.endswith(word[-1:] * 2):
            word = word[:-1]
        return word
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("es") and len(word) > 4:
        if word.endswith(("ches", "shes", "sses", "xes", "zes")):
            return word[:-2]
        return word[:-1]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _relation_words(text: str) -> frozenset[str]:
    return frozenset(
        _relation_word(token)
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if token.casefold() not in _RELATION_STOP_WORDS
    )


def _source_supports_relation(
    relation: RequirementRelationWire, source_text: str
) -> bool:
    """Reject model-invented relationships that have no literal source support."""

    source_words = _relation_words(source_text)
    predicate_words = _relation_words(relation.predicate)
    target_words = _relation_words(relation.target)
    return bool(predicate_words and predicate_words & source_words) and bool(
        target_words and target_words <= source_words
    )


def _stable_id(prefix: str, prompt: str, start: int, end: int, suffix: str = "") -> str:
    digest = hashlib.sha1(
        f"{prompt}:{start}:{end}:{suffix}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _trim_span(prompt: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and (prompt[start].isspace() or prompt[start] in ",;:"):
        start += 1
    while end > start and (prompt[end - 1].isspace() or prompt[end - 1] in ",;:"):
        end -= 1
    if start >= end:
        return None
    text = prompt[start:end]
    if not re.search(r"[A-Za-z0-9]", text) or _DISCOURSE_ONLY.fullmatch(text.strip()):
        return None
    return start, end


def _candidate_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Split assertive prose using grammar only, never domain vocabulary."""

    spans: list[tuple[int, int]] = []
    sentence_start = 0
    sentence_regions: list[tuple[int, int]] = []
    for index, character in enumerate(prompt):
        if character not in ".;!?":
            continue
        if character == ".":
            before = prompt[max(0, index - 12) : index]
            next_character = prompt[index + 1] if index + 1 < len(prompt) else ""
            # Initialisms/abbreviations and decimal-like tokens are not sentence
            # boundaries. This is lexical punctuation handling, not semantics.
            if (
                index > 0 and prompt[index - 1].isalnum() and next_character.isalnum()
            ) or re.search(r"(?:\b[A-Za-z]\.)+[A-Za-z]$", before):
                continue
        sentence_regions.append((sentence_start, index))
        sentence_start = index + 1
    sentence_regions.append((sentence_start, len(prompt)))

    for region_start, region_end in sentence_regions:
        cursor = region_start
        for boundary in _CLAUSE_BOUNDARY.finditer(prompt, region_start, region_end):
            trimmed = _trim_span(prompt, cursor, boundary.start())
            if trimmed:
                spans.append(trimmed)
            # Retain grammar markers so the model sees accompaniment/negation.
            cursor = (
                boundary.start() if boundary.group(0).strip() != "," else boundary.end()
            )
        trimmed = _trim_span(prompt, cursor, region_end)
        if trimmed:
            spans.append(trimmed)
    return tuple(dict.fromkeys(spans))


def _quantity_from_match(prompt: str, match: re.Match[str]) -> ExplicitQuantity:
    token = re.sub(r"\s+", " ", match.group("token").casefold())
    if token == "both":
        mode: Literal["exact", "minimum", "qualitative"] = "exact"
        value, label = 2, ""
    elif token == "a couple of":
        mode, value, label = "exact", 2, ""
    elif token in _QUALITATIVE_QUANTIFIERS:
        mode, value, label = "qualitative", None, token
    else:
        mode = "minimum" if match.group("minimum") else "exact"
        value = int(token) if token.isdigit() else _NUMBER_VALUES.get(token, 1)
        label = ""
    start, end = match.span()
    return ExplicitQuantity(
        quantity_id=_stable_id("qty", prompt, start, end),
        evidence=PromptEvidence(text=prompt[start:end], start=start, end=end),
        mode=mode,
        value=value,
        label=label,
    )


def literal_candidates_from_prompt(
    prompt: str,
) -> tuple[LiteralObligationCandidate, ...]:
    """Preserve literal clauses and quantities without semantic classification."""

    candidates: list[LiteralObligationCandidate] = []
    for start, end in _candidate_spans(prompt):
        text = prompt[start:end]
        if _FORBIDDEN_MODALITY.search(text):
            modality: CandidateModality = "forbidden"
        elif _OPTIONAL_MODALITY.search(text):
            modality = "optional"
        else:
            modality = "required"
        quantities = tuple(
            _quantity_from_match(prompt, match)
            for match in _QUANTITY_PATTERN.finditer(prompt, start, end)
        )
        candidates.append(
            LiteralObligationCandidate(
                candidate_id=_stable_id("candidate", prompt, start, end),
                evidence=PromptEvidence(text=text, start=start, end=end),
                modality=modality,
                explicit_quantities=quantities,
            )
        )
    return tuple(candidates)
