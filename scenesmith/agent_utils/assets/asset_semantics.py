"""Deterministic semantic guards for retrieved and cached assets.

Embedding similarity is useful for recall, but it is not a safe acceptance
criterion.  In particular, ranks normalized independently per catalog can make
unrelated objects tie.  These helpers enforce coarse ontology compatibility
before an asset is allowed into a scene.
"""

from __future__ import annotations

import re

from collections.abc import Iterable, Sequence

_FAMILY_TERMS: dict[str, frozenset[str]] = {
    "chair": frozenset(
        {
            "armchair",
            "chair",
            "chairs",
            "recliner",
            "seat",
        }
    ),
    "sofa": frozenset({"couch", "loveseat", "sofa"}),
    "bench": frozenset({"bench", "pew"}),
    "stool": frozenset({"stool"}),
    "table": frozenset(
        {
            "counter",
            "desk",
            "table",
            "tables",
            "workbench",
            "workstation",
        }
    ),
    "storage": frozenset(
        {
            "bookcase",
            "bookshelf",
            "cabinet",
            "credenza",
            "cupboard",
            "dresser",
            "locker",
            "shelf",
            "shelves",
            "shelving",
            "sideboard",
            "storage",
            "wardrobe",
        }
    ),
    "bed": frozenset({"bed", "cot", "gurney", "mattress"}),
    "lighting": frozenset(
        {
            "chandelier",
            "lamp",
            "light",
            "lighting",
            "luminaire",
            "pendant",
            "sconce",
        }
    ),
    "mirror": frozenset({"mirror", "mirrors"}),
    "artwork": frozenset(
        {"art", "artwork", "canvas", "painting", "picture", "poster", "print"}
    ),
    "sculpture": frozenset(
        {"bust", "busts", "figurine", "figurines", "sculpture", "statue", "statues"}
    ),
    "plant": frozenset({"flower", "flowers", "plant", "plants", "tree"}),
    "clock": frozenset({"clock", "timepiece"}),
    "rug": frozenset({"carpet", "mat", "rug", "runner"}),
    "display": frozenset(
        {"display", "monitor", "screen", "television", "terminal", "tv"}
    ),
    "board": frozenset({"blackboard", "board", "chalkboard", "easel", "whiteboard"}),
    "plumbing": frozenset(
        {"basin", "bathtub", "faucet", "shower", "sink", "toilet", "tub"}
    ),
    "appliance": frozenset(
        {
            "dishwasher",
            "freezer",
            "fridge",
            "microwave",
            "oven",
            "refrigerator",
            "stove",
            "washer",
        }
    ),
    "stairs": frozenset(
        {
            "elevator",
            "escalator",
            "ladder",
            "ramp",
            "stair",
            "staircase",
            "stairs",
            "steps",
        }
    ),
    "opening": frozenset({"door", "doorway", "window"}),
}

_STRUCTURAL_TERMS = frozenset(
    {
        "balcony",
        "balustrade",
        "bridge",
        "catwalk",
        "elevator",
        "escalator",
        "ladder",
        "landing",
        "mezzanine",
        "platform",
        "railing",
        "ramp",
        "stair",
        "staircase",
        "stairs",
        "steps",
    }
)


def semantic_tokens(value: str) -> set[str]:
    """Return lowercase word tokens, splitting common catalog separators."""

    return set(re.findall(r"[a-z]+", value.casefold().replace("_", " ")))


def semantic_families(value: str) -> frozenset[str]:
    """Map free-form catalog text to coarse, mutually meaningful families."""

    tokens = semantic_tokens(value)
    return frozenset(
        family for family, terms in _FAMILY_TERMS.items() if tokens & terms
    )


def is_structural_architecture_request(value: str) -> bool:
    """Whether a request belongs to the structural compiler, not furniture.

    The terms deliberately cover traversable architectural connectors and
    platforms.  Broad adjectives such as ``wall`` and ``floor`` are excluded so
    ordinary requests like "wall mirror" and "floor lamp" remain valid.
    """

    return bool(semantic_tokens(value) & _STRUCTURAL_TERMS)


def catalog_candidate_is_compatible(
    *,
    request_text: str,
    candidate_text: str,
    quality_score: float | None = None,
    minimum_quality: float = 0.0,
) -> tuple[bool, str]:
    """Apply a cheap hard gate before accepting a catalog candidate.

    Returns a boolean and a loggable reason.  Unknown object families remain
    eligible when they meet the quality floor; known but incompatible ontology
    branches are rejected.  A candidate containing extra major object families
    (for example, a table-and-chair set for a table request) is also rejected.
    """

    if quality_score is not None and quality_score < minimum_quality:
        return (
            False,
            f"quality {quality_score:.2f} is below {minimum_quality:.2f}",
        )

    request_tokens = semantic_tokens(request_text)
    candidate_tokens = semantic_tokens(candidate_text)
    if "stationary" in request_tokens:
        moving_mechanisms = candidate_tokens & {
            "rocker",
            "rocking",
            "swivel",
            "wheeled",
            "wheel",
            "wheels",
        }
        if moving_mechanisms:
            return (
                False,
                "stationary request is incompatible with moving mechanism: "
                + ", ".join(sorted(moving_mechanisms)),
            )

    requested = semantic_families(request_text)
    candidate_catalog_text = candidate_text.casefold()
    requests_shelving = bool(
        request_tokens & {"bookcase", "bookshelf", "shelf", "shelves", "shelving"}
    )
    intrinsic_cabinet_branch = bool(
        re.search(
            r"storage\s+furniture\s*/\s*(?:cabinets?|cupboards?)\b"
            r"|wordnet\s*/\s*(?:cabinet|cupboard)\.n\.",
            candidate_catalog_text,
        )
    )
    if requests_shelving and intrinsic_cabinet_branch:
        return (
            False,
            "bookcase/shelving request is incompatible with the intrinsic cabinet "
            "or cupboard catalog branch",
        )
    if "storage" in requested and re.search(
        r"\bbooks\s*(?:&|and)\s*documents\b", candidate_catalog_text
    ):
        return (
            False,
            "storage furniture request is incompatible with the Books & Documents "
            "catalog branch",
        )
    requests_human_sculpture = "sculpture" in requested and bool(
        request_tokens & {"female", "human", "male", "man", "person", "woman"}
    )
    candidate_is_animal_sculpture = bool(
        re.search(
            r"sculptures?\s*(?:&|and)\s*figurines?\s*/\s*animal figures?",
            candidate_catalog_text,
        )
        or candidate_tokens
        & {
            "bull",
            "cat",
            "elephant",
            "fish",
            "giraffe",
            "horse",
            "lion",
            "ray",
            "shark",
            "whale",
        }
    )
    if requests_human_sculpture and candidate_is_animal_sculpture:
        return (
            False,
            "human figure sculpture request is incompatible with an animal figure",
        )
    candidate = semantic_families(candidate_text)
    if not requested:
        return True, "request has no coarse ontology family"
    if not candidate:
        return False, "candidate metadata has no matching ontology family"
    if not requested.intersection(candidate):
        return (
            False,
            "ontology mismatch: requested "
            + ", ".join(sorted(requested))
            + "; candidate is "
            + ", ".join(sorted(candidate)),
        )
    unexpected = candidate - requested
    if "sculpture" in requested:
        # Catalog taxonomies commonly nest sculpture under "Decor & Art". The
        # broad parent label does not make a statue a composite artwork object.
        unexpected = unexpected - {"artwork"}
    if unexpected:
        return (
            False,
            "candidate is a composite or different object family: "
            + ", ".join(sorted(unexpected)),
        )
    return True, "compatible ontology family"


def tall_furniture_dimensions_are_compatible(
    *,
    request_text: str = "",
    desired_dimensions: Sequence[float] | None,
    bbox_min: Sequence[float] | None,
    bbox_max: Sequence[float] | None,
    minimum_target_height_m: float = 1.2,
    minimum_height_ratio: float = 0.6,
) -> tuple[bool, str]:
    """Reject tall furniture meshes that cannot approach their requested height."""

    if bbox_min is None or bbox_max is None:
        return True, "dimension metadata unavailable"
    if len(bbox_min) < 3 or len(bbox_max) < 3:
        return True, "dimension metadata incomplete"

    request_tokens = semantic_tokens(request_text)
    explicit_full_height = {"full", "height"} <= request_tokens
    produced_height = float(bbox_max[2]) - float(bbox_min[2])
    if desired_dimensions is None:
        if explicit_full_height and produced_height + 1e-9 < 1.6:
            return (
                False,
                f"produced height {produced_height:.3f}m is below the 1.600m "
                "minimum for explicit full-height furniture",
            )
        return True, "requested target dimensions unavailable"
    if len(desired_dimensions) < 3:
        return True, "dimension metadata incomplete"

    target_height = float(desired_dimensions[2])
    if target_height < minimum_target_height_m:
        return True, "requested furniture is not tall"

    required_ratio = 0.8 if explicit_full_height else minimum_height_ratio
    required_height = required_ratio * target_height
    if produced_height + 1e-9 < required_height:
        return (
            False,
            f"produced height {produced_height:.3f}m is below "
            f"{required_ratio:.0%} of requested height {target_height:.3f}m",
        )
    return True, "produced height is compatible with requested tall furniture"


def catalog_candidate_satisfies_request_details(
    *,
    request_text: str,
    candidate_text: str,
    supports_detail_fill: bool = False,
) -> tuple[bool, str]:
    """Enforce explicit semantic capabilities beyond coarse object family."""

    request_tokens = semantic_tokens(request_text)
    candidate_tokens = semantic_tokens(candidate_text)
    wants_visible_books = bool(
        request_tokens & {"book", "books", "volumes"}
        and request_tokens & {"dense", "densely", "filled", "populated", "visible"}
    )
    if wants_visible_books:
        has_visible_books = bool(
            candidate_tokens
            & {"book", "books", "encyclopedia", "encyclopedias", "volume", "volumes"}
        )
        is_fillable_shelving = bool(
            semantic_families(candidate_text) & {"storage"}
            and candidate_tokens
            & {"bookcase", "bookshelf", "shelf", "shelves", "shelving"}
        )
        if (
            not has_visible_books
            and not supports_detail_fill
            and not is_fillable_shelving
        ):
            return (
                False,
                "request requires visible books but candidate is neither populated, "
                "fillable shelving, nor exposes support zones for detail fill",
            )
    return True, "candidate satisfies explicit request details"


def candidate_metadata_text(
    *,
    name: str = "",
    description: str = "",
    aliases: Iterable[str] = (),
    tags: Iterable[str] = (),
    ontology_path: str = "",
) -> str:
    """Join all deterministic catalog semantics used by the compatibility gate."""

    return " ".join((name, description, *aliases, *tags, ontology_path))
