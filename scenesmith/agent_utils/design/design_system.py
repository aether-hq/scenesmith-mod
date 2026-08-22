"""User-authored art direction compiled into stage-specific design contracts."""

from __future__ import annotations

import json
import os
import re
import tempfile

from pathlib import Path
from typing import Literal

import yaml

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DesignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LightingDirection(DesignModel):
    mood: str = "warm and dimensional"
    color_temperature_k: int = Field(default=3000, ge=1800, le=10000)
    contrast: Literal["soft", "balanced", "dramatic"] = "balanced"
    practical_density: Literal["sparse", "balanced", "layered"] = "balanced"


class SetDressingDirection(DesignModel):
    density: Literal["sparse", "balanced", "layered"] = "balanced"
    focal_hierarchy: tuple[str, ...] = ()
    repetition: str = "repeat key shapes and materials with measured variation"
    variation: str = "vary scale, silhouette, and finish without breaking the palette"
    motifs: tuple[str, ...] = ()
    forbidden_motifs: tuple[str, ...] = ()


class DesignSystem(DesignModel):
    """Reusable user-facing visual language for scene generation."""

    schema_version: Literal[1] = 1
    design_system_id: str
    name: str
    description: str = ""
    palette: tuple[str, ...] = ("#d8d1c4", "#2f3437", "#8a6a45")
    material_roles: dict[str, str] = Field(
        default_factory=lambda: {
            "floor": "natural wood",
            "walls": "warm mineral plaster",
            "primary_furniture": "wood and tactile upholstery",
            "accent": "aged metal",
        }
    )
    lighting: LightingDirection = Field(default_factory=LightingDirection)
    shape_vocabulary: tuple[str, ...] = ("clean planes", "softened corners")
    style_keywords: tuple[str, ...] = ("warm modern",)
    era: str = "contemporary"
    contrast: float = Field(default=0.55, ge=0.0, le=1.0)
    saturation: float = Field(default=0.45, ge=0.0, le=1.0)
    set_dressing: SetDressingDirection = Field(default_factory=SetDressingDirection)

    @field_validator("palette")
    @classmethod
    def palette_has_usable_values(cls, palette: tuple[str, ...]) -> tuple[str, ...]:
        if len(palette) < 2 or len(palette) > 12:
            raise ValueError("palette must contain 2-12 colors")
        for color in palette:
            if not color.strip() or len(color) > 64:
                raise ValueError("palette colors must be short non-empty values")
        return palette


class StyleBible(DesignModel):
    """Compiled immutable instructions consumed by construction stages."""

    schema_version: Literal[1] = 1
    design_system_id: str
    name: str
    palette_roles: dict[str, str]
    material_roles: dict[str, str]
    asset_search_tags: tuple[str, ...]
    floor_plan_direction: str
    furniture_direction: str
    wall_direction: str
    ceiling_direction: str
    detail_direction: str
    forbidden_motifs: tuple[str, ...]

    def to_prompt_brief(self) -> str:
        return "StyleBible v1 (treat as a scene-wide visual contract):\n" + json.dumps(
            self.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
        )


BUILTIN_DESIGN_SYSTEMS: dict[str, DesignSystem] = {
    "warm-modern": DesignSystem(
        design_system_id="warm-modern",
        name="Warm modern",
        description="Quiet contemporary interiors with tactile natural materials.",
        palette=("#E7DED0", "#B98A5A", "#66554A", "#263238", "#B5B88A"),
        material_roles={
            "floor": "matte natural oak or honed warm stone",
            "walls": "warm limewash plaster",
            "primary_furniture": "walnut, oak, and oatmeal textile",
            "accent": "aged brass and muted olive",
        },
        lighting=LightingDirection(
            mood="warm pools of practical light with soft daylight",
            color_temperature_k=2900,
            contrast="balanced",
            practical_density="layered",
        ),
        shape_vocabulary=("calm horizontal lines", "soft radii", "slender dark frames"),
        style_keywords=("warm modern", "tactile", "restrained", "human scale"),
        era="contemporary",
        contrast=0.48,
        saturation=0.35,
        set_dressing=SetDressingDirection(
            density="balanced",
            focal_hierarchy=(
                "primary activity zone",
                "feature light",
                "crafted object",
            ),
            motifs=("natural grain", "handmade ceramics", "textile layers"),
            forbidden_motifs=("glossy all-white showroom", "neon cyberpunk clutter"),
        ),
    ),
    "jewel-maximalist": DesignSystem(
        design_system_id="jewel-maximalist",
        name="Jewel maximalist",
        description="Vibrant layered sets with theatrical color and collected detail.",
        palette=("#143D3B", "#7D1F3A", "#C89B3C", "#4B2D73", "#E9D8B4"),
        material_roles={
            "floor": "dark parquet or patterned stone",
            "walls": "deep pigmented plaster or patterned textile",
            "primary_furniture": "velvet, carved dark wood, and lacquer",
            "accent": "polished brass, colored glass, and fringe",
        },
        lighting=LightingDirection(
            mood="theatrical layered practicals with luminous focal pools",
            color_temperature_k=2700,
            contrast="dramatic",
            practical_density="layered",
        ),
        shape_vocabulary=("bold curves", "ornamental silhouettes", "nested patterns"),
        style_keywords=("maximalist", "jewel toned", "theatrical", "collected"),
        era="eclectic historical contemporary",
        contrast=0.82,
        saturation=0.78,
        set_dressing=SetDressingDirection(
            density="layered",
            focal_hierarchy=("hero furniture", "statement art", "decorative lighting"),
            repetition="repeat jewel colors at three scales across the room",
            variation="mix pattern scale while repeating metal and wood finishes",
            motifs=(
                "arched forms",
                "botanical pattern",
                "colored glass",
                "books and art",
            ),
            forbidden_motifs=("empty white walls", "identical furniture grid"),
        ),
    ),
    "calm-natural": DesignSystem(
        design_system_id="calm-natural",
        name="Calm natural",
        description="Low-saturation biophilic spaces with generous breathing room.",
        palette=("#E8E3D8", "#A9A58F", "#6F8062", "#675B4D", "#C9B99B"),
        material_roles={
            "floor": "light timber, cork, or matte limestone",
            "walls": "soft clay plaster",
            "primary_furniture": "ash wood, linen, wool, and woven fiber",
            "accent": "patinated ceramic and living greenery",
        },
        lighting=LightingDirection(
            mood="diffuse daylight with gentle warm evening practicals",
            color_temperature_k=3200,
            contrast="soft",
            practical_density="balanced",
        ),
        shape_vocabulary=("organic curves", "low profiles", "simple crafted joints"),
        style_keywords=("biophilic", "quiet", "natural", "soft minimal"),
        era="timeless contemporary",
        contrast=0.28,
        saturation=0.28,
        set_dressing=SetDressingDirection(
            density="sparse",
            focal_hierarchy=(
                "view or plant",
                "primary activity zone",
                "textural object",
            ),
            motifs=("woven texture", "stone", "branches", "hand-thrown ceramics"),
            forbidden_motifs=(
                "plastic gloss",
                "harsh primary colors",
                "visual clutter",
            ),
        ),
    ),
}


def compile_style_bible(system: DesignSystem) -> StyleBible:
    """Compile editable tokens into exact stage-specific instructions."""

    palette = system.palette
    palette_roles = {
        "base": palette[0],
        "primary": palette[1 % len(palette)],
        "secondary": palette[2 % len(palette)],
        "accent": palette[3 % len(palette)],
        "highlight": palette[4 % len(palette)],
    }
    vocabulary = ", ".join(system.shape_vocabulary)
    styles = ", ".join(system.style_keywords)
    dressing = system.set_dressing
    motifs = ", ".join(dressing.motifs) or "no required motif"
    common = (
        f"Use {styles} art direction from the {system.era} era; shape vocabulary: "
        f"{vocabulary}; contrast {system.contrast:.2f}; saturation "
        f"{system.saturation:.2f}."
    )
    return StyleBible(
        design_system_id=system.design_system_id,
        name=system.name,
        palette_roles=palette_roles,
        material_roles=dict(system.material_roles),
        asset_search_tags=tuple(
            dict.fromkeys(
                (*system.style_keywords, *system.shape_vocabulary, system.era)
            )
        ),
        floor_plan_direction=(
            f"{common} Establish a clear focal hierarchy: "
            f"{', '.join(dressing.focal_hierarchy) or 'primary activity zone'}. "
            f"Compose at {dressing.density} density with intentional negative space."
        ),
        furniture_direction=(
            f"{common} Select coherent silhouettes and materials. {dressing.repetition}; "
            f"{dressing.variation}. Preserve functional groupings before decoration."
        ),
        wall_direction=(
            f"Use wall treatment '{system.material_roles.get('walls', 'coherent finish')}'. "
            f"Dress walls with motifs: {motifs}; support the focal hierarchy rather "
            "than filling every surface."
        ),
        ceiling_direction=(
            f"Lighting mood: {system.lighting.mood}; approximately "
            f"{system.lighting.color_temperature_k}K, {system.lighting.contrast} "
            f"contrast, {system.lighting.practical_density} practical density."
        ),
        detail_direction=(
            f"Set-dress at {dressing.density} density. Motifs: {motifs}. Repeat and "
            "vary props across foreground, midground, and focal surfaces."
        ),
        forbidden_motifs=dressing.forbidden_motifs,
    )


def load_design_system(path: Path) -> DesignSystem:
    """Load JSON or YAML using the same strict schema."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("design system root must be an object")
    return DesignSystem.model_validate(payload)


def load_design_system_from_env() -> DesignSystem | None:
    configured = os.environ.get("SCENESMITH_DESIGN_SYSTEM_PATH", "").strip()
    if not configured:
        return None
    return load_design_system(Path(configured).expanduser().resolve())


def apply_style_bible(prompt: str, bible: StyleBible) -> str:
    marker = f"StyleBible v1 ({bible.design_system_id})"
    if marker in prompt:
        return prompt
    return f"{prompt}\n\n{marker}\n{bible.to_prompt_brief()}"


def persist_design_contract(
    system: DesignSystem, bible: StyleBible, output_dir: Path
) -> None:
    """Atomically persist both editable tokens and compiled instructions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("design_system.json", system.model_dump_json(indent=2) + "\n"),
        ("style_bible.json", bible.model_dump_json(indent=2) + "\n"),
    ):
        destination = output_dir / name
        fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def design_system_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
