"""Imported mesh surfaces and their semantic annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    SurfaceRole,
    Transform3D,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    _finite,
    _require_id,
)


@dataclass(frozen=True)
class MeshSurfaceAnnotation:
    """Semantic roles authored for selected triangles of a structural mesh."""

    annotation_id: str
    triangle_indices: tuple[int, ...]
    roles: frozenset[SurfaceRole]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_id",
            _require_id(self.annotation_id, "annotation_id"),
        )
        indices = tuple(int(index) for index in self.triangle_indices)
        if not indices or any(index < 0 for index in indices):
            raise GeometryValidationError(
                "invalid_mesh_annotation",
                "triangle_indices must contain non-negative indices",
                entity_id=self.annotation_id,
            )
        if len(indices) != len(set(indices)):
            raise GeometryValidationError(
                "invalid_mesh_annotation",
                "triangle_indices must not contain duplicates",
                entity_id=self.annotation_id,
            )
        roles = frozenset(SurfaceRole(role) for role in self.roles)
        if not roles:
            raise GeometryValidationError(
                "missing_surface_role",
                "at least one semantic role is required",
                entity_id=self.annotation_id,
            )
        object.__setattr__(self, "triangle_indices", indices)
        object.__setattr__(self, "roles", roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.annotation_id,
            "triangle_indices": list(self.triangle_indices),
            "roles": sorted(role.value for role in self.roles),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshSurfaceAnnotation":
        return cls(
            annotation_id=data["id"],
            triangle_indices=tuple(data["triangle_indices"]),
            roles=frozenset(SurfaceRole(role) for role in data["roles"]),
        )


@dataclass(frozen=True)
class StructuralMeshSpec:
    """Imported/freeform structural mesh with explicit units and semantics."""

    mesh_id: str
    space_id: str
    mesh_path: str
    unit_scale: float
    transform: Transform3D = field(default_factory=Transform3D)
    annotations: tuple[MeshSurfaceAnnotation, ...] = ()
    require_watertight: bool = False
    normal_orientation: str = "unspecified"
    """Expected watertight winding: exterior, interior (cavern), or unspecified."""
    replaces_room_shell: bool = False
    """Use this mesh as the room shell instead of generating a flat box room."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_id", _require_id(self.mesh_id, "mesh_id"))
        object.__setattr__(self, "space_id", _require_id(self.space_id, "space_id"))
        path = str(self.mesh_path).strip()
        if not path:
            raise GeometryValidationError(
                "missing_mesh_path",
                "mesh_path must not be empty",
                entity_id=self.mesh_id,
            )
        scale = _finite(self.unit_scale, "unit_scale", entity_id=self.mesh_id)
        if scale <= 0:
            raise GeometryValidationError(
                "invalid_mesh_scale",
                f"unit_scale must be positive; got {scale}",
                entity_id=self.mesh_id,
            )
        annotations = tuple(self.annotations)
        normal_orientation = str(self.normal_orientation).strip().lower()
        if normal_orientation not in {"unspecified", "exterior", "interior"}:
            raise GeometryValidationError(
                "invalid_normal_orientation",
                "normal_orientation must be exterior, interior, or unspecified",
                entity_id=self.mesh_id,
            )
        annotation_ids = [annotation.annotation_id for annotation in annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise GeometryValidationError(
                "duplicate_mesh_annotation",
                "annotation IDs must be unique",
                entity_id=self.mesh_id,
            )
        object.__setattr__(self, "mesh_path", path)
        object.__setattr__(self, "unit_scale", scale)
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "normal_orientation", normal_orientation)
        object.__setattr__(self, "replaces_room_shell", bool(self.replaces_room_shell))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.mesh_id,
            "space_id": self.space_id,
            "mesh_path": self.mesh_path,
            "unit_scale": self.unit_scale,
            "transform": self.transform.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "require_watertight": self.require_watertight,
            "normal_orientation": self.normal_orientation,
            "replaces_room_shell": self.replaces_room_shell,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuralMeshSpec":
        if "unit_scale" not in data:
            raise GeometryValidationError(
                "missing_mesh_units",
                "freeform structural meshes require explicit unit_scale",
                entity_id=data.get("id"),
            )
        return cls(
            mesh_id=data["id"],
            space_id=data["space_id"],
            mesh_path=data["mesh_path"],
            unit_scale=data["unit_scale"],
            transform=Transform3D.from_dict(data.get("transform")),
            annotations=tuple(
                MeshSurfaceAnnotation.from_dict(annotation)
                for annotation in data.get("annotations", [])
            ),
            require_watertight=bool(data.get("require_watertight", False)),
            normal_orientation=data.get("normal_orientation", "unspecified"),
            replaces_room_shell=bool(data.get("replaces_room_shell", False)),
        )
