"""Physics validation for scenes using Drake."""

import logging

from dataclasses import dataclass

console_logger = logging.getLogger(__name__)


@dataclass
class CollisionPair:
    """Represents a collision between two objects."""

    object_a_name: str
    """Name of the first object in collision."""

    object_a_id: str
    """UniqueID as string of the first object."""

    object_b_name: str
    """Name of the second object in collision."""

    object_b_id: str
    """UniqueID as string of the second object."""

    penetration_depth: float
    """Penetration depth in meters (positive value indicates penetration)."""

    def to_description(self) -> str:
        """Format for agent consumption with directly usable object IDs."""
        penetration_cm = self.penetration_depth * 100
        # Only show penetration depth if meaningful (>0.1mm).
        if penetration_cm > 0.01:
            return (
                f"{self.object_a_id} collides with {self.object_b_id} "
                f"({penetration_cm:.1f}cm penetration)"
            )
        else:
            return f"{self.object_a_id} collides with {self.object_b_id} (touching)"
