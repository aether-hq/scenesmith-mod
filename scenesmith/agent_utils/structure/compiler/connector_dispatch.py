"""Dispatch structural connector specifications to concrete compilers."""

from scenesmith.agent_utils.structure.compiler.connector_primitives import (
    compile_ladder,
    compile_spiral_stairs,
    compile_straight_ramp,
    compile_straight_stairs,
)
from scenesmith.agent_utils.structure.compiler.models import CompiledStructure
from scenesmith.agent_utils.structure.compiler.multisegment_connectors import (
    compile_multisegment_ramp,
    compile_multisegment_stairs,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorSpec,
    ConnectorType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    UnsupportedGeometryError,
)


def compile_connector(connector: ConnectorSpec) -> CompiledStructure:
    """Compile a supported connector or fail with an explicit diagnostic."""

    if connector.connector_type == ConnectorType.STAIRS_STRAIGHT:
        return compile_straight_stairs(connector)
    if connector.connector_type in {ConnectorType.STAIRS_L, ConnectorType.STAIRS_U}:
        return compile_multisegment_stairs(connector)
    if connector.connector_type == ConnectorType.STAIRS_SPIRAL:
        return compile_spiral_stairs(connector)
    if connector.connector_type == ConnectorType.LADDER:
        return compile_ladder(connector)
    if connector.connector_type == ConnectorType.RAMP:
        if connector.parameters.get("waypoints"):
            return compile_multisegment_ramp(connector)
        return compile_straight_ramp(connector)
    raise UnsupportedGeometryError(
        f"no compiler is implemented for connector type "
        f"'{connector.connector_type.value}'",
        entity_id=connector.connector_id,
    )
