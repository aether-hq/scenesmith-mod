"""Structured tool inputs shared by manipuland tool mixins."""

from typing_extensions import TypedDict


class FillAssetItem(TypedDict):
    """Typed fill-asset input compatible with the agents SDK schema."""

    id: str
    x: float
    y: float
    rotation: float
