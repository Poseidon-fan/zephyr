"""
Zephyr native bindings
"""
from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['VectorAdder']
class VectorAdder:
    def __init__(self) -> None:
        ...
    def add(self, left: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], right: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> list[float]:
        ...
