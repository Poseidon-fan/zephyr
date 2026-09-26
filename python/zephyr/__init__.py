"""Zephyr's native inference engine with synchronous and asynchronous text APIs."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import (
        AsyncEngine,
        Engine,
        EngineConfig,
        GenerationOutput,
        GenerationParams,
    )
    from .vector_adder import VectorAdder

__all__ = [
    "AsyncEngine",
    "Engine",
    "EngineConfig",
    "GenerationOutput",
    "GenerationParams",
    "VectorAdder",
]


def __getattr__(name: str) -> Any:
    # Keep CLI help and tokenizer-only use independent of CUDA extension loading.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".vector_adder" if name == "VectorAdder" else ".engine"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
