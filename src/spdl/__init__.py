"""Top-level module for SPDL."""
from typing import Any, List

_convert_funcs = [
    "to_numpy",
    "to_torch",
    "to_numba",
]


__all__ = _convert_funcs


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    if name in _convert_funcs:
        from . import _convert

        return getattr(_convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
