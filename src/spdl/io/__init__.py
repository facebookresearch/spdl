"""Implements the core I/O functionalities."""

from typing import Any, List

from . import _async, _concurrent, _convert, _types

__all__ = sorted(
    _convert.__all__ + _async.__all__ + _concurrent.__all__ + _types.__all__
)

_doc_submodules = [
    "_async",
    "_types",
    "_concurrent",
    "_convert",
]


def __dir__() -> List[str]:
    return __all__


def __getattr__(name: str) -> Any:
    if name in _convert.__all__:
        return getattr(_convert, name)

    if name in _async.__all__:
        return getattr(_async, name)

    if name in _concurrent.__all__:
        return getattr(_concurrent, name)

    if name in _types.__all__:
        return getattr(_types, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
