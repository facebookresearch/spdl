"""Top-level module for SPDL."""

from typing import Any, List

from . import _async, _convert

__all__ = sorted(_convert.__all__ + _async.__all__)


def __dir__() -> List[str]:
    return __all__


def __getattr__(name: str) -> Any:
    if name in _convert.__all__:
        return getattr(_convert, name)

    if name in _async.__all__:
        return getattr(_async, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
