"""Utility functions."""

from typing import Any, List

from . import _build, _ffmpeg, _folly, _futures, _tracing

_mods = [
    _build,
    _ffmpeg,
    _folly,
    _futures,
    _tracing,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)

_doc_submodules = [mod.__name__.split(".")[-1] for mod in _mods]


def __dir__() -> List[str]:
    return __all__


def __getattr__(name: str) -> Any:
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
