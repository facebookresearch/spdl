""""""

from typing import Any, List

from . import _ffmpeg, _folly, _tracing


__all__ = _tracing.__all__ + _ffmpeg.__all__ + _folly.__all__

_doc_submodules = [
    "_ffmpeg",
    "_folly",
    "_tracing",
]


def __dir__() -> List[str]:
    return __all__


def __getattr__(name: str) -> Any:
    if name in _tracing.__all__:
        return getattr(_tracing, name)

    if name in _ffmpeg.__all__:
        return getattr(_ffmpeg, name)

    if name in _folly.__all__:
        return getattr(_folly, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
