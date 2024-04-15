""""""

from typing import Any, List

from . import _ffmpeg, _folly, _futures, _tracing


__all__ = sorted(_ffmpeg.__all__ + _folly.__all__ + _futures.__all__ + _tracing.__all__)

_doc_submodules = [
    "_ffmpeg",
    "_folly",
    "_futures",
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

    if name in _futures.__all__:
        return getattr(_futures, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
