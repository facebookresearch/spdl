"""Utility functions."""

from . import _async, _build, _ffmpeg, _flist, _folly, _tracing

_mods = [
    _async,
    _build,
    _flist,
    _ffmpeg,
    _folly,
    _tracing,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)

_doc_submodules = [mod.__name__.split(".")[-1] for mod in _mods]


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str):
    if name == "apply_async":
        import warnings
        message = (
            "`apply_async` has been deprecated. Use `async_generate` then `async_iterate`."
        )
        warnings.warn(message, FutureWarning, stacklevel=2)
    
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
