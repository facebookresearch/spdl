"""Utilities to run I/O operations efficiently."""

# pyre-unsafe

from . import (  # noqa: E402
    _bg_consumer,
    _bg_generator,
    _flist,
    _hook,
    _pipeline,
    _utils,
)

_mods = [
    _bg_consumer,
    _bg_generator,
    _hook,
    _pipeline,
    _utils,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__():
    return __all__


def __getattr__(name: str):
    for mod in _mods + [_flist]:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
