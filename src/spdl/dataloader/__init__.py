"""Utilities to run I/O operations efficiently."""

# pyre-unsafe

from . import _bg_consumer, _bg_generator, _pipeline  # noqa: E402

_mods = [
    _bg_consumer,
    _bg_generator,
    _pipeline,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__():
    return __all__


def __getattr__(name: str):
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
