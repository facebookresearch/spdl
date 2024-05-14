"""Implements the core I/O functionalities."""

# This has to happen before other sub modules are imporeted.
# Otherwise circular import would occur.
#
# I know, I should not use `*`. I don't want to either, but
# for creating annotation for types from C++ code, which might not be
# available at the runtime, while simultaneously pleasing all the linters
# (black, flake8 and pyre) and documentation tools, this seems like
# the simplest solution.
# This import is just for annotation, so pleaes overlook this one.
from ._type_stub import *  # noqa

from . import (
    _async,
    _concurrent,
    _config,
    _convert,
    _encoding,
    _misc,
    _preprocessing,
    _type_stub,
)

_mods = [
    _async,
    _concurrent,
    _config,
    _convert,
    _encoding,
    _preprocessing,
    _type_stub,
    _misc,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)

_doc_submodules = [
    mod.__name__.split(".")[-1] for mod in _mods if mod not in [_type_stub]
]


def __dir__():
    return __all__


def __getattr__(name: str):
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
