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

from . import _async, _concurrent, _convert, _preprocessing, _type_stub, _types


__all__ = sorted(
    _type_stub.__all__
    + _convert.__all__
    + _async.__all__
    + _concurrent.__all__
    + _preprocessing.__all__
    + _types.__all__
)

_doc_submodules = [
    "_async",
    "_concurrent",
    "_convert",
    "_preprocessing",
    "_types",
]


def __dir__():
    return __all__


def __getattr__(name: str):
    if name in _convert.__all__:
        return getattr(_convert, name)

    if name in _async.__all__:
        return getattr(_async, name)

    if name in _concurrent.__all__:
        return getattr(_concurrent, name)

    if name in _types.__all__:
        return getattr(_types, name)

    if name in _preprocessing.__all__:
        return getattr(_preprocessing, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
