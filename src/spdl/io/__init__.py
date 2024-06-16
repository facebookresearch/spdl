"""Implements the core I/O functionalities."""

# pyre-unsafe

# This has to happen before other sub modules are imporeted.
# Otherwise circular import would occur.
#
# I know, I should not use `*`. I don't want to either, but
# for creating annotation for types from C++ code, which might not be
# available at the runtime, while simultaneously pleasing all the linters
# (black, flake8 and pyre) and documentation tools, this seems like
# the simplest solution.
# This import is just for annotation, so please overlook this one.
from ._type_stub import *  # noqa

from . import (  # noqa: E402
    _composite,
    _config,
    _convert,
    _core,
    _preprocessing,
    _type_stub,
)


_mods = [
    _composite,
    _config,
    _convert,
    _core,
    _preprocessing,
    _type_stub,
]


__all__ = sorted(item for mod in _mods for item in mod.__all__)

_doc_submodules = [
    mod.__name__.split(".")[-1] for mod in _mods if mod not in [_type_stub]
]


def __dir__():
    return __all__


_deprecated = {
    "async_batch_load_image": (
        "async_load_image_batch",
        "`async_load_image_batch`",
    ),
    "async_batch_load_image_nvdec": (
        "async_load_image_batch_nvdec",
        "`async_load_image_batch_nvdec`",
    ),
}


def __getattr__(name: str):
    if name in _deprecated:
        import warnings

        new_name, replacements = _deprecated[name]
        warnings.warn(
            f"`{name}` has been deprecated. Please use {replacements}.",
            category=FutureWarning,
            stacklevel=2,
        )
        name = new_name

    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
