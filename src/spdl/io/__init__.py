# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements the core I/O functionalities."""

# pyre-unsafe

from typing import Any

# This has to happen before other sub modules are imporeted.
# Otherwise circular import would occur.
#
# I know, I should not use `*`. I don't want to either, but
# for creating annotation for types from C++ code, which might not be
# available at the runtime, while simultaneously pleasing all the linters
# (black, flake8 and pyre) and documentation tools, this seems like
# the simplest solution.
# This import is just for annotation, so please overlook this one.
from ._type_stub import *  # noqa: F403

from . import _composite, _config, _convert, _core, _preprocessing

_mods = [
    _composite,
    _config,
    _convert,
    _core,
    _preprocessing,
]


__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__():
    return __all__


_deprecated = {
    "streaming_demux_audio",
    "streaming_demux_video",
    "async_streaming_demux_audio",
    "async_streaming_demux_video",
    "streaming_load_audio",
    "streaming_load_video",
    "async_streaming_load_audio",
    "async_streaming_load_video",
}


def __getattr__(name: str) -> Any:
    if name in _deprecated:
        import warnings

        warnings.warn(
            f"`{name}` has been deprecated. Please use `spdl.io.Demuxer`.",
            category=FutureWarning,
            stacklevel=2,
        )
        if "demux" in name:
            return getattr(_core, name)
        return getattr(_composite, name)

    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
