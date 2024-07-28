# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions."""

# pyre-unsafe

from . import _build, _ffmpeg, _tracing

_mods = [
    _build,
    _ffmpeg,
    _tracing,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str):
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
