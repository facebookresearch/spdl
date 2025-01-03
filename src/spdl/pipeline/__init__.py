# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements :py:class:`~spdl.pipeline.Pipeline`, a generic task execution engine."""

# pyre-unsafe

from typing import Any

from . import _builder, _hook, _pipeline, _utils

_mods = [
    _builder,
    _hook,
    _pipeline,
    _utils,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
