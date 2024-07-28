# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities to run I/O operations efficiently."""

# pyre-unsafe

from . import _builder, _flist, _hook, _pipeline, _utils  # noqa: E402

_mods = [
    _builder,
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
