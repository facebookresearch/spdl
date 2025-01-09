# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task specific data loading solutions based on :py:class:`~spdl.pipeline.Pipeline`."""

# pyre-unsafe

from typing import Any

from . import _dataloader, _pytorch_dataloader

_mods = [
    _dataloader,
    _pytorch_dataloader,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    # For backward compatibility
    if name == "iterate_in_subprocess":
        import warnings

        warnings.warn(
            "`iterate_in_subprocess` has been moved to `spdl.source.utils`. "
            "Please update the import statement to "
            "`from spdl.source.utils import iterate_in_subprocess`.",
            stacklevel=2,
        )
        import spdl.source.utils

        return spdl.source.utils.iterate_in_subprocess

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
