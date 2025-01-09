# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task specific data loading solutions based on :py:class:`~spdl.pipeline.Pipeline`."""

# pyre-unsafe

import warnings
from typing import Any

from . import _dataloader, _iterators, _pytorch_dataloader
from ._source import _type

_mods = [
    _dataloader,
    _iterators,
    _pytorch_dataloader,
    _type,
]

__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    # For backward compatibility
    import spdl.pipeline

    if name in spdl.pipeline.__all__:
        warnings.warn(
            f"{name} has been moved to {spdl.pipeline.__name__}. "
            "Please update the import statement to "
            f"`from {spdl.pipeline.__name__} import {name}`.",
            stacklevel=2,
        )
        return getattr(spdl.pipeline, name)

    if name == "run_in_subprocess":
        warnings.warn(
            "`run_in_subprocess` has been deprecated. "
            "Use `iterate_in_subprocess` instead.",
            stacklevel=2,
        )
        return _iterators.run_in_subprocess

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
