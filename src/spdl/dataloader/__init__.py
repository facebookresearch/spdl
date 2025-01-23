# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task specific data loading solutions based on :py:class:`~spdl.pipeline.Pipeline`."""

from typing import Any

from ._dataloader import DataLoader
from ._pytorch_dataloader import get_pytorch_dataloader, PyTorchDataLoader

__all__ = [
    "DataLoader",
    "get_pytorch_dataloader",
    "PyTorchDataLoader",
]

# pyre-strict


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> Any:  # pyre-ignore: [3]
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
