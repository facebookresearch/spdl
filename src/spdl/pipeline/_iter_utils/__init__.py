# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Iterator utilities for the pipeline module.

This package provides utilities for running iterables in subprocesses and
subinterpreters, as well as caching iterators for performance testing.
"""

from ._cache_iterator import cache_iterator
from ._subinterpreter import iterate_in_subinterpreter
from ._subprocess import iterate_in_subprocess

__all__ = [
    "iterate_in_subprocess",
    "iterate_in_subinterpreter",
    "cache_iterator",
]


def __getattr__(name: str) -> object:
    # Following imports are documentation purpose
    import os

    if os.environ.get("SPDL_DOC_SPHINX") == "1":
        if name in (
            "_execute_iterable",
            "_Cmd",
            "_Status",
            "_enter_iteration_mode",
            "_iterate_results",
        ):
            from . import _common

            return getattr(_common, name)

        if name == "_SubprocessIterable":
            from ._subprocess import _SubprocessIterable

            return _SubprocessIterable

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
