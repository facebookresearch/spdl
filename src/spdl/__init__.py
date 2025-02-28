# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level module for SPDL.

Do not add anything here, thanks.
"""


def __getattr__(name: str):
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("spdl")
        except PackageNotFoundError:
            return "unknown"

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# pyre-strict
