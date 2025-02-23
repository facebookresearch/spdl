# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore


def __getattr__(name: str):
    import warnings

    import spdl.io

    warnings.warn(
        "`spdl.utils` module has been moved to "
        "`spdl.io.utils`. Please update the ipmort statement.",
        stacklevel=2,
    )

    return getattr(spdl.io.utils, name)
