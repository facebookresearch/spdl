# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from spdl.io.lib import _libspdl

__all__ = [
    "is_cuda_available",
    "is_nvcodec_available",
]


def is_cuda_available() -> bool:
    """Check if SPDL is compiled with CUDA support.

    Returns:
        True if SPDL is compiled with CUDA support.
    """
    return _libspdl.is_cuda_available()


def is_nvcodec_available() -> bool:
    """Check if SPDL is compiled with NVCODEC support.

    Returns:
        True if SPDL is compiled with NVCODEC support.
    """
    return _libspdl.is_nvcodec_available()
