# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from spdl.io.lib import _libspdl_cuda

__all__ = [
    "is_cuda_available",
    "is_nvcodec_available",
    "is_nvjpeg_available",
]


def is_cuda_available() -> bool:
    """Check if SPDL is compiled with CUDA support.

    Returns:
        True if SPDL is compiled with CUDA support.
    """
    try:
        return _libspdl_cuda.is_cuda_available()
    except Exception:
        return False


def is_nvcodec_available() -> bool:
    """Check if SPDL is compiled with NVCODEC support.

    Returns:
        True if SPDL is compiled with NVCODEC support.
    """
    try:
        return _libspdl_cuda.is_nvcodec_available()
    except Exception:
        return False


def is_nvjpeg_available() -> bool:
    """Check if SPDL is compiled with NVJPEG support.

    Returns:
        True if SPDL is compiled with NVJPEG support.
    """
    try:
        return _libspdl_cuda.is_nvjpeg_available()
    except Exception:
        return False
