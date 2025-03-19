# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from spdl.io.lib import _libspdl

__all__ = [
    "built_with_cuda",
    "built_with_nvcodec",
    "built_with_nvjpeg",
]


def built_with_cuda() -> bool:
    """Check if SPDL is compiled with CUDA support.

    Returns:
        True if SPDL is compiled with CUDA support.
    """
    return _libspdl.built_with_cuda()


def built_with_nvcodec() -> bool:
    """Check if SPDL is compiled with NVCODEC support.

    Returns:
        True if SPDL is compiled with NVCODEC support.
    """
    return _libspdl.built_with_nvcodec()


def built_with_nvjpeg() -> bool:
    """Check if SPDL is compiled with NVJPEG support.

    Returns:
        True if SPDL is compiled with NVJPEG support.
    """
    return _libspdl.built_with_nvjpeg()
