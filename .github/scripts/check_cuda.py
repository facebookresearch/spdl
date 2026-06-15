#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import spdl.io.utils


def check_cuda() -> None:
    print("Checking that SPDL is built with CUDA", flush=True)
    assert spdl.io.utils.built_with_cuda()


def check_nvdec() -> None:
    print("Checking that SPDL is built with NVDEC", flush=True)
    assert spdl.io.utils.built_with_nvcodec()


def main():
    logging.basicConfig(level=logging.DEBUG)

    check_cuda()
    if "nvdec" in sys.argv:
        check_nvdec()


if __name__ == "__main__":
    main()
