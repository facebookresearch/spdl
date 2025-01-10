# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import spdl.io
import torch
from spdl.io import encode_config

# Mostly smoke test
# TODO: Inspect the output


@pytest.mark.parametrize(
    "fmt,enc_cfg",
    product(
        [((16, 16, 3), "rgb24"), ((16, 16), "gray"), ((3, 16, 16), "yuv444p")],
        [None, encode_config(width=128, height=96, scale_algo="neighbor")],
    ),
)
def test_encode_smoketest(fmt, enc_cfg):
    shape, pix_fmt = fmt
    data = np.random.randint(255, size=shape, dtype=np.uint8)

    def _test(arr):
        with NamedTemporaryFile(suffix=".png") as f:
            spdl.io.encode_image(
                f.name,
                arr,
                pix_fmt=pix_fmt,
                encode_config=enc_cfg,
            )

    _test(data)
    _test(torch.from_numpy(data))


def test_encode_png_gray16be():
    data = np.random.randint(256, size=(32, 64), dtype=np.uint16)
    enc_cfg = spdl.io.encode_config(format="gray16be")

    def _test(arr):
        with NamedTemporaryFile(suffix=".png") as f:
            spdl.io.encode_image(
                f.name,
                arr,
                pix_fmt="gray16",
                encode_config=enc_cfg,
            )

    _test(data)


def _test_rejects(pix_fmt, dtype):
    data = np.ones((32, 64), dtype=dtype)
    with NamedTemporaryFile(suffix=".png") as f:
        with pytest.raises(RuntimeError):
            spdl.io.encode_image(
                f.name,
                data,
                pix_fmt=pix_fmt,
            )


@pytest.mark.parametrize(
    "pix_fmt,dtype",
    product(
        ["rgb24", "gray", "yuv444p"],
        [
            np.uint16,
            np.uint32,
            np.uint64,
            np.int16,
            np.int32,
            np.int64,
            float,
            np.double,
        ],
    ),
)
def test_encode_rejects_dtypes(pix_fmt, dtype):
    _test_rejects(pix_fmt, dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint32,
        np.uint64,
        np.int16,
        np.int32,
        np.int64,
        float,
        np.double,
    ],
)
def test_encode_rejects_dtypes_gray16be(dtype):
    _test_rejects("gray16be", dtype)
