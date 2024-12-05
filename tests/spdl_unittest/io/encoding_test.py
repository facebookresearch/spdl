# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
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

    async def _test(arr):
        with NamedTemporaryFile(suffix=".png") as f:
            await spdl.io.async_encode_image(
                f.name,
                arr,
                pix_fmt=pix_fmt,
                encode_config=enc_cfg,
            )

    asyncio.run(_test(data))
    asyncio.run(_test(torch.from_numpy(data)))


def test_encode_png_gray16be():
    data = np.random.randint(256, size=(32, 64), dtype=np.uint16)
    enc_cfg = spdl.io.encode_config(format="gray16be")

    async def _test(arr):
        with NamedTemporaryFile(suffix=".png") as f:
            await spdl.io.async_encode_image(
                f.name,
                arr,
                pix_fmt="gray16",
                encode_config=enc_cfg,
            )

    asyncio.run(_test(data))


def _test_rejects(pix_fmt, dtype):
    async def _test(arr):
        with NamedTemporaryFile(suffix=".png") as f:
            with pytest.raises(RuntimeError):
                await spdl.io.async_encode_image(
                    f.name,
                    arr,
                    pix_fmt=pix_fmt,
                )

    data = np.ones((32, 64), dtype=dtype)
    asyncio.run(_test(data))


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
