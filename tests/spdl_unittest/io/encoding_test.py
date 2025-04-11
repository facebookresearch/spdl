# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import sys
from itertools import product
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import spdl.io
import torch

from ..fixture import load_ref_image


@pytest.mark.parametrize(
    "pix_fmt,torch_tensor", product(["rgb24", "gray"], [False, True])
)
def test_encode_image_parity_simple(pix_fmt, torch_tensor):
    shape = (16, 16, 3) if pix_fmt == "rgb24" else (16, 16)
    ref = np.random.randint(256, size=shape, dtype=np.uint8)

    with NamedTemporaryFile(suffix=".png") as f:
        spdl.io.save_image(
            f.name,
            torch.from_numpy(ref) if torch_tensor else ref,
            pix_fmt=pix_fmt,
        )

        hyp = load_ref_image(f.name, shape, filter_desc=None)
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_encode_image_parity_png_gray16be():
    shape = (32, 64)

    ref = np.random.randint(256, size=shape, dtype=np.uint16)
    if sys.byteorder == "little":
        ref = ref.byteswap()

    with NamedTemporaryFile(suffix=".png") as f:
        spdl.io.save_image(
            f.name,
            ref,
            pix_fmt="gray16be",
        )

        hyp = load_ref_image(
            f.name,
            shape,
            dtype=np.uint16,
            filter_desc=None,
        )
    np.testing.assert_array_equal(hyp, ref, strict=True)


def _test_rejects(pix_fmt, dtype):
    data = np.ones((32, 64), dtype=dtype)
    with NamedTemporaryFile(suffix=".png") as f:
        with pytest.raises(ValueError):
            spdl.io.save_image(
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
        np.int32,
        np.int64,
        float,
        np.double,
    ],
)
def test_encode_rejects_dtypes_gray16be(dtype):
    _test_rejects("gray16be", dtype)
