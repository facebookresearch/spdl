# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO

import numpy as np
import pytest
import spdl.io


@pytest.mark.parametrize(
    "dtype",
    [
        (np.uint8),
        (np.uint16),
        (np.int16),
        (np.int32),
    ],
)
def test_load_npy_integral(dtype):
    """`load_npy` can reconstruct original array from bytes without copy."""
    rng = np.random.default_rng()
    shape = (2, 3, 4, 5)
    info = np.iinfo(dtype)
    ref = rng.integers(low=info.min, high=info.max, size=shape, dtype=dtype)
    ref[0, 0, 0, 0] = info.min
    ref[-1, -1, -1, -1] = info.max

    buffer = BytesIO()
    np.save(buffer, ref)
    buffer.seek(0)
    data = buffer.getvalue()
    recon = spdl.io.load_npy(data)
    assert np.array_equal(recon, ref)

    # Use bytearray to check if the change to the original is refrected to the recon
    # (which means that the recon is referring to the original, no copy)
    data = bytearray(data)
    print(f"{id(data)=}")
    print(f"{id(recon.data.obj)=}")
    recon = spdl.io.load_npy(data)
    assert np.array_equal(recon, ref)

    assert np.any(recon)
    # Fill zeros. The header is cleared too, but it's already parsed, so not an issue.
    data[:] = b"\x00" * len(data)
    assert not np.any(recon)


@pytest.mark.parametrize(
    "dtype",
    [
        (np.float32),
        (np.float64),
    ],
)
def test_load_npy_float(dtype):
    """`load_npy` can reconstruct original array from bytes without copy."""
    rng = np.random.default_rng()
    shape = (2, 3, 4, 5)
    info = np.finfo(dtype)
    ref = rng.random(size=shape, dtype=dtype)
    ref[0, 0, 0, 0] = info.min
    ref[-1, -1, -1, -1] = info.max

    buffer = BytesIO()
    np.save(buffer, ref)
    buffer.seek(0)
    data = buffer.getvalue()
    recon = spdl.io.load_npy(data)
    assert np.array_equal(recon, ref)

    # Use bytearray to check if the change to the original is refrected to the recon
    # (which means that the recon is referring to the original, no copy)
    data = bytearray(data)
    print(f"{id(data)=}")
    print(f"{id(recon.data.obj)=}")
    recon = spdl.io.load_npy(data)
    assert np.array_equal(recon, ref)

    assert np.any(recon)
    # Fill zeros. The header is cleared too, but it's already parsed, so not an issue.
    data[:] = b"\x00" * len(data)
    assert not np.any(recon)
