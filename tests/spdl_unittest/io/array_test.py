# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import io
import unittest
from io import BytesIO

import numpy as np
import spdl.io
from parameterized import parameterized
from spdl.io.lib._archive import parse_zip  # pyre-ignore


def _dump_npy(arr: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return buffer.getvalue()


class TestLoadNpy(unittest.TestCase):
    @parameterized.expand(
        [
            (np.uint8,),
            (np.uint16,),
            (np.int16,),
            (np.int32,),
        ]
    )
    def test_load_npy_integral(
        self, dtype: type[np.unsignedinteger | np.signedinteger]
    ) -> None:
        """`load_npy` can reconstruct original array from bytes without copy."""
        rng = np.random.default_rng()
        shape = (2, 3, 4, 5)
        info = np.iinfo(dtype)
        ref = rng.integers(low=info.min, high=info.max, size=shape, dtype=dtype)
        ref[0, 0, 0, 0] = info.min
        ref[-1, -1, -1, -1] = info.max

        data = _dump_npy(ref)
        recon = spdl.io.load_npy(data)
        self.assertTrue(np.array_equal(recon, ref))

        # Use bytearray to check if the change to the original is refrected to the recon
        # (which means that the recon is referring to the original, no copy)
        data = bytearray(data)
        print(f"{id(data)=}")
        print(f"{id(recon.data.obj)=}")
        recon = spdl.io.load_npy(data)
        self.assertTrue(np.array_equal(recon, ref))

        self.assertTrue(np.any(recon))
        # Fill zeros. The header is cleared too, but it's already parsed, so not an issue.
        data[:] = b"\x00" * len(data)
        self.assertFalse(np.any(recon))

    @parameterized.expand(
        [
            (np.float32,),
            (np.float64,),
        ]
    )
    def test_load_npy_float(self, dtype: type[np.floating]) -> None:
        """`load_npy` can reconstruct original array from bytes without copy."""
        rng = np.random.default_rng()
        shape = (2, 3, 4, 5)
        info = np.finfo(dtype)
        ref = rng.random(size=shape, dtype=dtype)
        ref[0, 0, 0, 0] = info.min
        ref[-1, -1, -1, -1] = info.max

        data = _dump_npy(ref)
        recon = spdl.io.load_npy(data)
        self.assertTrue(np.array_equal(recon, ref))

        # Use bytearray to check if the change to the original is refrected to the recon
        # (which means that the recon is referring to the original, no copy)
        data = bytearray(data)
        print(f"{id(data)=}")
        print(f"{id(recon.data.obj)=}")
        recon = spdl.io.load_npy(data)
        self.assertTrue(np.array_equal(recon, ref))

        self.assertTrue(np.any(recon))
        # Fill zeros. The header is cleared too, but it's already parsed, so not an issue.
        data[:] = b"\x00" * len(data)
        self.assertFalse(np.any(recon))


##############################################################################
# NPZ
##############################################################################


class TestParseZip(unittest.TestCase):
    def test_parse_zip_too_short(self) -> None:
        for i in range(21):
            with self.assertRaisesRegex(
                RuntimeError, "The data is not a valid zip file."
            ):
                parse_zip(b"o" * i)

    def test_parse_zip_no_eocdr_sig(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "Failed to locate the end of the central directory."
        ):
            parse_zip(b"foooooooooooooooooooooooooo")


def _get_test_float_arr(dtype: type[np.floating]) -> np.ndarray:
    finfo = np.finfo(dtype)
    return np.array([finfo.min, finfo.max, 0], dtype=dtype)


def _get_test_int_arr(dtype: type[np.signedinteger | np.unsignedinteger]) -> np.ndarray:
    iinfo = np.iinfo(dtype)
    return np.array([iinfo.min, iinfo.max, 0], dtype=dtype)


def _dump_npz(*arrays: np.ndarray, **kwarrays: np.ndarray) -> bytes:
    with io.BytesIO() as buf:
        np.savez(buf, *arrays, allow_pickle=False, **kwarrays)
        buf.seek(0)
        return buf.read()


def _dump_npz_compressed(*arrays: np.ndarray, **kwarrays: np.ndarray) -> bytes:
    with io.BytesIO() as buf:
        np.savez_compressed(buf, *arrays, allow_pickle=False, **kwarrays)
        buf.seek(0)
        return buf.read()


class TestLoadNpz(unittest.TestCase):
    def test_load_npz(self) -> None:
        """spdl.io.load_npz() should load a .npz file."""
        x = np.arange(10)
        y = np.sin(x)

        zeros = np.zeros((0, 0))
        ones = np.ones((3, 4, 5))
        bool_array = np.array([False, True], dtype=bool)
        float16_array = _get_test_float_arr(np.float16)
        float32_array = _get_test_float_arr(np.float32)
        float64_array = _get_test_float_arr(np.float64)
        uint8_array = _get_test_int_arr(np.uint8)
        int16_array = _get_test_int_arr(np.int16)
        uint16_array = _get_test_int_arr(np.uint16)
        int32_array = _get_test_int_arr(np.int32)
        uint32_array = _get_test_int_arr(np.uint32)
        int64_array = _get_test_int_arr(np.int64)
        uint64_array = _get_test_int_arr(np.uint64)

        dumped = _dump_npz(
            x,
            y,
            zeros=zeros,
            ones=ones,
            bool_array=bool_array,
            float16_array=float16_array,
            float32_array=float32_array,
            float64_array=float64_array,
            uint8_array=uint8_array,
            int16_array=int16_array,
            uint16_array=uint16_array,
            int32_array=int32_array,
            uint32_array=uint32_array,
            int64_array=int64_array,
            uint64_array=uint64_array,
        )
        data = spdl.io.load_npz(dumped)

        np.testing.assert_array_equal(data["arr_0"], x)
        np.testing.assert_array_equal(data["arr_1"], y)
        np.testing.assert_array_equal(data["zeros"], zeros)
        np.testing.assert_array_equal(data["ones"], ones)
        np.testing.assert_array_equal(data["bool_array"], bool_array)
        np.testing.assert_array_equal(data["float16_array"], float16_array)
        np.testing.assert_array_equal(data["float32_array"], float32_array)
        np.testing.assert_array_equal(data["float64_array"], float64_array)
        np.testing.assert_array_equal(data["uint8_array"], uint8_array)
        np.testing.assert_array_equal(data["int16_array"], int16_array)
        np.testing.assert_array_equal(data["uint16_array"], uint16_array)
        np.testing.assert_array_equal(data["int32_array"], int32_array)
        np.testing.assert_array_equal(data["uint32_array"], uint32_array)
        np.testing.assert_array_equal(data["int64_array"], int64_array)
        np.testing.assert_array_equal(data["uint64_array"], uint64_array)

    def test_load_npz_compressed(self) -> None:
        """Can load files compressed with DEFLATED method"""
        x = np.arange(10)
        y = np.sin(x)

        zeros = np.zeros((0, 0))
        ones = np.ones((3, 4, 5))
        bool_array = np.array([False, True], dtype=bool)
        float16_array = _get_test_float_arr(np.float16)
        float32_array = _get_test_float_arr(np.float32)
        float64_array = _get_test_float_arr(np.float64)
        uint8_array = _get_test_int_arr(np.uint8)
        int16_array = _get_test_int_arr(np.int16)
        uint16_array = _get_test_int_arr(np.uint16)
        int32_array = _get_test_int_arr(np.int32)
        uint32_array = _get_test_int_arr(np.uint32)
        int64_array = _get_test_int_arr(np.int64)
        uint64_array = _get_test_int_arr(np.uint64)

        dumped = _dump_npz_compressed(
            x,
            y,
            zeros=zeros,
            ones=ones,
            bool_array=bool_array,
            float16_array=float16_array,
            float32_array=float32_array,
            float64_array=float64_array,
            uint8_array=uint8_array,
            int16_array=int16_array,
            uint16_array=uint16_array,
            int32_array=int32_array,
            uint32_array=uint32_array,
            int64_array=int64_array,
            uint64_array=uint64_array,
        )
        data = spdl.io.load_npz(dumped)

        np.testing.assert_array_equal(data["arr_0"], x)
        np.testing.assert_array_equal(data["arr_1"], y)
        np.testing.assert_array_equal(data["zeros"], zeros)
        np.testing.assert_array_equal(data["ones"], ones)
        np.testing.assert_array_equal(data["bool_array"], bool_array)
        np.testing.assert_array_equal(data["float16_array"], float16_array)
        np.testing.assert_array_equal(data["float32_array"], float32_array)
        np.testing.assert_array_equal(data["float64_array"], float64_array)
        np.testing.assert_array_equal(data["uint8_array"], uint8_array)
        np.testing.assert_array_equal(data["int16_array"], int16_array)
        np.testing.assert_array_equal(data["uint16_array"], uint16_array)
        np.testing.assert_array_equal(data["int32_array"], int32_array)
        np.testing.assert_array_equal(data["uint32_array"], uint32_array)
        np.testing.assert_array_equal(data["int64_array"], int64_array)
        np.testing.assert_array_equal(data["uint64_array"], uint64_array)

    def test_load_npy_cpp(self) -> None:
        """load_npy can handle version 1, 2 and 3."""
        for shape in [(), (3,), (3, 4, 5)]:
            ref = np.random.randint(255, size=shape)
            data = _dump_npy(ref)

            buffer = spdl.io.load_npy(data)
            hyp = np.array(buffer, copy=False)
            np.testing.assert_array_equal(hyp, ref)
