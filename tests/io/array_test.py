# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gc
import io
import sys
import unittest
from collections.abc import Callable
from io import BytesIO

import numpy as np
import spdl.io
from parameterized import parameterized


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
        # pyrefly: ignore [no-matching-overload]
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
        # pyrefly: ignore [no-matching-overload]
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
                spdl.io.load_npz(b"o" * i)

    def test_parse_zip_no_eocdr_sig(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "Failed to locate the end of the central directory."
        ):
            spdl.io.load_npz((b"foooooooooooooooooooooooooo"))


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


def _reuse_freed_memory(size: int, count: int = 2000) -> list[bytearray]:
    """Fill recently freed memory with a recognizable pattern.

    CPython hands memory from deallocated objects back out to later
    allocations of a similar size, so `count` buffers of `size` bytes are
    likely to land on the block the archive just freed. A stale pointer into
    it then reads `0xAB` and the caller's `assert_array_equal` fails; without
    this, it would likely read the original bytes and pass.

    Best-effort by nature -- the refcount assertions below are the
    deterministic check.
    """
    return [bytearray(b"\xab" * size) for _ in range(count)]


class TestNpzBufferLifetime(unittest.TestCase):
    """`NpzFile` does not copy the archive.

    It holds a raw pointer into the source buffer, and the arrays it returns for
    stored (uncompressed) entries are views into the same memory. Both read
    freed memory unless the source buffer is kept alive.

    The tests come in two flavors. The ones asserting on `sys.getrefcount`
    check the contract directly, and are the authoritative check. The ones
    calling `_reuse_freed_memory` additionally try to turn a violation into an
    observable data corruption; see that function for the caveats.
    """

    def test_load_npz_retains_source(self) -> None:
        """`load_npz` keeps a reference to the source buffer."""
        ref = np.arange(10)
        data = _dump_npz(x=ref)

        num_refs = sys.getrefcount(data)
        npz = spdl.io.load_npz(data)

        self.assertGreater(
            sys.getrefcount(data),
            num_refs,
            "`NpzFile` must keep a reference to the source buffer, "
            "as it holds a pointer into it.",
        )
        np.testing.assert_array_equal(npz["x"], ref)

    def test_getitem_retains_source(self) -> None:
        """Arrays of stored entries keep the source buffer alive.

        Such an array is a view into the archive, so it can outlive the
        `NpzFile` it was retrieved from.
        """
        ref = np.arange(10)
        data = _dump_npz(x=ref)

        num_refs = sys.getrefcount(data)
        # The `NpzFile` is released as soon as the entry is retrieved.
        arr = spdl.io.load_npz(data)["x"]
        gc.collect()

        self.assertGreater(
            sys.getrefcount(data),
            num_refs,
            "The array must keep a reference to the source buffer, "
            "as it is a view into it.",
        )
        np.testing.assert_array_equal(arr, ref)

    @parameterized.expand(
        [
            ("stored", _dump_npz),
            ("deflated", _dump_npz_compressed),
        ]
    )
    def test_load_npz_source_may_be_temporary(
        self, _: str, dump: Callable[..., bytes]
    ) -> None:
        """Entries are readable when the caller does not hold the source."""
        ref = np.arange(1000, dtype=np.int64)
        size = len(dump(x=ref))

        # The source is a temporary, so it is released when `load_npz` returns
        # unless `NpzFile` retains it.
        npz = spdl.io.load_npz(dump(x=ref))
        gc.collect()
        # If `NpzFile` failed to retain the temporary, `npz["x"]` now points
        # into freed memory, and reads `0xAB` instead of `ref`.
        clobber = _reuse_freed_memory(size)

        np.testing.assert_array_equal(npz["x"], ref)

        del clobber

    def test_array_outlives_npz_file(self) -> None:
        """A stored entry stays valid after the `NpzFile` is released."""
        ref = np.arange(1000, dtype=np.int64)
        size = len(_dump_npz(x=ref))

        arr = spdl.io.load_npz(_dump_npz(x=ref))["x"]
        gc.collect()
        # If the source buffer was not retained, `arr` now views freed memory,
        # and reads `0xAB` instead of `ref`.
        clobber = _reuse_freed_memory(size)

        np.testing.assert_array_equal(arr, ref)

        del clobber
