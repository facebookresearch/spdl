# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import gc
import io
import struct
import sys
import unittest
import weakref
import zipfile
from collections.abc import Callable
from io import BytesIO

import numpy as np
import spdl.io
from parameterized import parameterized
from spdl.io import lib as _libspdl


def _dump_npy(arr: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return buffer.getvalue()


def _dump_npy_with_descr(arr: np.ndarray, descr: str) -> bytes:
    header = (
        f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {arr.shape!r}, }}"
    ).encode()
    preamble_size = 10
    padding = (64 - ((preamble_size + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    return (
        b"\x93NUMPY\x01\x00"
        + len(header).to_bytes(2, "little")
        + header
        + arr.tobytes()
    )


class TestLoadNpy(unittest.TestCase):
    @parameterized.expand(
        [
            ("int32",),
            ("double",),
            ("half",),
            ("int",),
            ("d",),
            ("e",),
            ("datetime64[ns]",),
        ]
    )
    def test_load_npy_accepts_dtype_alias(self, descr: str) -> None:
        """The loader accepts scalar dtype aliases understood by NumPy."""
        dtype = np.dtype(descr)
        ref = np.array([1, 2], dtype=dtype)

        array = spdl.io.load_npy(_dump_npy_with_descr(ref, descr))

        self.assertEqual(array.dtype, dtype)
        np.testing.assert_array_equal(array, ref)

    def test_load_npy_rejects_structured_dtype(self) -> None:
        """Structured records are rejected instead of being misparsed as scalars."""
        dtype = np.dtype("i4,f8")
        ref = np.array([(1, 2.0)], dtype=dtype)

        with self.assertRaisesRegex(RuntimeError, "Structured NPY dtypes"):
            spdl.io.load_npy(_dump_npy(ref))

    def test_compressed_binding_accepts_size_t_arguments(self) -> None:
        """Compressed sizes are not narrowed to 32 bits by the binding."""
        # Nanobind converts compressed_size before entering the C++ body. The
        # former uint32_t binding rejected 1 << 32; reaching the unrelated
        # uncompressed_size validation proves that size_t accepted the value
        # without allocating a buffer larger than 4 GiB.
        with self.assertRaisesRegex(RuntimeError, "uncompressed_size"):
            _libspdl._archive.load_npy_compressed(
                1,
                0,
                1 << 32,
                0,
            )

    def test_zero_copy_array_retains_source(self) -> None:
        """A zero-copy array keeps its borrowed NPY source alive."""
        ref = np.arange(10, dtype=np.int64)
        src = np.frombuffer(_dump_npy(ref), dtype=np.uint8).copy()
        src_ref = weakref.ref(src)

        arr = spdl.io.load_npy(src)
        del src
        gc.collect()

        self.assertIsNotNone(src_ref())
        np.testing.assert_array_equal(arr, ref)

        del arr
        gc.collect()
        self.assertIsNone(src_ref())

    def test_zero_copy_array_inherits_source_writability(self) -> None:
        """A zero-copy array cannot mutate an immutable source buffer."""
        ref = np.arange(10, dtype=np.int64)

        readonly = spdl.io.load_npy(_dump_npy(ref))
        writable = spdl.io.load_npy(bytearray(_dump_npy(ref)))
        copied = spdl.io.load_npy(_dump_npy(ref), copy=True)

        # @lint-ignore SPELL NumPy spells the public flag this way.
        self.assertFalse(readonly.flags.writeable)
        # @lint-ignore SPELL NumPy spells the public flag this way.
        self.assertTrue(writable.flags.writeable)
        # @lint-ignore SPELL NumPy spells the public flag this way.
        self.assertTrue(copied.flags.writeable)
        with self.assertRaises(ValueError):
            readonly[0] = -1
        copied[0] = -1
        self.assertEqual(copied[0], -1)

    def test_load_npy_preserves_fortran_order(self) -> None:
        """The loader exposes Fortran-contiguous payloads with byte strides."""
        ref = np.asfortranarray(np.arange(12).reshape(3, 4))

        array = spdl.io.load_npy(_dump_npy(ref))

        np.testing.assert_array_equal(array, ref)
        self.assertEqual(array.strides, ref.strides)
        self.assertTrue(array.flags.f_contiguous)
        self.assertFalse(array.flags.c_contiguous)

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
        np.testing.assert_array_equal(recon, ref, strict=True)

        # Use bytearray to check if the change to the original is refrected to the recon
        # (which means that the recon is referring to the original, no copy)
        data = bytearray(data)
        print(f"{id(data)=}")
        print(f"{id(recon.data.obj)=}")
        recon = spdl.io.load_npy(data)
        np.testing.assert_array_equal(recon, ref, strict=True)

        self.assertTrue(np.any(recon))
        # Fill zeros. The header is cleared too, but it's already parsed, so not an issue.
        data[:] = b"\x00" * len(data)
        np.testing.assert_array_equal(recon, 0)

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
        np.testing.assert_array_equal(recon, ref, strict=True)

        # Use bytearray to check if the change to the original is refrected to the recon
        # (which means that the recon is referring to the original, no copy)
        data = bytearray(data)
        print(f"{id(data)=}")
        print(f"{id(recon.data.obj)=}")
        recon = spdl.io.load_npy(data)
        np.testing.assert_array_equal(recon, ref, strict=True)

        self.assertTrue(np.any(recon))
        # Fill zeros. The header is cleared too, but it's already parsed, so not an issue.
        data[:] = b"\x00" * len(data)
        np.testing.assert_array_equal(recon, 0)


##############################################################################
# NPZ
##############################################################################


class TestParseZip(unittest.TestCase):
    def test_npz_file_rejects_unsupported_compression_method(self) -> None:
        """Unsupported methods cannot bypass decompression validation."""
        with self.assertRaisesRegex(ValueError, "unsupported compression method"):
            spdl.io.NpzFile(
                b"x",
                {"x.npy": (0, 1, 1 << 60, 99)},
            )

    def test_npz_file_limits_declared_decompression_size(self) -> None:
        """Untrusted metadata cannot request an unbounded native allocation."""
        data = b"x"
        oversized_meta = {"x.npy": (0, 1, (1 << 30) + 1, 8)}

        with self.assertRaisesRegex(ValueError, "max_uncompressed_bytes"):
            spdl.io.NpzFile(data, oversized_meta)

        archive = spdl.io.NpzFile(
            data,
            oversized_meta,
            max_uncompressed_bytes=(1 << 30) + 1,
        )
        self.assertEqual(len(archive), 1)

    def test_load_npz_enforces_cumulative_decompression_budget(self) -> None:
        """The convenience loader forwards its explicit allocation budget."""
        data = _dump_npz_compressed(x=np.arange(4), y=np.arange(4))

        with self.assertRaisesRegex(ValueError, "max_uncompressed_bytes"):
            spdl.io.load_npz(data, max_uncompressed_bytes=1)

    def test_npz_file_accepts_legacy_metadata_without_crc(self) -> None:
        """Legacy four-field metadata loads stored and deflated entries."""
        ref = np.arange(4)
        for name, data in (
            ("stored", _dump_npz(x=ref)),
            ("deflated", _dump_npz_compressed(x=ref)),
        ):
            with self.subTest(name=name):
                with zipfile.ZipFile(BytesIO(data)) as archive:
                    info = archive.getinfo("x.npy")
                filename_size, extra_size = struct.unpack_from(
                    "<HH", data, info.header_offset + 26
                )
                data_offset = info.header_offset + 30 + filename_size + extra_size
                legacy_meta = {
                    info.filename: (
                        data_offset,
                        info.compress_size,
                        info.file_size,
                        info.compress_type,
                    )
                }

                restored = spdl.io.NpzFile(data, legacy_meta)

                np.testing.assert_array_equal(restored["x"], ref)

    def test_npz_file_rejects_out_of_bounds_public_metadata(self) -> None:
        """Public metadata cannot reach the native loader outside its buffer."""
        data = b"data"
        cases = [
            (len(data) + 1, 0),
            (0, len(data) + 1),
            (len(data), 1),
            (-1, 1),
            (0, -1),
        ]
        for offset, compressed_size in cases:
            with self.subTest(offset=offset, compressed_size=compressed_size):
                with self.assertRaisesRegex(ValueError, "payload|negative"):
                    spdl.io.NpzFile(
                        data,
                        {
                            "x.npy": (
                                offset,
                                compressed_size,
                                compressed_size,
                                0,
                            )
                        },
                    )

    def test_npz_file_snapshots_public_metadata(self) -> None:
        """Caller mutation cannot replace previously validated payload bounds."""
        ref = np.arange(4)
        data = _dump_npy(ref)
        meta = {"x.npy": (0, len(data), len(data), 0)}
        archive = spdl.io.NpzFile(data, meta)

        meta.clear()

        np.testing.assert_array_equal(archive["x"], ref)

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

    def test_parse_zip_with_unaligned_comment_length(self) -> None:
        """The EOCD search examines every byte offset allowed by ZIP."""
        data = bytearray(_dump_npz(x=np.arange(4)))
        data[-2:] = struct.pack("<H", 1)
        data.append(ord("x"))

        np.testing.assert_array_equal(spdl.io.load_npz(data)["x"], np.arange(4))

    def test_parse_zip_allows_trailing_data(self) -> None:
        """Bytes after the declared EOCD comment do not hide the archive."""
        ref = np.arange(4)
        data = _dump_npz(x=ref) + b"trailing data"

        np.testing.assert_array_equal(spdl.io.load_npz(data)["x"], ref)

    def test_parse_zip_ignores_false_unsupported_eocd_candidates(self) -> None:
        """Unsupported-looking signatures do not stop the EOCD search."""
        false_candidates = {
            "multi-disk": struct.pack("<IHHHHIIH", 0x06054B50, 1, 0, 0, 0, 0, 0, 0),
            "zip64": struct.pack(
                "<IHHHHIIH",
                0x06054B50,
                0,
                0,
                0xFFFF,
                0xFFFF,
                0xFFFFFFFF,
                0xFFFFFFFF,
                0,
            ),
        }
        ref = np.arange(4)
        for name, false_candidate in false_candidates.items():
            with self.subTest(name=name):
                data = _dump_npz(x=ref) + false_candidate

                np.testing.assert_array_equal(spdl.io.load_npz(data)["x"], ref)

    def test_parse_zip_rejects_selected_unsupported_eocd(self) -> None:
        """A selected multi-disk or ZIP64 EOCD remains unsupported."""
        cases = [
            ("multi-disk", 4, 1, "Multi-disk ZIP"),
            ("zip64", 10, 0xFFFF, "ZIP64 central directories"),
        ]
        for name, field_offset, field_value, message in cases:
            with self.subTest(name=name):
                data = bytearray(_dump_npz(x=np.arange(4)))
                eocd_offset = data.rfind(b"PK\x05\x06")
                self.assertGreaterEqual(eocd_offset, 0)
                struct.pack_into("<H", data, eocd_offset + field_offset, field_value)

                with self.assertRaisesRegex(RuntimeError, message):
                    spdl.io.load_npz(data)

    def test_parse_zip_rejects_gap_before_eocd(self) -> None:
        """The central directory must end immediately before the EOCD."""
        data = bytearray(_dump_npz(x=np.arange(4)))
        eocd_offset = data.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(eocd_offset, 0)
        data[eocd_offset:eocd_offset] = b"gap"

        with self.assertRaisesRegex(
            RuntimeError,
            "central directory does not end at the EOCD",
        ):
            spdl.io.load_npz(data)

    def test_parse_zip_rejects_out_of_bounds_payload(self) -> None:
        """A central-directory size cannot expose bytes outside the archive."""
        data = bytearray(_dump_npz(x=np.arange(4)))
        cd_offset = data.find(b"PK\x01\x02")
        self.assertGreaterEqual(cd_offset, 0)
        struct.pack_into("<I", data, cd_offset + 20, len(data))

        with self.assertRaisesRegex(ValueError, "payload is outside"):
            spdl.io.load_npz(data)

    def test_parse_zip_rejects_mismatched_stored_sizes(self) -> None:
        """Stored ZIP entries must declare identical compressed and raw sizes."""
        data = bytearray(_dump_npz(x=np.arange(4)))
        cd_offset = data.find(b"PK\x01\x02")
        self.assertGreaterEqual(cd_offset, 0)
        compressed_size = struct.unpack_from("<I", data, cd_offset + 20)[0]
        struct.pack_into("<I", data, cd_offset + 24, compressed_size + 1)

        with self.assertRaisesRegex(ValueError, "matching compressed"):
            spdl.io.load_npz(data)

    def test_parse_zip64_entry_metadata(self) -> None:
        """ZIP64 sizes are read from the central directory's extra field."""
        ref = np.arange(4)
        data = bytearray(_dump_npz(x=ref))
        cd_offset = data.find(b"PK\x01\x02")
        eocd_offset = data.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(cd_offset, 0)
        self.assertGreater(eocd_offset, cd_offset)

        compressed_size = struct.unpack_from("<I", data, cd_offset + 20)[0]
        uncompressed_size = struct.unpack_from("<I", data, cd_offset + 24)[0]
        filename_size = struct.unpack_from("<H", data, cd_offset + 28)[0]
        extra_size = struct.unpack_from("<H", data, cd_offset + 30)[0]
        insert_at = cd_offset + 46 + filename_size + extra_size
        zip64_extra = struct.pack(
            "<HHQQ", 0x0001, 16, uncompressed_size, compressed_size
        )

        struct.pack_into("<II", data, cd_offset + 20, 0xFFFFFFFF, 0xFFFFFFFF)
        struct.pack_into("<H", data, cd_offset + 30, extra_size + len(zip64_extra))
        data[insert_at:insert_at] = zip64_extra
        eocd_offset += len(zip64_extra)
        cd_size = struct.unpack_from("<I", data, eocd_offset + 12)[0]
        struct.pack_into("<I", data, eocd_offset + 12, cd_size + len(zip64_extra))

        np.testing.assert_array_equal(spdl.io.load_npz(data)["x"], ref)

    def test_parse_zip64_rejects_truncated_required_fields(self) -> None:
        """Every sentinel-backed ZIP64 value must fit in its extra field."""
        sentinel_fields = [
            (20, "<I", 0xFFFFFFFF),
            (24, "<I", 0xFFFFFFFF),
            (34, "<H", 0xFFFF),
            (42, "<I", 0xFFFFFFFF),
        ]
        for field_offset, field_format, sentinel in sentinel_fields:
            with self.subTest(field_offset=field_offset):
                data = bytearray(_dump_npz(x=np.arange(4)))
                cd_offset = data.find(b"PK\x01\x02")
                eocd_offset = data.rfind(b"PK\x05\x06")
                self.assertGreaterEqual(cd_offset, 0)
                self.assertGreater(eocd_offset, cd_offset)

                filename_size = struct.unpack_from("<H", data, cd_offset + 28)[0]
                extra_size = struct.unpack_from("<H", data, cd_offset + 30)[0]
                insert_at = cd_offset + 46 + filename_size + extra_size
                truncated_zip64_extra = struct.pack("<HH", 0x0001, 0)

                struct.pack_into(field_format, data, cd_offset + field_offset, sentinel)
                struct.pack_into(
                    "<H",
                    data,
                    cd_offset + 30,
                    extra_size + len(truncated_zip64_extra),
                )
                data[insert_at:insert_at] = truncated_zip64_extra
                eocd_offset += len(truncated_zip64_extra)
                cd_size = struct.unpack_from("<I", data, eocd_offset + 12)[0]
                struct.pack_into(
                    "<I",
                    data,
                    eocd_offset + 12,
                    cd_size + len(truncated_zip64_extra),
                )

                with self.assertRaisesRegex(ValueError, "ZIP64 extended metadata"):
                    spdl.io.load_npz(data)

    def test_parse_zip64_rejects_unrequested_fields(self) -> None:
        """ZIP64 fields must correspond exactly to sentinel-backed values."""
        data = bytearray(_dump_npz(x=np.arange(4)))
        cd_offset = data.find(b"PK\x01\x02")
        eocd_offset = data.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(cd_offset, 0)
        self.assertGreater(eocd_offset, cd_offset)

        compressed_size = struct.unpack_from("<I", data, cd_offset + 20)[0]
        uncompressed_size = struct.unpack_from("<I", data, cd_offset + 24)[0]
        filename_size = struct.unpack_from("<H", data, cd_offset + 28)[0]
        extra_size = struct.unpack_from("<H", data, cd_offset + 30)[0]
        insert_at = cd_offset + 46 + filename_size + extra_size
        zip64_extra = struct.pack(
            "<HHQQ", 0x0001, 16, uncompressed_size, compressed_size
        )

        struct.pack_into("<I", data, cd_offset + 20, 0xFFFFFFFF)
        struct.pack_into("<H", data, cd_offset + 30, extra_size + len(zip64_extra))
        data[insert_at:insert_at] = zip64_extra
        eocd_offset += len(zip64_extra)
        cd_size = struct.unpack_from("<I", data, eocd_offset + 12)[0]
        struct.pack_into("<I", data, eocd_offset + 12, cd_size + len(zip64_extra))

        with self.assertRaisesRegex(ValueError, "ZIP64 extended metadata"):
            spdl.io.load_npz(data)

    def test_rejects_crc_mismatch(self) -> None:
        """Stored and compressed entries are checked against their CRC-32."""
        for name, dump in (
            ("stored", _dump_npz),
            ("deflated", _dump_npz_compressed),
        ):
            with self.subTest(name=name):
                data = bytearray(dump(x=np.arange(4)))
                cd_offset = data.find(b"PK\x01\x02")
                self.assertGreaterEqual(cd_offset, 0)
                crc32 = struct.unpack_from("<I", data, cd_offset + 16)[0]
                struct.pack_into("<I", data, cd_offset + 16, crc32 ^ 1)

                archive = spdl.io.load_npz(data)
                with self.assertRaisesRegex(RuntimeError, "CRC-32"):
                    archive["x"]


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
    def test_rejects_incorrect_uncompressed_size(self) -> None:
        """DEFLATE output must match the central directory's declared size."""
        data = bytearray(_dump_npz_compressed(x=np.arange(4)))
        cd_offset = data.find(b"PK\x01\x02")
        self.assertGreaterEqual(cd_offset, 0)
        uncompressed_size = struct.unpack_from("<I", data, cd_offset + 24)[0]
        struct.pack_into("<I", data, cd_offset + 24, uncompressed_size + 1)

        archive = spdl.io.load_npz(bytes(data))
        with self.assertRaisesRegex(RuntimeError, "Failed to decompress"):
            archive["x"]

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

    def test_array_writability_matches_entry_storage(self) -> None:
        """Stored entries inherit the source; inflated entries own storage."""
        ref = np.arange(3)
        stored_archive = _dump_npz(x=ref)

        readonly = spdl.io.load_npz(stored_archive)["x"]
        writable = spdl.io.load_npz(memoryview(bytearray(stored_archive)))["x"]
        inflated = spdl.io.load_npz(_dump_npz_compressed(x=ref))["x"]

        self.assertFalse(readonly.flags["W"])
        self.assertTrue(writable.flags["W"])
        self.assertTrue(inflated.flags["W"])
        with self.assertRaisesRegex(ValueError, "read-only"):
            readonly[0] = 10
        writable[0] = 11
        inflated[0] = 12
        self.assertEqual(writable[0], 11)
        self.assertEqual(inflated[0], 12)

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
