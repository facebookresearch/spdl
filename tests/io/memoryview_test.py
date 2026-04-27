# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests that spdl.io functions accept memoryview in addition to bytes."""

import io
import tarfile
import unittest
import wave

import numpy as np
import spdl.io


def _create_wav_data(
    sample_rate: int = 8000,
    num_channels: int = 1,
    bits_per_sample: int = 16,
    num_samples: int = 100,
) -> bytes:
    dtype = {8: np.uint8, 16: np.int16, 32: np.int32}[bits_per_sample]
    samples = np.zeros((num_samples, num_channels), dtype=dtype)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(num_channels)
        w.setsampwidth(bits_per_sample // 8)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _create_tar_data() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="hello.txt")
        content = b"Hello, World!"
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf.getvalue()


def _create_npz_data() -> bytes:
    buf = io.BytesIO()
    np.savez(buf, x=np.arange(5), allow_pickle=False)
    buf.seek(0)
    return buf.read()


class TestLoadWavMemoryview(unittest.TestCase):
    def test_load_wav_with_memoryview(self) -> None:
        wav_data = _create_wav_data()
        mv = memoryview(wav_data)

        result_bytes = spdl.io.load_wav(wav_data)
        result_mv = spdl.io.load_wav(mv)

        samples_bytes = spdl.io.to_numpy(result_bytes)
        samples_mv = spdl.io.to_numpy(result_mv)

        np.testing.assert_array_equal(samples_mv, samples_bytes)

    def test_load_wav_memoryview_with_time_window(self) -> None:
        wav_data = _create_wav_data(sample_rate=8000, num_samples=8000)
        mv = memoryview(wav_data)

        result = spdl.io.load_wav(mv, time_offset_seconds=0.5, duration_seconds=0.5)
        samples = spdl.io.to_numpy(result)

        self.assertEqual(samples.shape[0], 4000)


class TestParseWavMemoryview(unittest.TestCase):
    def test_parse_wav_with_memoryview(self) -> None:
        wav_data = _create_wav_data(sample_rate=44100, num_channels=2)
        mv = memoryview(wav_data)

        header_bytes = spdl.io.parse_wav(wav_data)
        header_mv = spdl.io.parse_wav(mv)

        self.assertEqual(header_mv.sample_rate, header_bytes.sample_rate)
        self.assertEqual(header_mv.num_channels, header_bytes.num_channels)
        self.assertEqual(header_mv.bits_per_sample, header_bytes.bits_per_sample)
        self.assertEqual(header_mv.data_size, header_bytes.data_size)


class TestIterTarfileMemoryview(unittest.TestCase):
    def test_iter_tarfile_with_memoryview(self) -> None:
        tar_data = _create_tar_data()
        mv = memoryview(tar_data)

        entries_bytes = list(spdl.io.iter_tarfile(tar_data))
        entries_mv = list(spdl.io.iter_tarfile(mv))

        self.assertEqual(len(entries_mv), len(entries_bytes))
        for (name_b, content_b), (name_m, content_m) in zip(
            entries_bytes, entries_mv, strict=True
        ):
            self.assertEqual(name_m, name_b)
            self.assertEqual(bytes(content_m), bytes(content_b))


class TestLoadNpzMemoryview(unittest.TestCase):
    def test_load_npz_with_memoryview(self) -> None:
        npz_data = _create_npz_data()
        mv = memoryview(npz_data)

        data = spdl.io.load_npz(mv)
        np.testing.assert_array_equal(data["x"], np.arange(5))
