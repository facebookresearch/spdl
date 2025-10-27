# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import sys
import unittest
from tempfile import NamedTemporaryFile

import numpy as np
import spdl.io
from parameterized import parameterized

from ..fixture import FFMPEG_CLI, get_sample, load_ref_audio

sample_fmt2dtype = {
    "s16": np.int16,
    "s32": np.int32,
    "s64": np.int64,
    "s16p": np.int16,
    "s32p": np.int32,
    "s64p": np.int64,
    "f32": np.float32,
    "f64": np.float64,
    "flt": np.float32,
    "fltp": np.float32,
    "dbl": np.float64,
    "dblp": np.float64,
}


class TestEncodeAudioInteger(unittest.TestCase):
    @parameterized.expand(
        [
            ("s16",),
            ("s32",),
            ("s64",),
        ]
    )
    def test_encode_audio_integer(self, sample_fmt: str) -> None:
        """Can save audio from integer data"""
        sample_rate = 44100
        duration = 3
        num_channels = 2

        dtype = sample_fmt2dtype[sample_fmt]

        shape = (sample_rate * duration, num_channels)
        ii = np.iinfo(dtype)
        ref = np.random.randint(ii.min, ii.max, size=shape, dtype=dtype)

        with NamedTemporaryFile(suffix=".wav") as f:
            f.close()  # for windows

            muxer = spdl.io.Muxer(f.name)
            encoder = muxer.add_encode_stream(
                config=spdl.io.audio_encode_config(
                    num_channels=num_channels,
                    sample_fmt=sample_fmt,
                    sample_rate=sample_rate,
                ),
                encoder=f"pcm_{sample_fmt}le",
            )

            with muxer.open():
                for i in range(duration):
                    start = i * sample_rate
                    frames = spdl.io.create_reference_audio_frame(
                        array=ref[start : start + sample_rate, :],
                        sample_fmt=sample_fmt,
                        sample_rate=sample_rate,
                        pts=start,
                    )
                    print(frames)

                    if (packets := encoder.encode(frames)) is not None:
                        print(packets)
                        muxer.write(0, packets)

                if (packets := encoder.flush()) is not None:
                    print(packets)
                    muxer.write(0, packets)

            if sample_fmt != "s64":
                hyp = load_ref_audio(
                    f.name,
                    filter_desc=None,
                    shape=shape,
                    format=f"{sample_fmt}le",
                    dtype=dtype,
                )

                np.testing.assert_array_equal(hyp, ref)


class TestEncodeAudioFloat(unittest.TestCase):
    @parameterized.expand(
        [
            ("f32",),
            ("f64",),
        ]
    )
    def test_encode_audio_float(self, sample_fmt: str) -> None:
        """Can save audio from floating point data"""
        sample_rate = 44100
        duration = 3
        num_channels = 2

        dtype = sample_fmt2dtype[sample_fmt]
        format = "dbl" if sample_fmt == "f64" else "flt"

        shape = (sample_rate * duration, num_channels)
        ref = np.random.rand(*shape).astype(dtype=dtype)

        with NamedTemporaryFile(suffix=".wav") as f:
            f.close()  # for windows

            muxer = spdl.io.Muxer(f.name)
            encoder = muxer.add_encode_stream(
                config=spdl.io.audio_encode_config(
                    num_channels=num_channels,
                    sample_fmt=format,
                    sample_rate=sample_rate,
                ),
                encoder=f"pcm_{sample_fmt}le",
            )

            with muxer.open():
                for i in range(duration):
                    start = i * sample_rate
                    frames = spdl.io.create_reference_audio_frame(
                        array=ref[start : start + sample_rate, :],
                        sample_fmt=format,
                        sample_rate=sample_rate,
                        pts=start,
                    )
                    print(frames)

                    if (packets := encoder.encode(frames)) is not None:
                        print(packets)
                        muxer.write(0, packets)

                if (packets := encoder.flush()) is not None:
                    print(packets)
                    muxer.write(0, packets)

            hyp = load_ref_audio(
                f.name,
                filter_desc=None,
                shape=shape,
                format=f"{sample_fmt}le",
                dtype=dtype,
            )

            np.testing.assert_array_equal(hyp, ref)


class TestEncodeAudioIntegerPlanar(unittest.TestCase):
    @parameterized.expand(
        [
            ("s16p",),
            ("s32p",),
        ]
    )
    def test_encode_audio_integer_planar(self, sample_fmt: str) -> None:
        sample_rate = 44100
        duration = 3
        num_channels = 32
        # more than 8 channels require special handling in the internal implementation.
        # So we need to test that.

        dtype = sample_fmt2dtype[sample_fmt]

        shape = (num_channels, sample_rate * duration)
        ii = np.iinfo(dtype)
        ref = np.random.randint(ii.min, ii.max, size=shape, dtype=dtype)

        with NamedTemporaryFile(suffix=".nut") as f:
            f.close()  # windows

            muxer = spdl.io.Muxer(f.name)
            encoder = muxer.add_encode_stream(
                config=spdl.io.audio_encode_config(
                    num_channels=num_channels,
                    sample_fmt=sample_fmt,
                    sample_rate=sample_rate,
                ),
                encoder=f"pcm_{sample_fmt[:-1]}le_planar",
            )

            with muxer.open():
                for i in range(duration):
                    start = i * sample_rate
                    frames = spdl.io.create_reference_audio_frame(
                        array=ref[:, start : start + sample_rate],
                        sample_fmt=sample_fmt,
                        sample_rate=sample_rate,
                        pts=start,
                    )
                    print(frames)

                    if (packets := encoder.encode(frames)) is not None:
                        print(packets)
                        muxer.write(0, packets)

                if (packets := encoder.flush()) is not None:
                    print(packets)
                    muxer.write(0, packets)

            hyp = load_ref_audio(
                f.name,
                filter_desc=None,
                shape=(-1, num_channels),
                format=f"{sample_fmt[:-1]}le",
                dtype=dtype,
            )

            np.testing.assert_array_equal(hyp.T, ref)


class TestEncodeAudioSmokeTest(unittest.TestCase):
    @parameterized.expand(
        [
            (".mp3", "s16p"),
            (".flac", "s16"),
            (".aac", "fltp"),
        ]
    )
    def test_encode_audio_smoke_test(self, ext: str, sample_fmt: str) -> None:
        """Can save audio data in commoly used format."""

        # On Windows GHA CI, the default mp3 encoder is `mp3_mf`,
        # which does not support s16p.
        # `libmp3lame` seems to be not available.
        # We tried with s16, but there was an encoding failure.
        # https://github.com/facebookresearch/spdl/actions/runs/17614043561/job/50043084187?pr=935#step:5:598
        # TODO: fix te encoding error
        if sys.platform == "win32" and "GITHUB_ACTIONS" in os.environ:
            raise unittest.SkipTest(
                "Windows on GHA's default MP3 encoder mp3_mf does not support s16p. "
                "Encoding fails with s16. We will fix this later."
            )

        sample_rate = 44100
        duration = 3
        num_channels = 2

        is_planar = sample_fmt.endswith("p")
        dtype = sample_fmt2dtype[sample_fmt]
        num_frames = sample_rate * duration

        if is_planar:
            shape = [num_channels, num_frames]
        else:
            shape = [num_frames, num_channels]

        if sample_fmt.startswith("s"):
            ii = np.iinfo(dtype)
            ref = np.random.randint(ii.min, ii.max, size=shape, dtype=dtype)
        else:
            ref = np.random.random(shape).astype(dtype)

        with NamedTemporaryFile(suffix=ext) as f:
            f.close()  # for windows

            muxer = spdl.io.Muxer(f.name)
            encoder = muxer.add_encode_stream(
                config=spdl.io.audio_encode_config(
                    num_channels=num_channels,
                    sample_fmt=sample_fmt,
                    sample_rate=sample_rate,
                ),
            )

            frame_size = encoder.frame_size or 1024

            with muxer.open():
                for start in range(0, sample_rate * duration, frame_size):
                    end = start + frame_size
                    if is_planar:
                        data = ref[:, start:end]
                    else:
                        data = ref[start:end, :]

                    frames = spdl.io.create_reference_audio_frame(
                        array=data,
                        sample_fmt=sample_fmt,
                        sample_rate=sample_rate,
                        pts=start,
                    )
                    print(frames, flush=True)

                    if (packets := encoder.encode(frames)) is not None:
                        print(packets)
                        muxer.write(0, packets)

                if (packets := encoder.flush()) is not None:
                    print(packets)
                    muxer.write(0, packets)


class TestRemuxAudio(unittest.TestCase):
    def test_remux_audio(self) -> None:
        # fmt: off
        cmd = f"""
        {FFMPEG_CLI} -hide_banner -y \
        -f lavfi -i sine=sample_rate=8000:frequency=305:duration=5 \
        -f lavfi -i sine=sample_rate=8000:frequency=300:duration=5 \
        -filter_complex amerge  -c:a pcm_s16le sample.wav
        """
        # fmt: on
        sample = get_sample(cmd)

        demuxer = spdl.io.Demuxer(sample.path)

        with NamedTemporaryFile(suffix=".wav") as f:
            f.close()  # for windows

            muxer = spdl.io.Muxer(f.name)
            muxer.add_remux_stream(demuxer.audio_codec)

            with muxer.open():
                for packets in demuxer.streaming_demux(duration=1):
                    muxer.write(0, packets)

            ref = load_ref_audio(
                sample.path,
                filter_desc=None,
                shape=(40000, 2),
                format="s16le",
                dtype=np.int16,
            )

            hyp = load_ref_audio(
                f.name,
                filter_desc=None,
                shape=(40000, 2),
                format="s16le",
                dtype=np.int16,
            )

            np.testing.assert_array_equal(hyp, ref)
