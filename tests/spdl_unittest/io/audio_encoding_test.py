# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import spdl.io

from ..fixture import get_sample, load_ref_audio

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


@pytest.mark.parametrize("sample_fmt", ["s16", "s32", "s64"])
def test_encode_audio_integer(sample_fmt):
    """Can save audio from integer data"""
    sample_rate = 44100
    duration = 3
    num_channels = 2

    dtype = sample_fmt2dtype[sample_fmt]

    shape = [sample_rate * duration, num_channels]
    ii = np.iinfo(dtype)
    ref = np.random.randint(ii.min, ii.max, size=shape, dtype=dtype)

    with NamedTemporaryFile(suffix=".wav") as f:
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


@pytest.mark.parametrize("sample_fmt", ["f32", "f64"])
def test_encode_audio_float(sample_fmt):
    """Can save audio from floating point data"""
    sample_rate = 44100
    duration = 3
    num_channels = 2

    dtype = sample_fmt2dtype[sample_fmt]
    format = "dbl" if sample_fmt == "f64" else "flt"

    shape = [sample_rate * duration, num_channels]
    ref = np.random.rand(*shape).astype(dtype=dtype)

    with NamedTemporaryFile(suffix=".wav") as f:
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


@pytest.mark.parametrize("sample_fmt", ["s16p", "s32p"])
def test_encode_audio_integer_planar(sample_fmt):
    sample_rate = 44100
    duration = 3
    num_channels = 32
    # more than 8 channels require special handling in the internal implementation.
    # So we need to test that.

    dtype = sample_fmt2dtype[sample_fmt]

    shape = [num_channels, sample_rate * duration]
    ii = np.iinfo(dtype)
    ref = np.random.randint(ii.min, ii.max, size=shape, dtype=dtype)

    with NamedTemporaryFile(suffix=".nut") as f:
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
            shape=[-1, num_channels],
            format=f"{sample_fmt[:-1]}le",
            dtype=dtype,
        )

        np.testing.assert_array_equal(hyp.T, ref)


@pytest.mark.parametrize(
    "ext,sample_fmt", [(".mp3", "s16p"), (".flac", "s16"), (".aac", "fltp")]
)
def test_encode_audio_smoke_test(ext, sample_fmt):
    """Can save audio data in commoly used format."""
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


def test_remux_audio():
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y \
    -f lavfi -i 'sine=sample_rate=8000:frequency=305:duration=5' \
    -f lavfi -i 'sine=sample_rate=8000:frequency=300:duration=5' \
    -filter_complex amerge  -c:a pcm_s16le sample.wav
    """
    # fmt: on
    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)

    with NamedTemporaryFile(suffix=".wav") as f:
        muxer = spdl.io.Muxer(f.name)
        muxer.add_remux_stream(demuxer.audio_codec)

        with muxer.open():
            for packets in demuxer.streaming_demux(duration=1):
                muxer.write(0, packets)

        ref = load_ref_audio(
            sample.path,
            filter_desc=None,
            shape=[40000, 2],
            format="s16le",
            dtype=np.int16,
        )

        hyp = load_ref_audio(
            f.name,
            filter_desc=None,
            shape=[40000, 2],
            format="s16le",
            dtype=np.int16,
        )

        np.testing.assert_array_equal(hyp, ref)
