# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import numpy as np
import pytest
import spdl.io
import spdl.io.utils
from spdl.io import get_audio_filter_desc, get_filter_desc

from ..fixture import FFMPEG_CLI, get_sample, load_ref_audio


def _get_format(sample_fmt: str):
    match sample_fmt:
        case "s16p" | "s16":
            return np.int16, "s16le"
        case "fltp" | "flt":
            return np.float32, "f32le"


@pytest.mark.parametrize(
    "sample_fmt",
    ["s16p", "s16", "fltp", "flt"],
)
def test_load_audio(sample_fmt):
    # fmt: off
    cmd = f"""
    {FFMPEG_CLI} -hide_banner -y \
    -f lavfi -i 'sine=sample_rate=8000:frequency=305:duration=5' \
    -f lavfi -i 'sine=sample_rate=8000:frequency=300:duration=5' \
    -filter_complex amerge  -c:a pcm_s16le sample.wav
    """
    # fmt: on
    sample = get_sample(cmd)

    dtype, format = _get_format(sample_fmt)
    shape = (40000, 2)

    filter_desc = get_audio_filter_desc(sample_fmt=sample_fmt)
    buffer = spdl.io.load_audio(src=sample.path, filter_desc=filter_desc)
    hyp = spdl.io.to_numpy(buffer)

    ref = load_ref_audio(
        sample.path, shape, filter_desc=filter_desc, format=format, dtype=dtype
    )
    if sample_fmt.endswith("p"):
        ref = ref.T
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_batch_audio_conversion():
    # fmt: off
    cmd = f"""
    {FFMPEG_CLI} -hide_banner -y \
    -f lavfi -i 'sine=sample_rate=8000:frequency=305:duration=5' \
    -f lavfi -i 'sine=sample_rate=8000:frequency=300:duration=5' \
    -filter_complex amerge  -c:a pcm_s16le sample.wav
    """
    # fmt: on
    sample = get_sample(cmd)
    num_frames = 8_000

    timestamps = [(0, 1), (1, 1.5), (2, 2.7)]

    frames, refs = [], []
    demuxer = spdl.io.Demuxer(sample.path)
    for ts in timestamps:
        packets = demuxer.demux_audio(ts)
        filter_desc = get_filter_desc(packets, num_frames=num_frames)
        frames_ = spdl.io.decode_packets(packets, filter_desc=filter_desc)
        frames.append(frames_)

        ref = load_ref_audio(
            sample.path,
            (num_frames, 2),
            filter_desc=filter_desc,
            format="f32le",
            dtype=np.float32,
        )
        refs.append(ref.T)

    buffer = spdl.io.convert_frames(frames)
    hyp = spdl.io.to_numpy(buffer)
    assert hyp.shape == (3, 2, 8000)

    ref = np.stack(refs)
    np.testing.assert_array_equal(hyp, ref, strict=True)
