# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import spdl.io

CMDS = {
    "audio": "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav",
    "video": "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 25 sample.mp4",
    "image": "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


def _load_from_frames(frames):
    buffer = spdl.io.convert_frames(frames)
    return spdl.io.to_numpy(buffer)


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_frames(media_type, get_sample):
    """Cloning frames allows to decode twice"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    demux_func = {
        "audio": spdl.io.demux_audio,
        "video": spdl.io.demux_video,
        "image": spdl.io.demux_image,
    }[media_type]

    frames1 = spdl.io.decode_packets(demux_func(sample.path))
    frames2 = frames1.clone()

    array1 = _load_from_frames(frames1)
    array2 = _load_from_frames(frames2)

    assert np.all(array1 == array2)


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_invalid_frames(media_type, get_sample):
    """Attempt to clone already released frames raises RuntimeError instead of segfault"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    demux_func = {
        "audio": spdl.io.demux_audio,
        "video": spdl.io.demux_video,
        "image": spdl.io.demux_image,
    }[media_type]

    frames = spdl.io.decode_packets(demux_func(sample.path))
    _ = spdl.io.convert_frames(frames)
    with pytest.raises(TypeError):
        frames.clone()


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_clone_frames_multi(media_type, get_sample):
    """Can clone multiple times"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)
    N = 100

    demux_func = {
        "audio": spdl.io.demux_audio,
        "video": spdl.io.demux_video,
        "image": spdl.io.demux_image,
    }[media_type]

    frames = spdl.io.decode_packets(demux_func(sample.path))
    clones = [frames.clone() for _ in range(N)]

    array = _load_from_frames(frames)
    arrays = [_load_from_frames(c) for c in clones]

    for i in range(N):
        assert np.all(array == arrays[i])
