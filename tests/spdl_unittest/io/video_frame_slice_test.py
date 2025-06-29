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
from spdl.io import get_video_filter_desc

from ..fixture import FFMPEG_CLI, get_sample


def _to_numpy(frames):
    buffer = spdl.io.convert_frames(frames)
    return spdl.io.to_numpy(buffer)


def _decode_video(src, pix_fmt=None):
    return spdl.io.decode_packets(
        spdl.io.demux_video(src),
        filter_desc=get_video_filter_desc(pix_fmt=pix_fmt),
    )


def test_video_frames_getitem_slice():
    """VideoFrames.__getitem__ works for slice input"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    sample = get_sample(cmd)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == 100
    f2 = frames[::2]
    f3 = frames[::3]

    arr = _to_numpy(frames)

    assert len(f2) == 50
    assert np.array_equal(arr[::2], _to_numpy(f2))

    assert len(f3) == 34
    assert np.array_equal(arr[::3], _to_numpy(f3))


def test_video_frames_getitem_int():
    """VideoFrames.__getitem__ works for index input"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n
    frames_split = [frames[i] for i in range(n)]

    arr = _to_numpy(frames)
    for i in range(n):
        arr0 = _to_numpy(frames_split[i])
        assert np.array_equal(arr0, arr[i])


def test_video_frames_getitem_negative_int():
    """VideoFrames.__getitem__ works for negative index input"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n
    frames_split = [frames[-i - 1] for i in range(n)]

    arr = _to_numpy(frames)
    for i in range(n):
        arr0 = _to_numpy(frames_split[i])
        assert np.array_equal(arr0, arr[-i - 1])


def test_video_frames_iterate():
    """VideoFrames can be iterated"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    arrs = [_to_numpy(f) for f in frames]
    array = _to_numpy(frames)

    for i in range(n):
        assert np.array_equal(array[i], arrs[i])


def test_video_frames_list_slice():
    """VideoFrames can be sliced with list of integers"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    # The valid value range is [-n, n)
    idx = [0, 99, 1, 3, -1, -100]

    sampled_frames = frames[idx]

    refs = _to_numpy(frames)
    array = _to_numpy(sampled_frames)

    for i in range(len(idx)):
        assert np.array_equal(array[i], refs[idx[i]])


def test_video_frames_list_slice_empty():
    """VideoFrames can be sliced with an empty list"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    # The valid value range is [-n, n)
    sampled_frames = frames[[]]

    assert len(sampled_frames) == 0


def test_video_frames_list_slice_out_of_range():
    """Slicing VideoFrames with an out-of-range value fails"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    # The valid value range is [-n, n)
    with pytest.raises(IndexError):
        frames[[n]]

    # The valid value range is [-n, n)
    with pytest.raises(IndexError):
        frames[[-n - 1]]
