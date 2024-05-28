import asyncio

import numpy as np
import pytest

import spdl.io
import spdl.utils
from spdl.io import get_video_filter_desc


def _to_numpy(frames):
    buffer = asyncio.run(spdl.io.async_convert_frames(frames))
    return spdl.io.to_numpy(buffer)


def _decode_video(src, pix_fmt=None):
    async def _decode():
        return await spdl.io.async_decode_packets(
            await spdl.io.async_demux_video(src),
            filter_desc=get_video_filter_desc(pix_fmt=pix_fmt),
        )

    return asyncio.run(_decode())


def test_video_frames_getitem_slice(get_sample):
    """FFmpegVideoFrames.__getitem__ works for slice input"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == 100
    f2 = frames[::2]
    f3 = frames[::3]

    arr = _to_numpy(frames)

    assert len(f2) == 50
    assert np.array_equal(arr[::2], _to_numpy(f2))

    assert len(f3) == 34
    assert np.array_equal(arr[::3], _to_numpy(f3))


def test_video_frames_getitem_int(get_sample):
    """FFmpegVideoFrames.__getitem__ works for index input"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n
    frames_split = [frames[i] for i in range(n)]

    arr = _to_numpy(frames)
    for i in range(n):
        arr0 = _to_numpy(frames_split[i])
        assert np.array_equal(arr0, arr[i])


def test_video_frames_getitem_negative_int(get_sample):
    """FFmpegVideoFrames.__getitem__ works for negative index input"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n
    frames_split = [frames[-i - 1] for i in range(n)]

    arr = _to_numpy(frames)
    for i in range(n):
        arr0 = _to_numpy(frames_split[i])
        assert np.array_equal(arr0, arr[-i - 1])


def test_video_frames_iterate(get_sample):
    """FFmpegVideoFrames can be iterated"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    arrs = [_to_numpy(f) for f in frames]
    array = _to_numpy(frames)

    for i in range(n):
        assert np.array_equal(array[i], arrs[i])


def test_video_frames_list_slice(get_sample):
    """FFmpegVideoFrames can be sliced with list of integers"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    # The valid value range is [-n, n)
    idx = [0, 99, 1, 3, -1, -100]

    sampled_frames = frames[idx]

    refs = _to_numpy(frames)
    array = _to_numpy(sampled_frames)

    for i in range(len(idx)):
        assert np.array_equal(array[i], refs[idx[i]])


def test_video_frames_list_slice_empty(get_sample):
    """FFmpegVideoFrames can be sliced with an empty list"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    # The valid value range is [-n, n)
    sampled_frames = frames[[]]

    assert len(sampled_frames) == 0


def test_video_frames_list_slice_out_of_range(get_sample):
    """Slicing FFmpegVideoFrames with an out-of-range value fails"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    n = 100
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_video(sample.path, pix_fmt="rgb24")

    assert len(frames) == n

    # The valid value range is [-n, n)
    with pytest.raises(IndexError):
        frames[[n]]

    # The valid value range is [-n, n)
    with pytest.raises(IndexError):
        frames[[-n - 1]]
