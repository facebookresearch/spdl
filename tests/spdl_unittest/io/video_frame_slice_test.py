import numpy as np

import spdl.io
import spdl.utils
from spdl.io.preprocessing import get_video_filter_desc


def _to_numpy(frames):
    return spdl.io.to_numpy(spdl.io.convert_frames(frames).result())


def _decode_video(src, pix_fmt=None):
    packets = spdl.io.demux_media("video", src).result()
    return spdl.io.decode_packets(
        packets, filter_desc=get_video_filter_desc(pix_fmt=pix_fmt)
    ).result()


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
