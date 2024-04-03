import numpy as np
import pytest

import spdl.io
from spdl.lib import _libspdl


def _to_numpy(frames, index=None):
    return spdl.io.to_numpy(_libspdl.convert_to_cpu_buffer(frames, index))


@pytest.fixture
def yuv420p(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    return get_sample(cmd, width=320, height=240)


def test_video_frames_getitem_slice(yuv420p):
    """DecodedFrames.__getitem__ works for slice input"""
    decoded_frames = _libspdl.decode_video(
        src=yuv420p.path,
        timestamps=[(0.0, 0.5)],
        pix_fmt="rgb24",
    ).get()[0]

    num_frames = len(decoded_frames)
    arr = _to_numpy(decoded_frames)

    assert num_frames > 0

    f2 = decoded_frames[::2]
    arr2 = _to_numpy(f2)
    assert len(f2) == (num_frames + 1) // 2
    assert np.array_equal(arr[::2], arr2)

    f3 = decoded_frames[::3]
    arr3 = _to_numpy(f3)
    assert len(f3) == (num_frames + 2) // 3
    assert np.array_equal(arr[::3], arr3)


def test_video_frames_getitem_int(yuv420p):
    """DecodedFrames.__getitem__ works for index input"""
    decoded_frames = _libspdl.decode_video(
        src=yuv420p.path,
        timestamps=[(0.0, 0.5)],
        pix_fmt="rgb24",
    ).get()[0]

    num_frames = len(decoded_frames)
    arr = _to_numpy(decoded_frames)

    assert num_frames > 0
    for i in range(num_frames):
        arr0 = _to_numpy(decoded_frames[i])
        assert np.array_equal(arr0, arr[i])


def test_video_frames_iterate(yuv420p):
    """DecodedFrames works as iterator"""
    decoded_frames = _libspdl.decode_video(
        src=yuv420p.path,
        timestamps=[(0.0, 0.5)],
        pix_fmt="rgb24",
    ).get()[0]

    num_frames = len(decoded_frames)
    arr = _to_numpy(decoded_frames)

    assert num_frames > 0
    for i, decoded_frame in enumerate(decoded_frames):
        arr0 = _to_numpy(decoded_frame)
        assert np.array_equal(arr0, arr[i])
