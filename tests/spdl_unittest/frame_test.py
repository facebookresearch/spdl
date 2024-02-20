import numpy as np
from spdl import libspdl


def _get_video_frames(pix_fmt, h=128, w=256):
    src = "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
    return libspdl.decode_video(
        src=src,
        timestamps=[(0.0, 0.5)],
        pix_fmt=pix_fmt,
        height=h,
        width=w,
    ).get()[0]


def test_video_frames_getitem_slice():
    """DecodedFrames.__getitem__ works for slice input"""
    decoded_frames = _get_video_frames("rgb24")
    num_frames = len(decoded_frames)
    arr = libspdl.to_numpy(decoded_frames)

    assert num_frames > 0

    f2 = decoded_frames[::2]
    arr2 = libspdl.to_numpy(f2)
    assert len(f2) == (num_frames + 1) // 2
    assert np.array_equal(arr[::2], arr2)

    f3 = decoded_frames[::3]
    arr3 = libspdl.to_numpy(f3)
    assert len(f3) == (num_frames + 2) // 3
    assert np.array_equal(arr[::3], arr3)


def test_video_frames_getitem_int():
    """DecodedFrames.__getitem__ works for index input"""
    decoded_frames = _get_video_frames("rgb24")
    num_frames = len(decoded_frames)
    arr = libspdl.to_numpy(decoded_frames)

    assert num_frames > 0
    for i in range(num_frames):
        arr0 = libspdl.to_numpy(decoded_frames[i])
        assert np.array_equal(arr0, arr[i])


def test_video_frames_iterate():
    """DecodedFrames works as iterator"""
    decoded_frames = _get_video_frames("rgb24")
    num_frames = len(decoded_frames)
    arr = libspdl.to_numpy(decoded_frames)

    assert num_frames > 0
    for i, decoded_frame in enumerate(decoded_frames):
        arr0 = libspdl.to_numpy(decoded_frame)
        assert np.array_equal(arr0, arr[i])
