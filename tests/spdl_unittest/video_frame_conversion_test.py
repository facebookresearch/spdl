import gc
import sys

import pytest
from spdl import libspdl


def _get_video_frames(pix_fmt, h=128, w=256):
    src = "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
    return libspdl.decode_video(
        src=src,
        timestamps=[(0.0, 0.5)],
        pix_fmt=pix_fmt,
        height=h,
        width=w,
    )[0]


def test_video_buffer_conversion_refcount(pix_fmt="yuv420p"):
    """NumPy array created from VideoBuffer should increment a reference to the buffer
    so that array keeps working after the original VideoBuffer variable is deleted.
    """
    import numpy as np
    from spdl.libspdl import _BufferWrapper

    buf = _BufferWrapper(_get_video_frames(pix_fmt).to_buffer(None))
    assert hasattr(buf, "__array_interface__")
    print(f"{buf.__array_interface__=}")

    gc.collect()

    n = sys.getrefcount(buf)
    assert n == 2

    arr = np.array(buf, copy=False)

    n1 = sys.getrefcount(buf)
    assert n1 == n + 1

    print(f"{arr.__array_interface__=}")
    assert arr.__array_interface__ is not buf.__array_interface__
    assert arr.__array_interface__ == buf.__array_interface__

    vals = arr.tolist()

    # Not sure if this will properly fail in case that NumPy array does not
    # keep the reference to the Buffer object. But let's do it anyways
    del buf
    gc.collect()

    vals2 = arr.tolist()
    assert vals == vals2


@pytest.mark.parametrize("format", ["channel_first", "channel_last"])
def test_video_buffer_conversion_rgb24(format, pix_fmt="rgb24"):
    h, w = 128, 256

    frames = _get_video_frames(pix_fmt, h, w)

    for index in [None, 0]:
        array = libspdl.to_numpy(frames, index=index, format=format)
        expected_shape = (3, h, w) if format == "channel_first" else (h, w, 3)
        assert array.shape[1:4] == expected_shape

    # plane 1 & 2 are not defined
    for i in [1, 2]:
        with pytest.raises(RuntimeError):
            libspdl.to_numpy(frames, index=i, format=format)


@pytest.mark.parametrize("format", ["channel_first", "channel_last"])
def test_video_buffer_conversion_yuv420(format, pix_fmt="yuv420p"):
    h, w = 128, 256
    h2, w2 = h // 2, w // 2

    frames = _get_video_frames(pix_fmt, h, w)

    # combined
    array = libspdl.to_numpy(frames, format=format)
    expected_shape = (1, h + h2, w) if format == "channel_first" else (h + h2, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - Y
    array = libspdl.to_numpy(frames, index=0, format=format)
    expected_shape = (1, h, w) if format == "channel_first" else (h, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - U, V
    for i in range(1, 3):
        array = libspdl.to_numpy(frames, index=i, format=format)
        expected_shape = (1, h2, w2) if format == "channel_first" else (h2, w2, 1)
        assert array.shape[1:4] == expected_shape


@pytest.mark.parametrize("format", ["channel_first", "channel_last"])
def test_video_buffer_conversion_yuv444(format, pix_fmt="yuv444p"):
    h, w = 128, 256

    frames = _get_video_frames(pix_fmt, h, w)

    # combined
    array = libspdl.to_numpy(frames, format=format)
    expected_shape = (3, h, w) if format == "channel_first" else (h, w, 3)
    assert array.shape[1:4] == expected_shape
    # individual
    for i in range(0, 3):
        array = libspdl.to_numpy(frames, index=i, format=format)
        expected_shape = (1, h, w) if format == "channel_first" else (h, w, 1)
        assert array.shape[1:4] == expected_shape


@pytest.mark.parametrize("format", ["channel_first", "channel_last"])
def test_video_buffer_conversion_nv12(format, pix_fmt="nv12"):
    h, w = 128, 256
    h2, w2 = h // 2, w // 2

    frames = _get_video_frames(pix_fmt, h, w)

    # combined
    array = libspdl.to_numpy(frames, format=format)
    expected_shape = (1, h + h2, w) if format == "channel_first" else (h + h2, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - Y
    array = libspdl.to_numpy(frames, index=0, format=format)
    expected_shape = (1, h, w) if format == "channel_first" else (h, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - UV
    array = libspdl.to_numpy(frames, index=1, format=format)
    expected_shape = (2, h2, w2) if format == "channel_first" else (h2, w2, 2)
    assert array.shape[1:4] == expected_shape
