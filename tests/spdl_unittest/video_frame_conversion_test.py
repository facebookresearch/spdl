import gc
import sys

import pytest
from spdl import libspdl

from spdl_unittest import helpers


def _get_video_frames(pix_fmt, h=128, w=256):
    src = helpers.get_src_video()
    return helpers.get_video_frames(src=src, pix_fmt=pix_fmt, height=h, width=w)


def test_video_buffer_conversion_refcount(pix_fmt="yuv420p"):
    """NumPy array created from VideoBuffer should increment a reference to the buffer
    so that array keeps working after the original VideoBuffer variable is deleted.
    """
    buf = _get_video_frames(pix_fmt).to_video_buffer()
    assert hasattr(buf, "__array_interface__")
    print(f"{buf.__array_interface__=}")

    gc.collect()

    n = sys.getrefcount(buf)
    assert n == 2

    arr = libspdl.to_numpy(buf, format=None)

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


@pytest.mark.parametrize("format", ["NCHW", "NHWC"])
def test_video_buffer_conversion_rgb24(format, pix_fmt="rgb24"):
    h, w = 128, 256

    frames = _get_video_frames(pix_fmt, h, w)

    # combined (rgb24 is interweived, so extracting the first plane (i==0)
    # should return the same result.)
    for i in [-1, 0]:
        array = libspdl.to_numpy(frames.to_video_buffer(i), format=format)
        expected_shape = (3, h, w) if format == "NCHW" else (h, w, 3)
        assert array.shape[1:4] == expected_shape

    # plane 1 & 2 are not defined
    for i in [1, 2]:
        with pytest.raises(RuntimeError):
            libspdl.to_numpy(frames.to_video_buffer(i), format=format)


@pytest.mark.parametrize("format", ["NCHW", "NHWC"])
def test_video_buffer_conversion_yuv420(format, pix_fmt="yuv420p"):
    h, w = 128, 256
    h2, w2 = h // 2, w // 2

    frames = _get_video_frames(pix_fmt, h, w)

    # combined
    array = libspdl.to_numpy(frames.to_video_buffer(), format=format)
    expected_shape = (1, h + h2, w) if format == "NCHW" else (h + h2, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - Y
    array = libspdl.to_numpy(frames.to_video_buffer(0), format=format)
    expected_shape = (1, h, w) if format == "NCHW" else (h, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - U, V
    for i in range(1, 3):
        array = libspdl.to_numpy(frames.to_video_buffer(i), format=format)
        expected_shape = (1, h2, w2) if format == "NCHW" else (h2, w2, 1)
        assert array.shape[1:4] == expected_shape


@pytest.mark.parametrize("format", ["NCHW", "NHWC"])
def test_video_buffer_conversion_yuv444(format, pix_fmt="yuv444p"):
    h, w = 128, 256

    frames = _get_video_frames(pix_fmt, h, w)

    # combined
    array = libspdl.to_numpy(frames.to_video_buffer(), format=format)
    expected_shape = (3, h, w) if format == "NCHW" else (h, w, 3)
    assert array.shape[1:4] == expected_shape
    # individual
    for i in range(0, 3):
        array = libspdl.to_numpy(frames.to_video_buffer(i), format=format)
        expected_shape = (1, h, w) if format == "NCHW" else (h, w, 1)
        assert array.shape[1:4] == expected_shape


@pytest.mark.parametrize("format", ["NCHW", "NHWC"])
def test_video_buffer_conversion_nv12(format, pix_fmt="nv12"):
    h, w = 128, 256
    h2, w2 = h // 2, w // 2

    frames = _get_video_frames(pix_fmt, h, w)

    # combined
    array = libspdl.to_numpy(frames.to_video_buffer(), format=format)
    expected_shape = (1, h + h2, w) if format == "NCHW" else (h + h2, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - Y
    array = libspdl.to_numpy(frames.to_video_buffer(0), format=format)
    expected_shape = (1, h, w) if format == "NCHW" else (h, w, 1)
    assert array.shape[1:4] == expected_shape
    # individual - UV
    array = libspdl.to_numpy(frames.to_video_buffer(1), format=format)
    expected_shape = (2, h2, w2) if format == "NCHW" else (h2, w2, 2)
    assert array.shape[1:4] == expected_shape
