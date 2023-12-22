import pytest
from spdl import libspdl


def _get_src_video():
    return "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"


def _get_video_frames(pix_fmt, height, width):
    engine = libspdl.Engine(10)
    engine.enqueue(
        src=_get_src_video(),
        timestamps=[0.0],
        pix_fmt=pix_fmt,
        height=height,
        width=width,
    )
    frames = engine.dequeue()
    return frames


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
