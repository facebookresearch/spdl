import gc
import itertools
import sys

import numpy as np
import pytest
from spdl import libspdl


@pytest.fixture
def yuv420p(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    return get_sample(cmd, width=320, height=240)


def test_video_buffer_conversion_refcount(yuv420p):
    """NumPy array created from Buffer should increment a reference to the buffer
    so that array keeps working after the original Buffer variable is deleted.
    """
    decoded_frames = libspdl.decode_video(
        src=yuv420p.path,
        timestamps=[(0.0, 0.5)],
        pix_fmt="rgb24",
    ).get()[0]

    buf = libspdl._to_buffer(decoded_frames, None)
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
def test_video_buffer_conversion_rgb24(format, get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=rgb24 -frames:v 100 sample.mp4"
    h, w = 240, 320
    sample = get_sample(cmd, width=w, height=h)

    frames = libspdl.decode_video(
        src=sample.path,
        timestamps=[(0.0, 0.5)],
        pix_fmt="rgb24",
    ).get()[0]

    for index in [None, 0]:
        array = libspdl.to_numpy(frames, index=index, format=format)
        expected_shape = (3, h, w) if format == "channel_first" else (h, w, 3)
        assert array.shape[1:4] == expected_shape

    # plane 1 & 2 are not defined
    for i in [1, 2]:
        with pytest.raises(RuntimeError):
            libspdl.to_numpy(frames, index=i, format=format)


@pytest.mark.parametrize("format", ["channel_first", "channel_last"])
def test_video_buffer_conversion_yuv420(format, yuv420p):
    h, w = yuv420p.height, yuv420p.width
    h2, w2 = h // 2, w // 2

    frames = libspdl.decode_video(
        src=yuv420p.path,
        timestamps=[(0.0, 0.5)],
    ).get()[0]

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
def test_video_buffer_conversion_yuv444(format, get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 100 sample.mp4"
    h, w = 240, 320

    sample = get_sample(cmd, width=w, height=h)

    frames = libspdl.decode_video(
        src=sample.path,
        timestamps=[(0.0, 0.5)],
        pix_fmt="yuv444p",
    ).get()[0]

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
def test_video_buffer_conversion_nv12(format, get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 100 sample.mp4"
    h, w = 240, 320
    h2, w2 = h // 2, w // 2

    sample = get_sample(cmd, width=w, height=h)
    frames = libspdl.decode_video(
        src=sample.path,
        timestamps=[(0.0, 0.5)],
        pix_fmt="nv12",
    ).get()[0]

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


@pytest.mark.parametrize(
    "format,sample_fmts",
    itertools.product(
        ["channel_first", "channel_last"],
        [("s16p", "int16"), ("s16", "int16"), ("fltp", "float32"), ("flt", "float32")],
    ),
)
def test_audio_buffer_conversion_s16p(format, sample_fmts, get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=305:duration=5' -f lavfi -i 'sine=frequency=300:duration=5'  -filter_complex amerge  -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd, num_channels=2)

    sample_fmt, expected = sample_fmts
    frames = libspdl.decode_audio(
        src=sample.path,
        timestamps=[(0.0, 0.5)],
        sample_fmt=sample_fmt,
    ).get()[0]

    array = libspdl.to_numpy(frames, format=format)
    assert array.ndim == 2
    assert array.dtype == np.dtype(expected)
    if format == "channel_first":
        assert array.shape[0] == sample.num_channels
    if format == "channel_last":
        assert array.shape[1] == sample.num_channels
