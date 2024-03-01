import gc
import sys

import numpy as np
import pytest
import spdl
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

    buf = spdl._convert._to_cpu_buffer(decoded_frames, None)
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


def test_video_buffer_conversion_rgb24(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=rgb24 -frames:v 100 sample.mp4"
    h, w = 240, 320
    sample = get_sample(cmd, width=w, height=h)

    frames = libspdl.decode_video(
        src=sample.path,
        timestamps=[(0.0, 1.0)],
        pix_fmt="rgb24",
    ).get()[0]

    for index in [None, 0]:
        array = spdl.to_numpy(frames, index=index)
        assert array.shape[1:4] == (h, w, 3)

    # plane 1 & 2 are not defined
    for i in [1, 2]:
        with pytest.raises(RuntimeError):
            spdl.to_numpy(frames, index=i)


def test_video_buffer_conversion_yuv420(yuv420p):
    h, w = yuv420p.height, yuv420p.width
    h2, w2 = h // 2, w // 2

    frames = libspdl.decode_video(
        src=yuv420p.path,
        timestamps=[(0.0, 1.0)],
    ).get()[0]

    # combined
    array = spdl.to_numpy(frames)
    assert array.shape[1:4] == (1, h + h2, w)
    # individual - Y
    array = spdl.to_numpy(frames, index=0)
    assert array.shape[1:4] == (1, h, w)
    # individual - U, V
    for i in range(1, 3):
        array = spdl.to_numpy(frames, index=i)
        assert array.shape[1:4] == (1, h2, w2)


def test_video_buffer_conversion_yuv444(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 100 sample.mp4"
    h, w = 240, 320

    sample = get_sample(cmd, width=w, height=h)

    frames = libspdl.decode_video(
        src=sample.path,
        timestamps=[(0.0, 1.0)],
        pix_fmt="yuv444p",
    ).get()[0]

    # combined
    array = spdl.to_numpy(frames)
    assert array.shape[1:4] == (3, h, w)
    # individual
    for i in range(0, 3):
        array = spdl.to_numpy(frames, index=i)
        assert array.shape[1:4] == (1, h, w)


def test_video_buffer_conversion_nv12(get_sample):
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
    array = spdl.to_numpy(frames)
    assert array.shape[1:4] == (1, h + h2, w)
    # individual - Y
    array = spdl.to_numpy(frames, index=0)
    assert array.shape[1:4] == (1, h, w)
    # individual - UV
    array = spdl.to_numpy(frames, index=1)
    assert array.shape[1:4] == (h2, w2, 2)


@pytest.mark.parametrize(
    "sample_fmts",
    [("s16p", "int16"), ("s16", "int16"), ("fltp", "float32"), ("flt", "float32")],
)
def test_audio_buffer_conversion_s16p(sample_fmts, get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=305:duration=5' -f lavfi -i 'sine=frequency=300:duration=5'  -filter_complex amerge  -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    sample_fmt, expected = sample_fmts
    frames = libspdl.decode_audio(
        src=sample.path,
        timestamps=[(0.0, 0.5)],
        sample_fmt=sample_fmt,
    ).get()[0]

    array = spdl.to_numpy(frames)
    assert array.ndim == 2
    assert array.dtype == np.dtype(expected)
    if sample_fmt.endswith("p"):
        assert array.shape[0] == 2
    else:
        assert array.shape[1] == 2
