import asyncio

import numpy as np

import spdl.io
import spdl.utils
from spdl.io import get_filter_desc, get_video_filter_desc


def _decode_image(src, pix_fmt=None):
    buffer = asyncio.run(
        spdl.io.async_load_image(
            src,
            decode_options={"filter_desc": get_video_filter_desc(pix_fmt=pix_fmt)},
        )
    )
    return spdl.io.to_numpy(buffer)


def _batch_load_image(srcs, pix_fmt="rgb24"):
    buffer = asyncio.run(
        spdl.io.async_load_image_batch(srcs, width=None, height=None, pix_fmt=pix_fmt)
    )
    return spdl.io.to_numpy(buffer)


def test_decode_image_gray_black(get_sample):
    """PNG image (gray) can be decoded."""
    # Note: the output of this command is in limited color range. So [0, 255] -> [16, 235]
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png"
    sample = get_sample(cmd, width=320, height=240)

    gray = _decode_image(sample.path)

    assert gray.dtype == np.uint8
    assert gray.shape == (1, sample.height, sample.width)
    assert np.all(gray == 16)


def test_decode_image_gray_white(get_sample):
    """PNG image (gray) can be decoded."""
    # Note: the output of this command is in limited color range. So [0, 255] -> [16, 235]
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0xFFFFFF,format=gray -frames:v 1 sample.png"
    sample = get_sample(cmd, width=320, height=240)

    gray = _decode_image(sample.path)

    assert gray.dtype == np.uint8
    assert gray.shape == (1, sample.height, sample.width)
    assert np.all(gray == 235)


def test_decode_image_yuv422(get_sample):
    """JPEG image (yuvj422p) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd, width=320, height=240)

    yuv = _decode_image(sample.path)

    h, w = sample.height, sample.width

    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, h + h, w)


def test_decode_image_yuv420p(get_sample):
    """JPEG image can be converted to yuv420"""
    # This is yuvj420p format
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpeg"
    w, h = 320, 240
    h2 = h // 2

    sample = get_sample(cmd, width=w, height=h)

    yuv = _decode_image(sample.path, pix_fmt="yuv420p")

    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, h + h2, w)


def test_decode_image_rgb24(get_sample):
    """JPEG image (yuvj420p) can be decoded and converted to RGB"""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                          \
        -f lavfi -i color=color=0xff0000:size=32x64 \
        -f lavfi -i color=color=0x00ff00:size=32x64 \
        -f lavfi -i color=color=0x0000ff:size=32x64 \
        -filter_complex hstack=inputs=3             \
        -frames:v 1 sample_%03d.jpeg
    """
    height, width = 64, 32
    rgb = get_sample(cmd, width=3*width, height=height)

    array = _decode_image(rgb.path, pix_fmt="rgb24")

    assert array.dtype == np.uint8
    assert array.shape == (rgb.height, rgb.width, 3)

    red = array[:, :width]

    # Some channels,
    # FFmpeg 4.1 returns 0 while FFmpeg 6.0 returns 1
    # FFmpeg 4.1 returns 255 while FFmpeg 6.0 returns 254

    assert np.all(red[..., 0] >= 254)
    assert np.all(red[..., 1] <= 1)
    assert np.all(red[..., 2] == 0)

    green = array[:, width:2*width]
    assert np.all(green[..., 0] == 0)
    assert np.all(green[..., 1] >= 254)
    assert np.all(green[..., 2] <= 1)

    blue = array[:, 2*width:]
    assert np.all(blue[..., 0] <= 1)
    assert np.all(blue[..., 1] <= 1)
    assert np.all(blue[..., 2] >= 254)


def test_batch_decode_image_slice(get_samples):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 32 sample_%03d.png"
    n, h, w = 32, 240, 320
    flist = get_samples(cmd)

    array = _batch_load_image(flist, pix_fmt="rgb24")
    print(array.shape)
    assert array.shape == (n, h, w, 3)

    for i in range(n):
        arr = _decode_image(flist[i], pix_fmt="rgb24")
        print(arr.shape)
        assert np.all(arr == array[i])


def test_batch_decode_image_rgb24(get_samples):
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                          \
        -f lavfi -i color=color=0xff0000:size=32x64 \
        -f lavfi -i color=color=0x00ff00:size=32x64 \
        -f lavfi -i color=color=0x0000ff:size=32x64 \
        -filter_complex hstack=inputs=3             \
        -frames:v 32 sample_%03d.png
    """
    h, w = 64, 32
    # fmt: on
    flist = get_samples(cmd)

    arrays = _batch_load_image(flist, pix_fmt="rgb24")
    assert arrays.shape == (32, h, 3 * w, 3)

    for i in range(32):
        array = arrays[i]

        assert array.dtype == np.uint8
        assert array.shape == (h, 3 * w, 3)

        # Red
        assert np.all(array[:, :w, 0] >= 252)
        assert np.all(array[:, :w, 1] == 0)
        assert np.all(array[:, :w, 2] == 0)

        # Green
        assert np.all(array[:, w : 2 * w, 0] == 0)
        assert np.all(array[:, w : 2 * w, 1] == 254)
        assert np.all(array[:, w : 2 * w, 2] == 0)

        # Blue
        assert np.all(array[:, 2 * w :, 0] == 0)
        assert np.all(array[:, 2 * w :, 1] == 0)
        assert np.all(array[:, 2 * w :, 2] >= 253)


def test_batch_video_conversion(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(0, 1), (1, 1.5), (2, 2.7), (3, 3.6)]

    async def _test():
        decoding = []
        async for packets in spdl.io.async_streaming_demux_video(
            src=sample.path, timestamps=timestamps
        ):
            filter_desc = get_filter_desc(packets, num_frames=15)
            coro = spdl.io.async_decode_packets(packets, filter_desc=filter_desc)
            decoding.append(asyncio.create_task(coro))

        frames = await asyncio.gather(*decoding)

        buffer = await spdl.io.async_convert_frames(frames)
        return spdl.io.to_numpy(buffer)

    array = asyncio.run(_test())

    assert array.shape == (4, 15, 3, 240, 320)
