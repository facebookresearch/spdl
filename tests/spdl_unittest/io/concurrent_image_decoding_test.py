# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numpy as np
import spdl.io
import spdl.io.utils
from spdl.io import get_filter_desc, get_video_filter_desc

from ..fixture import get_sample, get_samples


def _decode_image(src, pix_fmt=None):
    buffer = spdl.io.load_image(
        src,
        filter_desc=get_video_filter_desc(pix_fmt=pix_fmt),
    )
    return spdl.io.to_numpy(buffer)


def test_decode_image_gray16_black():
    """16-bit PNG image (gray16be) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray16be -frames:v 1 sample.png"
    sample = get_sample(cmd)

    gray = _decode_image(sample.path, pix_fmt="gray16")
    assert gray.dtype == np.uint16
    assert gray.shape == (1, 240, 320)
    assert np.all(gray == 1)

    gray = _decode_image(sample.path, pix_fmt="rgb24")
    assert gray.dtype == np.uint8
    assert gray.shape == (240, 320, 3)
    assert np.all(gray == 0)


def test_decode_image_gray16_white():
    """16-bit PNG image (gray16be) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0xFFFFFF,format=gray16be -frames:v 1 sample.png"
    sample = get_sample(cmd)

    gray = _decode_image(sample.path, pix_fmt="gray16")
    assert gray.dtype == np.uint16
    assert gray.shape == (1, 240, 320)
    assert np.all(gray == 65277)

    gray = _decode_image(sample.path, pix_fmt="rgb24")
    assert gray.dtype == np.uint8
    assert gray.shape == (240, 320, 3)
    assert np.all(gray == 255)


def test_decode_image_yuv422():
    """JPEG image (yuvj422p) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd)

    yuv = _decode_image(sample.path, pix_fmt=None)

    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, 2 * 240, 320)


def test_decode_image_yuv420p():
    """JPEG image can be converted to yuv420"""
    # This is yuvj420p format
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpeg"
    w, h = 320, 240
    sample = get_sample(cmd)

    yuv = _decode_image(sample.path, pix_fmt="yuv420p")

    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, h + h // 2, w)


def test_decode_image_rgb24():
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
    rgb = get_sample(cmd)

    array = _decode_image(rgb.path, pix_fmt="rgb24")

    assert array.dtype == np.uint8
    assert array.shape == (height, 3*width, 3)

    red = array[:, :width]

    # Some channels,
    # FFmpeg 4.1 returns 0 while FFmpeg 6.0 returns 1
    # FFmpeg 4.1 returns 255 while FFmpeg 6.0 returns 254

    assert np.all(red[..., 0] >= 254)
    assert np.all(red[..., 1] <= 1)
    assert np.all(red[..., 2] == 0)

    green = array[:, width:2*width]
    assert np.all(green[..., 0] == 0)
    assert np.all(green[..., 1] >= 253)
    assert np.all(green[..., 2] <= 1)

    blue = array[:, 2*width:]
    assert np.all(blue[..., 0] <= 1)
    assert np.all(blue[..., 1] <= 1)
    assert np.all(blue[..., 2] >= 254)


def test_decode_image_16be():
    """PNG image (16be) can be decoded and converted to RGB"""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                                 \
        -f lavfi -i color=white:size=32x32,format=gray16be \
        -f lavfi -i color=black:size=32x32,format=gray16be \
        -filter_complex hstack=inputs=2                    \
        -frames:v 1 -pix_fmt gray16 sample_%03d.png
    """
    # fmt: on
    height, width = 32, 64
    sample = get_sample(cmd)

    array = _decode_image(sample.path, pix_fmt="gray16")
    assert array.dtype == np.uint16
    assert array.shape == (1, height, width)
    assert np.all(array[..., :32] == 65277)
    assert np.all(array[..., 32:] == 1)

    array = _decode_image(sample.path, pix_fmt="rgb24")
    assert array.dtype == np.uint8
    assert array.shape == (height, width, 3)
    assert np.all(array[:, :32, :] == 255)
    assert np.all(array[:, 32:, :] == 0)


def test_batch_decode_image_slice():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 32 sample_%03d.png"
    n, h, w = 32, 240, 320
    srcs = get_samples(cmd)

    buffer = spdl.io.load_image_batch(
        [src.path for src in srcs], width=None, height=None, pix_fmt="rgb24"
    )
    array = spdl.io.to_numpy(buffer)
    print(array.shape)
    assert array.shape == (n, h, w, 3)

    for i in range(n):
        arr = _decode_image(srcs[i].path, pix_fmt="rgb24")
        print(arr.shape)
        assert np.all(arr == array[i])


def test_batch_decode_image_rgb24():
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
    srcs = get_samples(cmd)

    buffer = spdl.io.load_image_batch(
        [src.path for src in srcs], width=None, height=None, pix_fmt="rgb24"
    )
    arrays = spdl.io.to_numpy(buffer)
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
        assert np.all(array[:, w : 2 * w, 1] >= 253)
        assert np.all(array[:, w : 2 * w, 2] == 0)

        # Blue
        assert np.all(array[:, 2 * w :, 0] == 0)
        assert np.all(array[:, 2 * w :, 1] == 0)
        assert np.all(array[:, 2 * w :, 2] >= 253)


# This test requires "tpad filter (FFmepg >= 4.2.")
if "tpad" in spdl.io.utils.get_ffmpeg_filters():

    def test_batch_video_conversion():
        """Can decode video clips."""
        cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
        sample = get_sample(cmd)

        timestamps = [(0, 1), (1, 1.5), (2, 2.7), (3, 3.6)]

        demuxer = spdl.io.Demuxer(sample.path)
        frames = []
        for ts in timestamps:
            packets = demuxer.demux_video(timestamp=ts)
            filter_desc = get_filter_desc(packets, num_frames=15)
            frames_ = spdl.io.decode_packets(packets, filter_desc=filter_desc)
            frames.append(frames_)

        buffer = spdl.io.convert_frames(frames)
        array = spdl.io.to_numpy(buffer)
        assert array.shape == (4, 15, 3, 240, 320)
