# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numpy as np
import pytest
import spdl.io
import spdl.io.utils
from spdl.io import get_video_filter_desc
from spdl.io.utils import get_ffmpeg_versions

from ..fixture import get_sample, get_samples, load_ref_data, load_ref_image


def _load_image(src, filter_graph="format=pix_fmts=rgb24"):
    return spdl.io.to_numpy(spdl.io.load_image(src, filter_desc=filter_graph))


def test_decode_image_gray16_native():
    """Can decode gray16 PNG image (16be) as-is"""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                                 \
        -f lavfi -i color=0xFFFFFF:size=32x32,format=gray16be \
        -f lavfi -i color=0x000000:size=32x32,format=gray16be \
        -filter_complex hstack=inputs=2                    \
        -frames:v 1 -pix_fmt gray16 sample_%03d.png
    """
    # fmt: on
    sample = get_sample(cmd)

    height, width = 32, 64
    shape = (1, height, width)

    hyp = _load_image(sample.path, filter_graph=None)
    ref = load_ref_image(sample.path, shape, dtype=np.uint16, filter_graph=None)
    np.testing.assert_array_equal(hyp, ref, strict=True)

    assert np.all(hyp[..., :32] == 65022)
    assert np.all(hyp[..., 32:] == 256)


def test_decode_image_16be_rgb24():
    """Can decode gray16 PNG image (16be) as rgb24"""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                                 \
        -f lavfi -i color=white:size=32x32,format=gray16be \
        -f lavfi -i color=black:size=32x32,format=gray16be \
        -filter_complex hstack=inputs=2                    \
        -frames:v 1 -pix_fmt gray16 sample_%03d.png
    """
    # fmt: on
    sample = get_sample(cmd)

    height, width = 32, 64
    shape = (height, width, 3)

    hyp = _load_image(sample.path)
    ref = load_ref_image(sample.path, shape)
    np.testing.assert_array_equal(hyp, ref, strict=True)

    assert np.all(hyp[:, :32, :] == 255)
    assert np.all(hyp[:, 32:, :] == 0)


def test_decode_image_yuvj422_native():
    """Can decode yuvj422p JPEG image as-is."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd)

    shape = (1, 2 * 240, 320)
    hyp = _load_image(sample.path, filter_graph=None)
    ref = load_ref_image(sample.path, shape, filter_graph=None)

    # from PIL import Image
    # Image.fromarray(hyp[0]).save("hyp.png")
    # Image.fromarray(ref[0]).save("ref.png")
    # Image.fromarray(hyp[0]-ref[0]).save("diff.png")
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_decode_image_yuvj422_as_rgb24():
    """Can decode yuvj422p JPEG image as rgb24."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd)

    shape = (240, 320, 3)
    hyp = _load_image(sample.path)
    ref = load_ref_image(sample.path, shape)

    # from PIL import Image
    # Image.fromarray(hyp[0]).save("hyp.png")
    # Image.fromarray(ref[0]).save("ref.png")
    # Image.fromarray(hyp[0]-ref[0]).save("diff.png")
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_decode_image_yuvj420p_native():
    """Can decode yuvj420p JPEG image as yuv420p"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 -pix_fmt yuvj420p sample.jpeg"
    sample = get_sample(cmd)

    w, h = 320, 240
    shape = (1, h + h // 2, w)

    hyp = _load_image(sample.path, filter_graph=None)
    ref = load_ref_image(sample.path, shape, filter_graph=None)

    # from PIL import Image
    # Image.fromarray(hyp[0]).save("hyp.png")
    # Image.fromarray(ref[0]).save("ref.png")
    # Image.fromarray(hyp[0]-ref[0]).save("diff.png")
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_decode_image_yuvj420p_as_rgb24():
    """Can decode yuvj420p JPEG image as rgb24"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 -pix_fmt yuvj420p sample.jpeg"
    sample = get_sample(cmd)

    shape = (240, 320, 3)
    hyp = _load_image(sample.path)
    ref = load_ref_image(sample.path, shape)
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_decode_image_yuvj420p_as_rgb24_edge_values():
    """Can decode yuvj420p JPEG image as rgb24 at edge values"""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                          \
        -f lavfi -i color=color=0xff0000:size=32x64 \
        -f lavfi -i color=color=0x00ff00:size=32x64 \
        -f lavfi -i color=color=0x0000ff:size=32x64 \
        -filter_complex hstack=inputs=3             \
        -frames:v 1 sample_%03d.jpeg
    """
    sample = get_sample(cmd)

    height, width = 64, 32
    shape = (height, 3*width, 3)
    hyp = _load_image(sample.path)
    ref = load_ref_image(sample.path, shape)
    np.testing.assert_array_equal(hyp, ref, strict=True)

    # Extra check
    # Note:
    # FFmpeg 4.1 returns 0 while FFmpeg 6.0 returns 1
    # FFmpeg 4.1 returns 255 while FFmpeg 6.0 returns 254

    red, green, blue = hyp[:, :width], hyp[:, width:2*width], hyp[:, 2*width:]

    assert np.all(red[..., 0] >= 254)
    assert np.all(red[..., 1] <= 1)
    assert np.all(red[..., 2] == 0)

    assert np.all(green[..., 0] == 0)
    assert np.all(green[..., 1] >= 253)
    assert np.all(green[..., 2] <= 1)

    assert np.all(blue[..., 0] <= 1)
    assert np.all(blue[..., 1] <= 1)
    assert np.all(blue[..., 2] >= 254)


def test_decode_image_yuvj444p_native():
    """Can decode yuvj444p JPEG image as-is."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj444p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd)

    shape = (3, 240, 320)
    hyp = _load_image(sample.path, filter_graph=None)
    ref = load_ref_image(sample.path, shape, filter_graph=None)

    # from PIL import Image
    # Image.fromarray(hyp[0]).save("hyp.png")
    # Image.fromarray(ref[0]).save("ref.png")
    # Image.fromarray(hyp[0]-ref[0]).save("diff.png")
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_decode_image_yuvj444p_as_rgb24():
    """Can decode yuvj444p JPEG image as rgb24."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj444p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd)

    shape = (240, 320, 3)
    hyp = _load_image(sample.path)
    ref = load_ref_image(sample.path, shape)

    # from PIL import Image
    # Image.fromarray(hyp[0]).save("hyp.png")
    # Image.fromarray(ref[0]).save("ref.png")
    # Image.fromarray(hyp[0]-ref[0]).save("diff.png")
    np.testing.assert_array_equal(hyp, ref, strict=True)


def test_decode_image_nv12_native():
    """Can decode nv12 image as-is"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 -pix_fmt nv12 -f rawvideo sample.nv12"
    w, h = 320, 240
    sample = get_sample(cmd)

    hyp = spdl.io.to_numpy(
        spdl.io.load_image(
            sample.path,
            filter_desc=get_video_filter_desc(pix_fmt=None),
            demux_config=spdl.io.demux_config(
                format="rawvideo",
                format_options={
                    "pixel_format": "nv12",
                    "video_size": "320x240",
                },
            ),
        )
    )

    shape = (1, h + h // 2, w)
    # fmt: off
    cmd = [
        "ffmpeg", "-hide_banner",
        "-f", "rawvideo",
        "-pixel_format", "nv12",
        "-video_size", "320x240",
        "-i", sample.path,
        "-f", "rawvideo", "pipe:"
    ]
    # fmt: on
    ref = load_ref_data(cmd, shape)

    # from PIL import Image
    # Image.fromarray(hyp[0]).save("hyp.png")
    # Image.fromarray(ref[0]).save("ref.png")
    # Image.fromarray(hyp[0]-ref[0]).save("diff.png")
    np.testing.assert_array_equal(hyp, ref, strict=True)


# This test fails with FFmpeg 4.4.2
if get_ffmpeg_versions()["libavutil"] >= 57:

    def test_decode_image_nv12_as_rgb24():
        """Can decode nv12 image as rgb24"""
        cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 -pix_fmt nv12 -f rawvideo sample.nv12"
        w, h = 320, 240
        sample = get_sample(cmd)

        hyp = spdl.io.to_numpy(
            spdl.io.load_image(
                sample.path,
                filter_desc=get_video_filter_desc(pix_fmt="rgb24"),
                demux_config=spdl.io.demux_config(
                    format="rawvideo",
                    format_options={
                        "pixel_format": "nv12",
                        "video_size": "320x240",
                    },
                ),
            )
        )

        shape = (h, w, 3)
        # fmt: off
        cmd = [
            "ffmpeg", "-hide_banner",
            "-f", "rawvideo",
            "-pixel_format", "nv12",
            "-video_size", "320x240",
            "-i", sample.path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:"
        ]
        # fmt: on
        ref = load_ref_data(cmd, shape)

        # from PIL import Image
        # Image.fromarray(hyp[0]).save("hyp.png")
        # Image.fromarray(ref[0]).save("ref.png")
        # Image.fromarray(hyp[0]-ref[0]).save("diff.png")
        np.testing.assert_array_equal(hyp, ref, strict=True)


################################################################################
# Batch decoding
################################################################################


def test_load_image_batch_native():
    """`load_image_batch` can decode series of images."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 32 sample_%03d.png"
    srcs = get_samples(cmd)

    n, h, w = 32, 240, 320
    buffer = spdl.io.load_image_batch(
        [src.path for src in srcs], width=None, height=None, pix_fmt=None
    )
    array = spdl.io.to_numpy(buffer)
    print(array.shape)
    assert array.shape == (n, h, w, 3)

    filter_graph = get_video_filter_desc(pix_fmt=None)
    for i in range(n):
        ref = load_ref_image(srcs[i].path, (h, w, 3), filter_graph=filter_graph)
        hyp = array[i]
        np.testing.assert_array_equal(hyp, ref, strict=True)


def test_load_image_batch_native_edge_values():
    """`load_image_batch` can decode series of images."""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                          \
        -f lavfi -i color=color=0xff0000:size=32x64 \
        -f lavfi -i color=color=0x00ff00:size=32x64 \
        -f lavfi -i color=color=0x0000ff:size=32x64 \
        -filter_complex hstack=inputs=3             \
        -frames:v 25 sample_%03d.png
    """
    # fmt: on
    srcs = get_samples(cmd)

    n, h, w = 25, 64, 32
    buffer = spdl.io.load_image_batch(
        [src.path for src in srcs], width=None, height=None, pix_fmt=None
    )
    arr = spdl.io.to_numpy(buffer)
    assert arr.shape == (n, h, 3 * w, 3)

    filter_graph = get_video_filter_desc(pix_fmt=None)
    for i in range(n):
        ref = load_ref_image(srcs[i].path, (h, w * 3, 3), filter_graph=filter_graph)
        hyp = arr[i]
        np.testing.assert_array_equal(hyp, ref, strict=True)

    left, middle, right = arr[..., :w, :], arr[..., w:-w, :], arr[..., -w:, :]
    # Red
    assert np.all(left[..., 0] >= 252)
    assert np.all(left[..., 1] == 0)
    assert np.all(left[..., 2] == 0)

    # Green
    assert np.all(middle[..., 0] == 0)
    assert np.all(middle[..., 1] >= 253)
    assert np.all(middle[..., 2] == 0)

    # Blue
    assert np.all(right[..., 0] == 0)
    assert np.all(right[..., 1] == 0)
    assert np.all(right[..., 2] >= 253)


def test_batch_decode_image_handle_failure():
    """load_image_batch dismisses failures when strict=False."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 32 sample_%03d.jpg"
    srcs = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG", *[src.path for src in srcs]]

    buffer = spdl.io.load_image_batch(
        flist, width=None, height=None, pix_fmt=None, strict=False
    )
    assert buffer.__array_interface__["shape"] == (32, 3, 240, 320)

    with pytest.raises(RuntimeError):
        spdl.io.load_image_batch(
            flist, width=None, height=None, pix_fmt=None, strict=True
        )
