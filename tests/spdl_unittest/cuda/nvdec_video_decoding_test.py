# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest
import spdl.io
import spdl.io.utils
import torch

from ..fixture import FFMPEG_CLI, get_sample

if not spdl.io.utils.built_with_nvcodec():
    pytest.skip(  # pyre-ignore: [29]
        "SPDL is not compiled with NVCODEC support", allow_module_level=True
    )


DEFAULT_CUDA = 0


def _decode_video(src, timestamp=None, allocator=None, **decode_options):
    device_config = spdl.io.cuda_config(
        device_index=DEFAULT_CUDA,
        allocator=allocator,
    )
    packets = spdl.io.demux_video(src, timestamp=timestamp)
    buffer = spdl.io.decode_packets_nvdec(
        packets, device_config=device_config, **decode_options
    )
    return spdl.io.to_torch(buffer)


@pytest.fixture
def dummy():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 2 sample.mp4"
    return get_sample(cmd)


def test_nvdec_no_file():
    """Does not crash when handling non-existing file"""
    with pytest.raises(RuntimeError):
        _decode_video("lcbjbbvbhiibdctiljrnfvttffictefh.mp4")


def test_nvdec_odd_size(dummy):
    """Odd width/height must be rejected"""
    with pytest.raises(RuntimeError):
        _decode_video(dummy.path, width=121)

    with pytest.raises(RuntimeError):
        _decode_video(dummy.path, height=257)


def test_nvdec_negative(dummy):
    """Negative options must be rejected"""
    with pytest.raises(RuntimeError):
        _decode_video(dummy.path, crop_left=-1)

    with pytest.raises(RuntimeError):
        _decode_video(dummy.path, crop_top=-1)

    with pytest.raises(RuntimeError):
        _decode_video(dummy.path, crop_right=-1)

    with pytest.raises(RuntimeError):
        _decode_video(dummy.path, crop_bottom=-1)


def test_nvdec_video_smoke_test():
    """Can decode video with NVDEC"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"

    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)
    print(packets)
    buffer = spdl.io.decode_packets_nvdec(
        packets,
        device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
    )

    tensor = spdl.io.to_torch(buffer)
    print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")
    assert tensor.shape[0] == 1000
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 240
    assert tensor.shape[3] == 320


def _save(array, prefix):
    from PIL import Image

    for i, arr in enumerate(array):
        Image.fromarray(arr[0]).save(f"{prefix}_{i}.png")


def split_nv12(array):
    w = array.shape[-1]
    h0 = array.shape[-2]
    h1, h2 = h0 * 2 // 3, h0 // 3
    y = array[:, :, :h1, :]
    uv = array[:, :, h1:, :].reshape(-1, 1, h2, w // 2, 2)
    u, v = uv[..., 0], uv[..., 1]
    return y, u, v


@pytest.fixture
def h264():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    return get_sample(cmd)


def test_nvdec_decode_h264_420p_basic(h264):
    """NVDEC can decode YUV420P video."""
    array = _decode_video(h264.path, timestamp=(0, 1.0))

    # y, u, v = split_nv12(array)
    # _save(array, "./base")
    # _save(y, "./base_y")
    # _save(u, "./base_u")
    # _save(v, "./base_v")

    assert array.dtype == torch.uint8
    assert array.shape == (25, 3, 240, 320)


# TODO: Test other formats like MJPEG, MPEG, HEVC, VC1 AV1 etc...


def test_nvdec_decode_video_torch_allocator(h264):
    """NVDEC can decode YUV420P video."""
    allocator_called, deleter_called = False, False

    def allocator(size, device, stream):
        print("Calling allocator", flush=True)
        ptr = torch.cuda.caching_allocator_alloc(size, device, stream)
        nonlocal allocator_called
        allocator_called = True
        return ptr

    def deleter(ptr):
        print("Calling deleter", flush=True)
        torch.cuda.caching_allocator_delete(ptr)
        nonlocal deleter_called
        deleter_called = True

    def _test():
        assert not allocator_called
        assert not deleter_called
        array = _decode_video(
            h264.path,
            timestamp=(0, 1.0),
            allocator=(allocator, deleter),
        )
        assert allocator_called
        assert array.dtype == torch.uint8
        assert array.shape == (25, 3, 240, 320)

    _test()

    gc.collect()
    assert deleter_called


@pytest.mark.xfail(
    raises=RuntimeError,
    reason=(
        "FFmpeg seems to have issue with seeking HEVC. "
        "It returns 'Operation not permitted'. "
        "See https://trac.ffmpeg.org/ticket/9412"
    ),
)
def test_nvdec_decode_hevc_P010_basic():
    """NVDEC can decode HEVC video."""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -t 3 -c:v libx265 -pix_fmt yuv420p10le -vtag hvc1 -y sample.hevc"
    sample = get_sample(cmd)

    array = _decode_video(
        sample.path,
        timestamp=(0, 1.0),
    )

    height = sample.height + sample.height // 2

    assert array.dtype == torch.uint8
    assert array.shape == (25, 1, height, 320)


def test_nvdec_decode_h264_420p_resize(h264):
    """Check NVDEC decoder with resizing."""
    width, height = 160, 120

    array = _decode_video(
        h264.path,
        timestamp=(0, 1.0),
        width=width,
        height=height,
    )

    # _save(array, "./resize")

    assert array.dtype == torch.uint8
    assert array.shape == (25, 3, height, width)


def test_nvdec_decode_h264_420p_crop(h264):
    """Check NVDEC decoder with cropping."""
    top, bottom, left, right = 40, 80, 100, 50
    h = 240 - top - bottom
    w = 320 - left - right

    rgb = _decode_video(
        h264.path,
        timestamp=(0, 1.0),
        crop_top=top,
        crop_bottom=bottom,
        crop_left=left,
        crop_right=right,
    )

    assert rgb.dtype == torch.uint8
    assert rgb.shape == (25, 3, h, w)

    rgba0 = _decode_video(
        h264.path,
        timestamp=(0, 1.0),
    )

    for i in range(3):
        assert torch.equal(rgb[:, i], rgba0[:, i, top : top + h, left : left + w])


def test_nvdec_decode_crop_resize(h264):
    """Check NVDEC decoder with cropping and resizing."""
    top, bottom, left, right = 40, 80, 100, 60
    h = (240 - top - bottom) // 2
    w = (320 - left - right) // 2

    array = _decode_video(
        h264.path,
        timestamp=(0.0, 1.0),
        crop_top=top,
        crop_bottom=bottom,
        crop_left=left,
        crop_right=right,
        width=w,
        height=h,
    )

    assert array.dtype == torch.uint8
    assert array.shape == (25, 3, h, w)


def _is_ffmpeg4():
    vers = spdl.io.utils.get_ffmpeg_versions()
    print(vers)
    return vers["libavutil"][0] < 57


@pytest.mark.skipif(
    _is_ffmpeg4(), reason="FFmpeg4 is known to return a different result."
)
def test_color_conversion_rgba():
    """Providing pix_fmt="rgba" should produce (N,4,H,W) array."""
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                          \
        -f lavfi -i color=color=0xff0000:size=32x64 \
        -f lavfi -i color=color=0x00ff00:size=32x64 \
        -f lavfi -i color=color=0x0000ff:size=32x64 \
        -filter_complex hstack=inputs=3             \
        -frames:v 25 sample.mp4
    """
    height, width = 64, 32
    sample = get_sample(cmd)

    array = _decode_video(sample.path, pix_fmt="rgb")

    assert array.dtype == torch.uint8
    assert array.shape == (25, 3, height, 3*width)

    # Red
    assert torch.all(array[:, 0, :, :width] == 255)
    assert torch.all(array[:, 1, :, :width] == 22)  # TODO: investivate if this is correct.
    assert torch.all(array[:, 2, :, :width] == 0)

    # Green
    assert torch.all(array[:, 0, :, width:2*width] == 0)
    assert torch.all(array[:, 1, :, width:2*width] == 217)  # TODO: investivate if this is correct.
    assert torch.all(array[:, 2, :, width:2*width] == 0)

    # Blue
    assert torch.all(array[:, 0, :, 2*width:] == 0)
    assert torch.all(array[:, 1, :, 2*width:] == 14)  # TODO: investivate if this is correct.
    assert torch.all(array[:, 2, :, 2*width:] == 255)
