from functools import partial

import pytest

import spdl.io
import torch
from spdl.lib import _libspdl


DEFAULT_CUDA = 0


def _decode_image(path, cuda_device_index=DEFAULT_CUDA, pix_fmt="rgba", executor=None):
    buffer = spdl.io.chain_futures(
        spdl.io.demux_media("image", path),
        partial(
            spdl.io.decode_packets_nvdec,
            cuda_device_index=cuda_device_index,
            pix_fmt=pix_fmt,
            executor=executor,
        ),
        spdl.io.convert_frames,
    ).result()
    return spdl.io.to_torch(buffer)


def test_decode_image_yuv422(get_sample):
    """JPEG image (yuv422p) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd, width=320, height=240)

    yuv = _decode_image(sample.path, pix_fmt=None)

    height = sample.height + sample.height // 2

    assert yuv.dtype == torch.uint8
    assert yuv.shape == (1, height, sample.width)

    rgba = _decode_image(sample.path, pix_fmt="rgba")

    assert rgba.dtype == torch.uint8
    assert rgba.shape == (4, sample.height, sample.width)


def test_decode_image_yuv420(get_sample):
    """JPEG image (yuvj420p) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj420p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd, width=320, height=240)

    yuv = _decode_image(sample.path, pix_fmt=None)

    height = sample.height + sample.height // 2

    assert yuv.dtype == torch.uint8
    assert yuv.shape == (1, height, sample.width)

    rgba = _decode_image(sample.path, pix_fmt="rgba")

    assert rgba.dtype == torch.uint8
    assert rgba.shape == (4, sample.height, sample.width)


def test_decode_image_rgba32(get_sample):
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
    # fmt: on
    sample = get_sample(cmd, width=3 * width, height=height)

    array = _decode_image(sample.path, pix_fmt="rgba")
    assert array.dtype == torch.uint8
    assert array.shape == (4, sample.height, sample.width)

    # Red
    assert torch.all(array[0, :, :32] == 255)
    assert torch.all(array[1, :, :32] == 10)
    assert torch.all(array[2, :, :32] == 0)

    # Green
    assert torch.all(array[0, :, 32:64] == 0)
    assert torch.all(array[1, :, 32:64] == 232)
    assert torch.all(array[2, :, 32:64] == 0)

    # Blue
    assert torch.all(array[0, :, 64:] == 0)
    assert torch.all(array[1, :, 64:] == 0)
    assert torch.all(array[2, :, 64:] == 255)

    # alpha
    assert torch.all(array[3] == 255)


def test_decode_multiple_invalid_input(get_sample):
    """When multiple identical invalid inputs are provided, the decoder must throw RuntimeError
    instead of InternalError (AssertionError).

    Fixed by: https://github.com/mthrok/spdl/commit/dcea39a736fdaf523c2622bf8ec1b1688fc575f0

    Before the fix, the decoder did not call `handle_video_sequence` callback for the second time,
    because these inputs are identical. However, the first time `handle_video_sequence` was called,
    the callback throws an exception because the size is not supported by NVDEC.
    This leaves the decoder in a bad state where the decoder is not properly initialized, but the
    decoding continues for the subsequent inputs.
    """
    # Valid JPEG but its size is not supported by NVDEC.
    cmd = (
        "ffmpeg -hide_banner -y -f lavfi -i testsrc=size=16x16 -frames:v 1 sample.jpeg"
    )
    sample = get_sample(cmd, width=16, height=16)

    executor = _libspdl.ThreadPoolExecutor(1, "SingleDecoderExecutor")
    for _ in range(2):
        with pytest.raises(RuntimeError):
            _decode_image(
                sample.path,
                executor=executor,
            ).get()
