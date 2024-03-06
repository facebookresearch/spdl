import numpy as np
import pytest

import spdl
from spdl import libspdl


DEFAULT_CUDA = 0


def _to_array(frame):
    array = spdl.to_torch(frame)

    assert str(array.device) == f"cuda:{DEFAULT_CUDA}"
    return array.cpu().numpy()


def test_decode_image_yuv422(get_sample):
    """JPEG image (yuv422p) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd, width=320, height=240)

    yuv = _to_array(
        libspdl.decode_image_nvdec(
            sample.path, cuda_device_index=DEFAULT_CUDA, pix_fmt=None
        ).get()
    )

    height = sample.height + sample.height // 2

    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, height, sample.width)


@pytest.fixture
def yuvj420p(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj420p -frames:v 1 sample.jpeg"
    return get_sample(cmd, width=320, height=240)


def test_decode_image_yuv420(yuvj420p):
    """JPEG image (yuvj420p) can be decoded."""
    yuv = _to_array(
        libspdl.decode_image_nvdec(
            yuvj420p.path, cuda_device_index=DEFAULT_CUDA, pix_fmt=None
        ).get()
    )

    height = yuvj420p.height + yuvj420p.height // 2

    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, height, yuvj420p.width)


def test_decode_image_convert_rgba(yuvj420p):
    """Providing pix_fmt="rgba" should produce (4,H,W) array."""
    array = _to_array(
        libspdl.decode_image_nvdec(
            yuvj420p.path,
            cuda_device_index=DEFAULT_CUDA,
            pix_fmt="rgba",
        ).get()
    )

    assert array.dtype == np.uint8
    assert array.shape == (4, yuvj420p.height, yuvj420p.width)


def test_batch_decode_image_rgba32(get_samples):
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                          \
        -f lavfi -i color=color=0xff0000:size=32x64 \
        -f lavfi -i color=color=0x00ff00:size=32x64 \
        -f lavfi -i color=color=0x0000ff:size=32x64 \
        -filter_complex hstack=inputs=3             \
        -frames:v 32 sample_%03d.jpeg
    """
    height, width = 64, 96
    # fmt: on
    flist = get_samples(cmd)

    frames = libspdl.batch_decode_image_nvdec(
        flist, cuda_device_index=DEFAULT_CUDA, pix_fmt="rgba"
    ).get()
    arrays = [_to_array(f) for f in frames]

    for array in arrays:
        assert array.dtype == np.uint8
        assert array.shape == (4, height, width)

        # Red
        assert np.all(array[0, :, :32] == 255)
        assert np.all(array[1, :, :32] == 10)
        assert np.all(array[2, :, :32] == 0)

        # Green
        assert np.all(array[0, :, 32:64] == 0)
        assert np.all(array[1, :, 32:64] == 232)
        assert np.all(array[2, :, 32:64] == 0)

        # Blue
        assert np.all(array[0, :, 64:] == 0)
        assert np.all(array[1, :, 64:] == 0)
        assert np.all(array[2, :, 64:] == 255)

        # alpha
        assert np.all(array[3] == 255)


def test_batch_decode_image_strict(get_samples):
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y                          \
        -f lavfi -i color=color=0xff0000:size=32x64 \
        -f lavfi -i color=color=0x00ff00:size=32x64 \
        -f lavfi -i color=color=0x0000ff:size=32x64 \
        -filter_complex hstack=inputs=3             \
        -frames:v 32 sample_%03d.jpeg
    """
    # fmt: on
    flist = get_samples(cmd)
    flist.append("foo.png")

    with pytest.raises(RuntimeError):
        libspdl.batch_decode_image_nvdec(flist, cuda_device_index=DEFAULT_CUDA).get(
            strict=True
        )

    frames = libspdl.batch_decode_image_nvdec(
        flist, cuda_device_index=DEFAULT_CUDA
    ).get(strict=False)

    assert len(frames) == 32


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

    # Because currently we cannot configure the thread pool size on-the-fly, so we run this many
    # times so that decoder threads receive the input at least twice.
    # TODO: make thread pool configurable and udpate test with single thread pool for better reproducibility.
    srcs = [sample.path] * 128
    for _ in range(10):
        with pytest.raises(RuntimeError):
            libspdl.batch_decode_image_nvdec(
                srcs, cuda_device_index=DEFAULT_CUDA, pix_fmt=None
            ).get()
