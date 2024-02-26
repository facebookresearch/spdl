import numpy as np
import pytest
from numba import cuda
from spdl import libspdl

DEFAULT_CUDA = 0


def _to_array(frame):
    cuda.select_device(DEFAULT_CUDA)  # temp
    return cuda.as_cuda_array(libspdl._BufferWrapper(frame)).copy_to_host()


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
