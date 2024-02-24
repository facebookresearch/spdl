from dataclasses import dataclass

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
        libspdl.decode_image_nvdec(sample.path, cuda_device_index=DEFAULT_CUDA).get()
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
        libspdl.decode_image_nvdec(yuvj420p.path, cuda_device_index=DEFAULT_CUDA).get()
    )

    height = yuvj420p.height + yuvj420p.height // 2

    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, height, yuvj420p.width)
