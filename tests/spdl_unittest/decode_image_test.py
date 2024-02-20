import numpy as np
from spdl import libspdl

SRC = "test.png"
WIDTH = 660
HEIGHT = 300


def _get_image_frame(**kwargs):
    future = libspdl.decode_image(SRC, **kwargs)
    return future.get(), HEIGHT, WIDTH


def test_decode_image_rgb24():
    frame, h, w = _get_image_frame(pix_fmt="rgb24")
    array = libspdl.to_numpy(frame)

    assert array.dtype == np.uint8
    assert array.shape == (h, w, 3)


def test_decode_image_yuv420():
    frame, h, w = _get_image_frame(pix_fmt="yuv420p")
    h2, w2 = h // 2, w // 2

    yuv = libspdl.to_numpy(frame)
    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, h + h2, w)

    y = libspdl.to_numpy(frame, index=0)
    assert y.dtype == np.uint8
    assert y.shape == (1, h, w)

    u = libspdl.to_numpy(frame, index=1)
    assert u.dtype == np.uint8
    assert u.shape == (1, h2, w2)
    assert np.array_equal(u, yuv[..., h:, :w2])

    v = libspdl.to_numpy(frame, index=2)
    assert v.dtype == np.uint8
    assert v.shape == (1, h2, w2)
    assert np.array_equal(v, yuv[..., h:, w2:])


def test_decode_image_yuv422():
    frame, h, w = _get_image_frame(pix_fmt="yuvj422p")
    w2 = w // 2

    yuv = libspdl.to_numpy(frame)
    assert yuv.dtype == np.uint8
    assert yuv.shape == (1, h + h, w)

    y = libspdl.to_numpy(frame, index=0)
    assert y.dtype == np.uint8
    assert y.shape == (1, h, w)

    u = libspdl.to_numpy(frame, index=1)
    assert u.dtype == np.uint8
    assert u.shape == (1, h, w2)
    assert np.array_equal(u, yuv[..., h:, :w2])

    v = libspdl.to_numpy(frame, index=2)
    assert v.dtype == np.uint8
    assert v.shape == (1, h, w2)
    assert np.array_equal(v, yuv[..., h:, w2:])
