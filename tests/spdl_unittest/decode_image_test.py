import numpy as np
from spdl import libspdl


def test_decode_image_gray_black(run_in_tmpdir):
    """PNG image (gray) can be decoded."""
    # Note: the output of this command is in limited color range. So [0, 255] -> [16, 235]
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample_%d.png"
    w, h = 320, 240

    tmpdir = run_in_tmpdir(cmd)
    src = str(tmpdir / "sample_1.png")
    frame = libspdl.decode_image(src).get()

    gray = libspdl.to_numpy(frame)
    assert gray.dtype == np.uint8
    assert gray.shape == (1, h, w)
    assert np.all(gray == 16)


def test_decode_image_gray_white(run_in_tmpdir):
    """PNG image (gray) can be decoded."""
    # Note: the output of this command is in limited color range. So [0, 255] -> [16, 235]
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0xFFFFFF,format=gray -frames:v 1 sample_%d.png"
    w, h = 320, 240

    tmpdir = run_in_tmpdir(cmd)
    src = str(tmpdir / "sample_1.png")
    frame = libspdl.decode_image(src).get()

    gray = libspdl.to_numpy(frame)
    assert gray.dtype == np.uint8
    assert gray.shape == (1, h, w)
    assert np.all(gray == 235)


def test_decode_image_yuv422(run_in_tmpdir):
    """JPEG image (yuvj422p) can be decoded."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuvj422p -frames:v 1 sample_%d.jpeg"
    w, h = 320, 240
    w2 = w // 2

    tmpdir = run_in_tmpdir(cmd)
    src = str(tmpdir / "sample_1.jpeg")
    frame = libspdl.decode_image(src).get()

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


def test_decode_image_yuv420p(run_in_tmpdir):
    """JPEG image can be converted to yuv420"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample_%d.jpeg"
    w, h = 320, 240
    h2, w2 = h // 2, w // 2

    tmpdir = run_in_tmpdir(cmd)
    src = str(tmpdir / "sample_1.jpeg")
    frame = libspdl.decode_image(src, pix_fmt="yuv420p").get()

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


def test_decode_image_rgb24_red(run_in_tmpdir):
    """JPEG image (yuvj420p) can be decoded and converted to RGB"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0xff0000,format=yuvj420p -frames:v 1 sample_%d.jpeg"
    width, height = 320, 240

    tmpdir = run_in_tmpdir(cmd)
    src = str(tmpdir / "sample_1.jpeg")
    frame = libspdl.decode_image(src, pix_fmt="rgb24").get()

    array = libspdl.to_numpy(frame)
    assert array.dtype == np.uint8
    assert array.shape == (height, width, 3)
    assert np.all(array[..., 0] == 254)
    # FFmpeg 4.1 returns 0 while FFmpeg 6.0 returns 1
    assert np.all(array[..., 1] <= 1)
    assert np.all(array[..., 2] == 0)


def test_decode_image_rgb24_green(run_in_tmpdir):
    """JPEG image (yuvj420p) can be decoded and converted to RGB"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x00ff00,format=yuvj420p -frames:v 1 sample_%d.jpeg"
    width, height = 320, 240

    tmpdir = run_in_tmpdir(cmd)
    src = str(tmpdir / "sample_1.jpeg")
    frame = libspdl.decode_image(src, pix_fmt="rgb24").get()

    array = libspdl.to_numpy(frame)
    assert array.dtype == np.uint8
    assert array.shape == (height, width, 3)
    assert np.all(array[..., 0] == 0)
    assert np.all(array[..., 1] == 255)
    assert np.all(array[..., 2] == 0)


def test_decode_image_rgb24_blue(run_in_tmpdir):
    """JPEG image (yuvj420p) can be decoded and converted to RGB"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x0000ff,format=yuvj420p -frames:v 1 sample_%d.jpeg"
    width, height = 320, 240

    tmpdir = run_in_tmpdir(cmd)
    src = str(tmpdir / "sample_1.jpeg")
    frame = libspdl.decode_image(src, pix_fmt="rgb24").get()

    array = libspdl.to_numpy(frame)
    assert array.dtype == np.uint8
    assert array.shape == (height, width, 3)
    assert np.all(array[..., 0] == 0)
    # FFmpeg 4.1 returns 0 while FFmpeg 6.0 returns 1
    assert np.all(array[..., 1] <= 1)
    assert np.all(array[..., 2] == 254)
