import numpy as np
import pytest
from numba import cuda
from spdl import libspdl


SRC = "/home/moto/test.mp4"
HEIGHT, WIDTH = 540, 960


def _save(array, prefix):
    from PIL import Image

    for i, arr in enumerate(array):
        Image.fromarray(arr[0]).save(f"{prefix}_{i}.png")


def _get_frames(cuda_device_index=0, timestamp=(0, 0.1), **cfg):
    cuda.select_device(cuda_device_index)
    futures = libspdl.decode_video_nvdec(
        SRC,
        timestamps=[timestamp],
        cuda_device_index=cuda_device_index,
        **cfg,
    )
    results = futures.get()
    return cuda.as_cuda_array(libspdl._BufferWrapper(results[0])).copy_to_host()


def _get_frames_set(timestamps, cuda_device_index=0, **cfg):
    cuda.select_device(cuda_device_index)
    futures = libspdl.decode_video_nvdec(
        SRC,
        timestamps=timestamps,
        cuda_device_index=cuda_device_index,
        **cfg,
    )
    return [
        cuda.as_cuda_array(libspdl._BufferWrapper(res)).copy_to_host()
        for res in futures.get()
    ]


def split_nv12(array):
    w = array.shape[-1]
    h0 = array.shape[-2]
    h1, h2 = h0 * 2 // 3, h0 // 3
    y = array[:, :, :h1, :]
    uv = array[:, :, h1:, :].reshape(-1, 1, h2, w // 2, 2)
    u, v = uv[..., 0], uv[..., 1]
    return y, u, v


def test_nvdec_odd_size():
    """Odd width/height must be rejected"""
    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            SRC, timestamps=[(0, 0.1)], cuda_device_index=0, width=121
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            SRC, timestamps=[(0, 0.1)], cuda_device_index=0, height=257
        )


def test_nvdec_negative():
    """Negative options must be rejected"""
    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            SRC, timestamps=[(0, 0.1)], cuda_device_index=0, crop_left=-1
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            SRC, timestamps=[(0, 0.1)], cuda_device_index=0, crop_top=-1
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            SRC, timestamps=[(0, 0.1)], cuda_device_index=0, crop_right=-1
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            SRC, timestamps=[(0, 0.1)], cuda_device_index=0, crop_bottom=-1
        )


def test_nvdec_decode():
    """Check NVDEC decoder with basic configuration."""
    cuda_device_index = 0

    array = _get_frames(cuda_device_index=cuda_device_index)

    # y, u, v = split_nv12(array)
    # _save(array, "./base")
    # _save(y, "./base_y")
    # _save(u, "./base_u")
    # _save(v, "./base_v")

    assert array.dtype == np.uint8
    assert array.shape == (3, 1, HEIGHT + HEIGHT // 2, WIDTH)


def test_nvdec_decode_resize():
    """Check NVDEC decoder with resizing."""
    cuda_device_index = 0
    width, height = 640, 540

    array = _get_frames(cuda_device_index=cuda_device_index, width=width, height=height)

    # _save(array, "./resize")

    assert array.dtype == np.uint8
    assert array.shape == (3, 1, height + height // 2, width)


def test_nvdec_decode_crop():
    """Check NVDEC decoder with resizing."""

    cuda_device_index = 0
    top, bottom, left, right = 80, 160, 200, 100
    height = HEIGHT - top - bottom
    width = WIDTH - left - right

    h2, w2 = height // 2, width // 2
    l2, t2 = left // 2, top // 2

    array = _get_frames(
        cuda_device_index=cuda_device_index,
        crop_top=top,
        crop_bottom=bottom,
        crop_left=left,
        crop_right=right,
    )
    y, u, v = split_nv12(array)

    # _save(array, "./crop")
    # _save(y, "./crop_y")
    # _save(u, "./crop_u")
    # _save(v, "./crop_v")

    assert array.dtype == np.uint8
    assert array.shape == (3, 1, height + h2, width)

    array0 = _get_frames(
        cuda_device_index=cuda_device_index,
    )
    y0, u0, v0 = split_nv12(array0)

    # _save(array0, "./crop_ref")
    # _save(y0, "./crop_ref_y")
    # _save(u0, "./crop_ref_u")
    # _save(v0, "./crop_ref_v")

    y0 = y0[:, :, top : top + height, left : left + width]
    u0 = u0[:, :, t2 : t2 + h2, l2 : l2 + w2]
    v0 = v0[:, :, t2 : t2 + h2, l2 : l2 + w2]

    # _save(y0, "./crop_ref_y_cropped")
    # _save(u0, "./crop_ref_u_cropped")
    # _save(v0, "./crop_ref_v_cropped")

    assert np.array_equal(y, y0)
    assert np.array_equal(u, u0)
    assert np.array_equal(v, v0)


def test_nvdec_decode_crop_resize():
    """Check NVDEC decoder with cropping and resizing."""

    cuda_device_index = 0
    top, bottom, left, right = 80, 160, 200, 100
    height = (HEIGHT - top - bottom) // 2
    width = (WIDTH - left - right) // 2

    h2 = height // 2

    array = _get_frames(
        cuda_device_index=cuda_device_index,
        crop_top=top,
        crop_bottom=bottom,
        crop_left=left,
        crop_right=right,
        width=width,
        height=height,
    )

    # y, u, v = split_nv12(array)
    # _save(array, "./crop_resize")
    # _save(y, "./crop_resize_y")
    # _save(u, "./crop_resize_u")
    # _save(v, "./crop_resize_v")

    assert array.dtype == np.uint8
    assert array.shape == (3, 1, height + h2, width)


def test_non_zero_keyframes():
    """Key frames at non-zero start time should produce proper number of frames."""
    timestamps = [
        (0.1, 1.0),
        (0.2, 1.0),
        (0.3, 1.0),
        (10, 10.1),
        (20, 20.1),
    ]
    features = libspdl.decode_video_nvdec(
        SRC,
        cuda_device_index=0,
        timestamps=timestamps,
    )
    for ts, frames in zip(timestamps, features.get(), strict=True):
        print(f"{len(frames):3d}, {ts=}")
        assert len(frames) > 0


def test_num_frames_arithmetics():
    """NVDEC with non-zero start time should produce proper number of frames."""
    cuda_device_index = 0

    # NOTE: The source video has 29.97 FPS.
    ref = _get_frames(
        cuda_device_index=cuda_device_index,
        timestamp=(0, 0.5),
    )

    cfgs = [
        # timestamp, args_for_slicing
        ((0, 0.1), [3]),
        ((0, 0.2), [6]),
        ((0, 0.3), [9]),
        ((0, 0.4), [12]),
        ((0, 0.5), [15]),
        ((0.1, 0.2), [-12, -9]),
        ((0.1, 0.3), [-12, -6]),
        ((0.1, 0.4), [-12, -3]),
        ((0.1, 0.5), [-12, None]),
        ((0.2, 0.3), [-9, -6]),
        ((0.2, 0.4), [-9, -3]),
        ((0.2, 0.5), [-9, None]),
        ((0.3, 0.4), [-6, -3]),
        ((0.3, 0.5), [-6, None]),
        ((0.4, 0.5), [-3, None]),
    ]
    frames = _get_frames_set(
        cuda_device_index=cuda_device_index, timestamps=[cfg[0] for cfg in cfgs]
    )

    for (ts, slice_args), fs in zip(cfgs, frames, strict=True):
        print(f"{fs.shape}, {ts=}, {slice_args=}")
        assert np.array_equal(fs, ref[slice(*slice_args)])
