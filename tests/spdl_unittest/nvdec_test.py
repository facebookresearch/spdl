import numpy as np
import pytest
import spdl
from spdl import libspdl

DEFAULT_CUDA = 0


@pytest.fixture
def dummy(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 2 sample.mp4"
    return get_sample(cmd, width=320, height=240)


def test_nvdec_odd_size(dummy):
    """Odd width/height must be rejected"""
    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            dummy.path, timestamps=[(0, 0.1)], cuda_device_index=0, width=121
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            dummy.path, timestamps=[(0, 0.1)], cuda_device_index=0, height=257
        )


def test_nvdec_negative(dummy):
    """Negative options must be rejected"""
    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            dummy.path, timestamps=[(0, 0.1)], cuda_device_index=0, crop_left=-1
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            dummy.path, timestamps=[(0, 0.1)], cuda_device_index=0, crop_top=-1
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            dummy.path, timestamps=[(0, 0.1)], cuda_device_index=0, crop_right=-1
        )

    with pytest.raises(RuntimeError):
        libspdl.decode_video_nvdec(
            dummy.path, timestamps=[(0, 0.1)], cuda_device_index=0, crop_bottom=-1
        )


def _save(array, prefix):
    from PIL import Image

    for i, arr in enumerate(array):
        Image.fromarray(arr[0]).save(f"{prefix}_{i}.png")


def _to_arrays(frames):
    return [spdl.to_torch(f).cpu().numpy() for f in frames]


def split_nv12(array):
    w = array.shape[-1]
    h0 = array.shape[-2]
    h1, h2 = h0 * 2 // 3, h0 // 3
    y = array[:, :, :h1, :]
    uv = array[:, :, h1:, :].reshape(-1, 1, h2, w // 2, 2)
    u, v = uv[..., 0], uv[..., 1]
    return y, u, v


@pytest.fixture
def h264(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    return get_sample(cmd, width=320, height=240)


def test_nvdec_decode_h264_420p_basic(h264):
    """NVDEC can decode YUV420P video."""
    results = libspdl.decode_video_nvdec(
        h264.path,
        timestamps=[(0, 1.0)],
        cuda_device_index=DEFAULT_CUDA,
    ).get()
    array = _to_arrays(results)[0]

    # y, u, v = split_nv12(array)
    # _save(array, "./base")
    # _save(y, "./base_y")
    # _save(u, "./base_u")
    # _save(v, "./base_v")

    assert array.dtype == np.uint8
    assert array.shape == (25, 4, h264.height, h264.width)


# TODO: Test other formats like MJPEG, MPEG, HEVC, VC1 AV1 etc...


@pytest.mark.xfail(
    raises=RuntimeError,
    reason=(
        "FFmpeg seems to have issue with seeking HEVC. "
        "It returns 'Operation not permitted'. "
        "See https://trac.ffmpeg.org/ticket/9412"
    ),
)
def test_nvdec_decode_hevc_P010_basic(get_sample):
    """NVDEC can decode HEVC video."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -t 3 -c:v libx265 -pix_fmt yuv420p10le -vtag hvc1 -y sample.hevc"
    sample = get_sample(cmd, width=320, height=240)

    results = libspdl.decode_video_nvdec(
        sample.path,
        timestamps=[(0, 1.0)],
        cuda_device_index=DEFAULT_CUDA,
    ).get()
    array = _to_arrays(results)[0]

    height = sample.height + sample.height // 2

    assert array.dtype == np.uint8
    assert array.shape == (25, 1, height, h264.width)


def test_nvdec_decode_h264_420p_resize(h264):
    """Check NVDEC decoder with resizing."""
    width, height = 160, 120

    results = libspdl.decode_video_nvdec(
        h264.path,
        timestamps=[(0, 1.0)],
        cuda_device_index=DEFAULT_CUDA,
        width=width,
        height=height,
    ).get()
    array = _to_arrays(results)[0]

    # _save(array, "./resize")

    assert array.dtype == np.uint8
    assert array.shape == (25, 4, height, width)


def test_nvdec_decode_h264_420p_crop(h264):
    """Check NVDEC decoder with cropping."""
    top, bottom, left, right = 40, 80, 100, 50
    h = h264.height - top - bottom
    w = h264.width - left - right

    results = libspdl.decode_video_nvdec(
        h264.path,
        timestamps=[(0, 1.0)],
        cuda_device_index=DEFAULT_CUDA,
        crop_top=top,
        crop_bottom=bottom,
        crop_left=left,
        crop_right=right,
    ).get()
    rgba = _to_arrays(results)[0]

    assert rgba.dtype == np.uint8
    assert rgba.shape == (25, 4, h, w)

    results = libspdl.decode_video_nvdec(
        h264.path,
        timestamps=[(0, 1.0)],
        cuda_device_index=DEFAULT_CUDA,
    ).get()
    rgba0 = _to_arrays(results)[0]

    for i in range(4):
        assert np.array_equal(rgba[:, i], rgba0[:, i, top : top + h, left : left + w])


def test_nvdec_decode_crop_resize(h264):
    """Check NVDEC decoder with cropping and resizing."""
    top, bottom, left, right = 40, 80, 100, 60
    h = (h264.height - top - bottom) // 2
    w = (h264.width - left - right) // 2

    array = _to_arrays(
        libspdl.decode_video_nvdec(
            h264.path,
            timestamps=[(0.0, 1.0)],
            cuda_device_index=DEFAULT_CUDA,
            crop_top=top,
            crop_bottom=bottom,
            crop_left=left,
            crop_right=right,
            width=w,
            height=h,
        ).get()
    )[0]

    assert array.dtype == np.uint8
    assert array.shape == (25, 4, h, w)


def test_num_frames_arithmetics(h264):
    """NVDEC with non-zero start time should produce proper number of frames."""
    ref = _to_arrays(
        libspdl.decode_video_nvdec(
            h264.path,
            timestamps=[(0, 1.0)],
            cuda_device_index=DEFAULT_CUDA,
        ).get()
    )[0]

    # NOTE: The source video has 25 FPS.
    # fmt: off
    cfgs = [
        # timestamp, args_for_slicing
        ((0, 0.2), [5]),
        ((0, 0.4), [10]),
        ((0, 0.6), [15]),
        ((0, 0.8), [20]),
        ((0, 1.0,), [25]),
        ((0.2, 0.4), [5, 10]),
        ((0.2, 0.6), [5, 15]),
        ((0.2, 0.8), [5, 20]),
        ((0.2, 1.0), [5, 25]),
        ((0.4, 0.6), [10, 15]),
        ((0.4, 0.8), [10, 20]),
        ((0.4, 1.0), [10, 25]),
        ((0.6, 0.8), [15, 20]),
        ((0.6, 1.0), [15, 25]),
        ((0.8, 1.0), [20, 25]),
    ]
    # fmt: on
    frames = _to_arrays(
        libspdl.decode_video_nvdec(
            h264.path,
            cuda_device_index=DEFAULT_CUDA,
            timestamps=[cfg[0] for cfg in cfgs],
        ).get()
    )

    for (ts, slice_args), fs in zip(cfgs, frames, strict=True):
        print(f"{fs.shape}, {ts=}, {slice_args=}")
        assert np.array_equal(fs, ref[slice(*slice_args)])


@pytest.fixture
def red(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0xFF0000 -frames:v 100 sample.mp4"
    return get_sample(cmd, width=320, height=240)


@pytest.fixture
def green(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x00FF00 -frames:v 100 sample.mp4"
    return get_sample(cmd, width=320, height=240)


@pytest.fixture
def blue(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x0000FF -frames:v 100 sample.mp4"
    return get_sample(cmd, width=320, height=240)


def test_color_conversion_rgba_red(red):
    """Providing pix_fmt="rgba" should produce (N,4,H,W) array."""
    array = _to_arrays(
        libspdl.decode_video_nvdec(
            red.path,
            timestamps=[(0, 1.0)],
            cuda_device_index=DEFAULT_CUDA,
            pix_fmt="rgba",
        ).get()
    )[0]

    assert array.dtype == np.uint8
    assert array.shape == (25, 4, red.height, red.width)

    assert np.all(array[:, 0, ...] == 255)
    assert np.all(array[:, 1, ...] == 22)  # TODO: investivate if this is correct.
    assert np.all(array[:, 2, ...] == 0)


def test_color_conversion_rgba_green(green):
    """Providing pix_fmt="rgba" should produce (N,4,H,W) array."""
    array = _to_arrays(
        libspdl.decode_video_nvdec(
            green.path,
            timestamps=[(0, 1.0)],
            cuda_device_index=DEFAULT_CUDA,
            pix_fmt="rgba",
        ).get()
    )[0]

    assert array.dtype == np.uint8
    assert array.shape == (25, 4, green.height, green.width)

    assert np.all(array[:, 0, ...] == 0)
    assert np.all(array[:, 1, ...] == 217)  # TODO: investivate if this is correct.
    assert np.all(array[:, 2, ...] == 0)


def test_color_conversion_rgba_blue(blue):
    """Providing pix_fmt="rgba" should produce (N,4,H,W) array."""
    array = _to_arrays(
        libspdl.decode_video_nvdec(
            blue.path,
            timestamps=[(0, 1.0)],
            cuda_device_index=DEFAULT_CUDA,
            pix_fmt="rgba",
        ).get()
    )[0]

    assert array.dtype == np.uint8
    assert array.shape == (25, 4, blue.height, blue.width)

    assert np.all(array[:, 0, ...] == 0)
    assert np.all(array[:, 1, ...] == 14)  # TODO: investivate if this is correct.
    assert np.all(array[:, 2, ...] == 255)
