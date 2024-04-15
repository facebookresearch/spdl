import pytest

import spdl.io
import torch

DEFAULT_CUDA = 0


def _decode_video(src, timestamp=None, **decode_options):
    decode_options["cuda_device_index"] = DEFAULT_CUDA
    future = spdl.io.load_media(
        "video",
        src,
        demux_options={"timestamp": timestamp},
        decode_options=decode_options,
        use_nvdec=True,
    )
    return spdl.io.to_torch(future.result())


def _decode_videos(src, timestamps, **kwargs):
    @spdl.utils.chain_futures
    def _f(packets_future):
        packets = yield packets_future
        frames = yield spdl.io.decode_packets_nvdec(
            packets, cuda_device_index=DEFAULT_CUDA, **kwargs
        )
        yield spdl.io.convert_frames(frames)

    futures = []
    for fut in spdl.io.streaming_demux("video", src, timestamps=timestamps):
        futures.append((_f(fut)))

    return [spdl.io.to_torch(f.result()) for f in futures]


@pytest.fixture
def dummy(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 2 sample.mp4"
    return get_sample(cmd, width=320, height=240)


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
def h264(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    return get_sample(cmd, width=320, height=240)


def test_nvdec_decode_h264_420p_basic(h264):
    """NVDEC can decode YUV420P video."""
    array = _decode_video(h264.path, timestamp=(0, 1.0))

    # y, u, v = split_nv12(array)
    # _save(array, "./base")
    # _save(y, "./base_y")
    # _save(u, "./base_u")
    # _save(v, "./base_v")

    assert array.dtype == torch.uint8
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

    array = _decode_video(
        sample.path,
        timestamp=(0, 1.0),
    )

    height = sample.height + sample.height // 2

    assert array.dtype == torch.uint8
    assert array.shape == (25, 1, height, h264.width)


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
    assert array.shape == (25, 4, height, width)


def test_nvdec_decode_h264_420p_crop(h264):
    """Check NVDEC decoder with cropping."""
    top, bottom, left, right = 40, 80, 100, 50
    h = h264.height - top - bottom
    w = h264.width - left - right

    rgba = _decode_video(
        h264.path,
        timestamp=(0, 1.0),
        crop_top=top,
        crop_bottom=bottom,
        crop_left=left,
        crop_right=right,
    )

    assert rgba.dtype == torch.uint8
    assert rgba.shape == (25, 4, h, w)

    rgba0 = _decode_video(
        h264.path,
        timestamp=(0, 1.0),
    )

    for i in range(4):
        assert torch.equal(rgba[:, i], rgba0[:, i, top : top + h, left : left + w])


def test_nvdec_decode_crop_resize(h264):
    """Check NVDEC decoder with cropping and resizing."""
    top, bottom, left, right = 40, 80, 100, 60
    h = (h264.height - top - bottom) // 2
    w = (h264.width - left - right) // 2

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
    assert array.shape == (25, 4, h, w)


def test_num_frames_arithmetics(h264):
    """NVDEC with non-zero start time should produce proper number of frames."""
    ref = _decode_video(
        h264.path,
        timestamp=(0, 1.0),
    )

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
    frames = _decode_videos(
        h264.path,
        timestamps=[cfg[0] for cfg in cfgs],
    )

    for (ts, slice_args), fs in zip(cfgs, frames, strict=True):
        print(f"{fs.shape}, {ts=}, {slice_args=}")
        assert torch.equal(fs, ref[slice(*slice_args)])


def test_color_conversion_rgba(get_sample):
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
    sample = get_sample(cmd, height=height, width=3 * width)

    array = _decode_video(sample.path, pix_fmt="rgba")

    assert array.dtype == torch.uint8
    assert array.shape == (25, 4, sample.height, sample.width)

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
