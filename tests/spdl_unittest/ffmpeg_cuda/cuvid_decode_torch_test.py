import spdl.io

from .test_fixtures import decode_video_h264_cuvid  # noqa

DEFAULT_CUDA = 7


def test_h264_cuvid(decode_video_h264_cuvid):
    """H264 video can be decoded with h264_cuvid and properly converted to CUDA array."""

    import torch

    torch.cuda.select_device(DEFAULT_CUDA)

    buffer = decode_video_h264_cuvid(DEFAULT_CUDA)
    array = spdl.io.to_torch(buffer)

    assert array.shape == (1000, 1, 360, 320)
    assert array.dtype == "uint8"
