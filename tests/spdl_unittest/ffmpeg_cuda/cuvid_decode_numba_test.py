import spdl.io

from .test_fixtures import decode_video_h264_cuvid  # noqa

DEFAULT_CUDA = 7


def test_h264_cuvid(decode_video_h264_cuvid):
    """H264 video can be decoded with h264_cuvid and properly converted to CUDA array."""
    buffer = decode_video_h264_cuvid(DEFAULT_CUDA)
    array = spdl.io.to_numba(buffer)

    assert array.shape == (1000, 1, 360, 320)
    assert array.dtype == "uint8"

    # Should not crash
    array.copy_to_host()
