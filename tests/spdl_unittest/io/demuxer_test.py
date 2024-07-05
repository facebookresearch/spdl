import numpy as np

import pytest
import spdl.io
import torch


@pytest.mark.parametrize(
    "cmd,expected",
    [
        (
            "ffmpeg -hide_banner -y -f lavfi -i 'sine=duration=3' -c:a pcm_s16le sample.wav",
            True,
        ),
        (
            "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4",
            True,
        ),
        ("ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample.mp4", False),
        ("ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg", False),
    ],
)
def test_demuxer_has_audio(get_sample, cmd, expected):
    """has_audio returns true for audio stream"""
    sample = get_sample(cmd)

    with spdl.io.Demuxer(sample.path) as demuxer:
        assert demuxer.has_audio() == expected


def test_demuxer_accept_numpy_array(get_sample):
    """Can instantiate Demuxer with numpy array as source without copying data."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
    sample = get_sample(cmd)

    with open(sample.path, "rb") as f:
        data = f.read()

    src = np.frombuffer(data, dtype=np.uint8)

    assert np.any(src)
    with spdl.io.Demuxer(src, _zero_clear=True) as demuxer:
        demuxer.demux_video()
    assert not np.any(src)


def test_demuxer_accept_torch_tensor(get_sample):
    """Can instantiate Demuxer with torch tensor as source without copying data."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -f lavfi -i sine -frames:v 10 sample.mp4"
    sample = get_sample(cmd)

    with open(sample.path, "rb") as f:
        data = f.read()

    src = torch.frombuffer(data, dtype=torch.uint8)

    assert torch.any(src)
    with spdl.io.Demuxer(src, _zero_clear=True) as demuxer:
        demuxer.demux_video()
    assert not torch.any(src)
