import pytest
import spdl.io


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
