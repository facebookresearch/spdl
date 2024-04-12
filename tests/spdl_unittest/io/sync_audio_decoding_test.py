from functools import partial

import numpy as np
import pytest

import spdl.io


def _decode_audio(src, sample_fmt=None):
    buffer = spdl.io.chain_futures(
        spdl.io.demux_media("audio", src),
        partial(spdl.io.decode_packets, sample_fmt=sample_fmt),
        spdl.io.convert_frames_cpu,
    ).result()
    return spdl.io.to_numpy(buffer)


@pytest.mark.parametrize(
    "sample_fmts",
    [("s16p", "int16"), ("s16", "int16"), ("fltp", "float32"), ("flt", "float32")],
)
def test_audio_buffer_conversion_s16p(sample_fmts, get_sample):
    # fmt: off
    cmd = """
    ffmpeg -hide_banner -y \
    -f lavfi -i 'sine=sample_rate=8000:frequency=305:duration=5' \
    -f lavfi -i 'sine=sample_rate=8000:frequency=300:duration=5' \
    -filter_complex amerge  -c:a pcm_s16le sample.wav
    """
    # fmt: on
    sample = get_sample(cmd)

    sample_fmt, expected = sample_fmts
    array = _decode_audio(src=sample.path, sample_fmt=sample_fmt)

    assert array.ndim == 2
    assert array.dtype == np.dtype(expected)
    shape = (2, 40000) if sample_fmt.endswith("p") else (40000, 2)
    assert array.shape == shape