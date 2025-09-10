# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import pytest
import spdl.io

from ..fixture import FFMPEG_CLI, get_sample


def test_demux_config_smoketest():
    """"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3 -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    demux_config = spdl.io.demux_config()
    _ = spdl.io.demux_audio(sample.path, demux_config=demux_config)

    demux_config = spdl.io.demux_config(format="wav")
    _ = spdl.io.demux_audio(sample.path, demux_config=demux_config)

    demux_config = spdl.io.demux_config(format_options={"ignore_length": "true"})
    _ = spdl.io.demux_audio(sample.path, demux_config=demux_config)

    demux_config = spdl.io.demux_config(buffer_size=1024)
    _ = spdl.io.demux_audio(sample.path, demux_config=demux_config)


def test_demux_config_headless():
    """Providing demux_config allows to load headeless audio"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3 -f s16le -c:a pcm_s16le sample.raw"
    sample = get_sample(cmd)

    with pytest.raises(RuntimeError):
        spdl.io.demux_audio(sample.path)

    demux_config = spdl.io.demux_config(format="s16le")
    _ = spdl.io.demux_audio(sample.path, demux_config=demux_config)


def test_decode_config_smoketest():
    """"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd)

    packets = spdl.io.demux_video(sample.path)

    cfg = spdl.io.decode_config()
    _ = spdl.io.decode_packets(packets.clone(), decode_config=cfg)

    cfg = spdl.io.decode_config(decoder="h264")
    _ = spdl.io.decode_packets(packets.clone(), decode_config=cfg)

    cfg = spdl.io.decode_config(
        decoder="h264", decoder_options={"nal_length_size": "4"}
    )
    _ = spdl.io.decode_packets(packets.clone(), decode_config=cfg)
