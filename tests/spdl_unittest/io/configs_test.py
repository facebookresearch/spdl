import asyncio

import pytest
import spdl.io


def test_demux_config_smoketest(get_sample):
    """"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _test(src):
        demux_config = spdl.io.demux_config()
        _ = await spdl.io.async_demux_audio(src, demux_config=demux_config)

        demux_config = spdl.io.demux_config(format="wav")
        _ = await spdl.io.async_demux_audio(src, demux_config=demux_config)

        demux_config = spdl.io.demux_config(format_options={"ignore_length": "true"})
        _ = await spdl.io.async_demux_audio(src, demux_config=demux_config)

        demux_config = spdl.io.demux_config(buffer_size=1024)
        _ = await spdl.io.async_demux_audio(src, demux_config=demux_config)

    asyncio.run(_test(sample.path))


def test_demux_config_headless(get_sample):
    """Providing demux_config allows to load headeless audio"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -f s16le -c:a pcm_s16le sample.raw"
    sample = get_sample(cmd)

    async def _test(src):
        with pytest.raises(RuntimeError):
            await spdl.io.async_demux_audio(src)

        demux_config = spdl.io.demux_config(format="s16le")
        _ = await spdl.io.async_demux_audio(src, demux_config=demux_config)

    asyncio.run(_test(sample.path))


def test_decode_config_smoketest(get_sample):
    """"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd)

    async def _test(src):
        packets = await spdl.io.async_demux_video(src)

        cfg = spdl.io.decode_config()
        _ = await spdl.io.async_decode_packets(packets.clone(), decode_config=cfg)

        cfg = spdl.io.decode_config(decoder="h264")
        _ = await spdl.io.async_decode_packets(packets.clone(), decode_config=cfg)

        cfg = spdl.io.decode_config(
            decoder="h264", decoder_options={"nal_length_size": "4"}
        )
        _ = await spdl.io.async_decode_packets(packets.clone(), decode_config=cfg)

    asyncio.run(_test(sample.path))
