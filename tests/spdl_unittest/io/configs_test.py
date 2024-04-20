import asyncio
import spdl.io

import pytest


def test_io_config_smoketest(get_sample):
    """"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _test(src):
        io_config = spdl.io.IOConfig()
        _ = await spdl.io.async_demux_media("audio", src, io_config=io_config)

        io_config = spdl.io.IOConfig(format="wav")
        _ = await spdl.io.async_demux_media("audio", src, io_config=io_config)

        io_config = spdl.io.IOConfig(format_options={"max_size": "1024"})
        _ = await spdl.io.async_demux_media("audio", src, io_config=io_config)

        io_config = spdl.io.IOConfig(buffer_size=1024)
        _ = await spdl.io.async_demux_media("audio", src, io_config=io_config)

    asyncio.run(_test(sample.path))


def test_io_config_headless(get_sample):
    """Providing io_config allows to load headeless audio"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -f s16le -c:a pcm_s16le sample.raw"
    sample = get_sample(cmd)

    async def _test(src):
        with pytest.raises(RuntimeError):
            await spdl.io.async_demux_media("audio", src)

        io_config = spdl.io.IOConfig(format="s16le")
        _ = await spdl.io.async_demux_media("audio", src, io_config=io_config)

    asyncio.run(_test(sample.path))
