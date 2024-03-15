import asyncio

import pytest

import spdl


# TODO:
# Add smoke test with other asyncio patterns like `asyncio.wait` and `asyncio.gather`


def test_failure():
    """demux async functoins fails normally if the input does not exist"""

    ts = [(0, 1)]

    async def _test_audio():
        async for packets in spdl.demux_audio_async(
            "FOO.mp3", timestamps=ts, _exception_backoff=1
        ):
            pass

    async def _test_video():
        async for packets in spdl.demux_video_async(
            "FOOBAR.mp4", timestamps=ts, _exception_backoff=1
        ):
            pass

    async def _test_image():
        await spdl.demux_image_async("FOO.jpg", _exception_backoff=1)

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_audio())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_video())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_image())


def test_demux_audio_clips(get_sample):
    """Can demux audio clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=10' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        async for packets in spdl.demux_audio_async(sample.path, timestamps=timestamps):
            print(packets)

    asyncio.run(_test())


def test_demux_video_clips(get_sample):
    """Can demux video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        async for packets in spdl.demux_video_async(sample.path, timestamps=timestamps):
            print(packets)

    asyncio.run(_test())


def test_demux_image(get_sample):
    """Can demux video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample_%03d.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        packets = await spdl.demux_image_async(sample.path)
        print(packets)

    asyncio.run(_test())
