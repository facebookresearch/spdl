import asyncio

import numpy as np

import pytest

import spdl


def test_failure():
    """demux async functoins fails normally if the input does not exist"""

    ts = [(0, 1)]

    async def _test_audio():
        async for packets in spdl.async_demux_audio(
            "FOO.mp3", timestamps=ts, _exception_backoff=1
        ):
            pass

    async def _test_video():
        async for packets in spdl.async_demux_video(
            "FOOBAR.mp4", timestamps=ts, _exception_backoff=1
        ):
            pass

    async def _test_image():
        await spdl.async_demux_image("FOO.jpg", _exception_backoff=1)

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_audio())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_video())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_image())


async def _test_async_decode(generator):
    decode_tasks = []
    conversion_tasks = []
    async for packets in generator:
        print(packets)
        decode_tasks.append(spdl.async_decode(packets))
    results = await asyncio.gather(*decode_tasks)
    for frames in results:
        print(frames)
        conversion_tasks.append(spdl.async_convert_cpu(frames))
    results = await asyncio.gather(*conversion_tasks)
    for buffer in results:
        array = np.array(buffer, copy=False)
        print(array.shape, array.dtype)


def test_decode_audio_clips(get_sample):
    """Can decode audio clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=10' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    timestamps = [(i, i + 1) for i in range(10)]

    coro = _test_async_decode(
        spdl.async_demux_audio(sample.path, timestamps=timestamps))

    asyncio.run(coro)


def test_decode_video_clips(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    coro = _test_async_decode(
        spdl.async_demux_video(sample.path, timestamps=timestamps))

    asyncio.run(coro)


def test_decode_image(get_sample):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        packets = await spdl.async_demux_image(sample.path)
        print(packets)
        frames = await spdl.async_decode(packets)
        print(frames)
        buffer = await spdl.async_convert_cpu(frames)
        array = np.array(buffer, copy=False)
        print(array.shape, array.dtype)

    asyncio.run(_test())


def test_batch_decode_image(get_samples):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = samples + ["NON_EXISTING_FILE.JPG"]

    async def _test():
        demuxing = [spdl.async_demux_image(path) for path in flist]
        decoding = []
        conversion = []
        for result in await asyncio.gather(*demuxing, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"@@@ Demuxing failed! {type(result).__name__}:{result}")
                continue
            print(f"    {result}")
            decoding.append(asyncio.create_task(spdl.async_decode(result)))

        done, _ = await asyncio.wait(decoding, return_when=asyncio.ALL_COMPLETED)
        for result in done:
            if err := result.exception():
                print(f"    Task: {result.get_name()} failed with error: {err}")
            else:
                print(f"    Task: {result.get_name()}: {result.result()}")
                conversion.append(spdl.async_convert_cpu(result.result()))

        for result in await asyncio.gather(*conversion):
            print(f"    {result}")

    asyncio.run(_test())
