import asyncio

import numpy as np

import pytest

import spdl
from spdl import libspdl


def test_failure():
    """demux async functoins fails normally if the input does not exist"""

    ts = [(0, 1)]

    async def _test_audio():
        async for packets in spdl.async_demux_audio(
            "FOO.mp3",
            timestamps=ts,
        ):
            pass

    async def _test_video():
        async for packets in spdl.async_demux_video(
            "FOOBAR.mp4",
            timestamps=ts,
        ):
            pass

    async def _test_image():
        await spdl.async_demux_image("FOO.jpg")

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_audio())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_video())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_image())


async def _test_async_decode(generator):
    decode_tasks = []
    conversions = []
    conversions_cpu = []
    async for packets in generator:
        print(packets)
        decode_tasks.append(spdl.async_decode(packets))
    results = await asyncio.gather(*decode_tasks)
    for frames in results:
        print(frames)
        conversions.append(spdl.async_convert(frames))

    results = await asyncio.gather(*conversions)
    for buffer in results:
        array = np.array(buffer, copy=False)
        print(array.shape, array.dtype)

    results = await asyncio.gather(*conversions_cpu)
    for buffer in results:
        array = np.array(buffer, copy=False)
        print(array.shape, array.dtype)


def test_decode_audio_clips(get_sample):
    """Can decode audio clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=10' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    timestamps = [(i, i + 1) for i in range(10)]

    coro = _test_async_decode(
        spdl.async_demux_audio(sample.path, timestamps=timestamps)
    )

    asyncio.run(coro)


def test_decode_video_clips(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    coro = _test_async_decode(
        spdl.async_demux_video(sample.path, timestamps=timestamps)
    )

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
        buffer = await spdl.async_convert(frames)
        array = np.array(buffer, copy=False)
        print(array.shape, array.dtype)

    asyncio.run(_test())


def test_batch_decode_image(get_samples):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG"] + samples

    async def _test():
        demuxing = [spdl.async_demux_image(path) for path in flist]
        decoding = []
        frames = []
        for i, result in enumerate(
            await asyncio.gather(*demuxing, return_exceptions=True)
        ):
            print(result)
            if i == 0:
                assert isinstance(result, Exception)
                continue
            decoding.append(asyncio.create_task(spdl.async_decode(result)))

        done, _ = await asyncio.wait(decoding, return_when=asyncio.ALL_COMPLETED)
        for result in done:
            print(result)
            frames.append(result.result())

        buffer = await spdl.async_convert(frames)
        assert buffer.shape == [250, 3, 240, 320]

    asyncio.run(_test())


def test_cancellation():
    """Async task is cancellable"""

    async def _test():
        loop = asyncio.get_running_loop()

        task = loop.create_task(spdl.async_sleep(3000))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


def test_cancellation_wait_for():
    """Task awaited with `wait_for` are cancelled simultaneously"""

    async def _test():
        loop = asyncio.get_running_loop()

        future = spdl.async_sleep(1000)
        task = loop.create_task(future)
        with pytest.raises(asyncio.exceptions.TimeoutError):
            await asyncio.wait_for(task, timeout=0.1)

    asyncio.run(_test())


def test_cancellation_multi_gather():
    """Multiple tasks awaited with `gather` are cancelled simultaneously"""

    async def _test(N: int):
        loop = asyncio.get_running_loop()

        tasks = [loop.create_task(spdl.async_sleep(3000)) for _ in range(N)]
        task = asyncio.gather(*tasks, return_exceptions=True)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(tasks) == N
        for t in tasks:
            with pytest.raises(asyncio.CancelledError):
                await t

    asyncio.run(_test(3))


def test_async_convert_audio_cpu(get_sample):
    """async_convert_cpu can convert FFmpegAudioFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=10' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _test(src):
        ts = [(0, float("inf"))]
        packets = None
        async for packets in spdl.async_demux_audio(sample.path, timestamps=ts):
            print(packets)
            break
        frames = await spdl.async_decode(packets)
        print(frames)
        assert type(frames) is libspdl.FFmpegAudioFrames
        buffer = await spdl.async_convert_cpu(frames)
        print(buffer)
        arr = spdl.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (480000, 1)

    asyncio.run(_test(sample.path))


def test_async_convert_video_cpu(get_sample):
    """async_convert_cpu can convert FFmpegVideoFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        ts = [(0, float("inf"))]
        packets = None
        async for packets in spdl.async_demux_video(sample.path, timestamps=ts):
            print(packets)
            break
        frames = await spdl.async_decode(packets)
        print(frames)
        assert type(frames) is libspdl.FFmpegVideoFrames
        buffer = await spdl.async_convert_cpu(frames)
        print(buffer)
        arr = spdl.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (1000, 3, 240, 320)

    asyncio.run(_test(sample.path))


def test_async_convert_image_cpu(get_sample):
    """async_convert_cpu can convert FFmpegImageFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        packets = await spdl.async_demux_image(sample.path)
        print(packets)
        frames = await spdl.async_decode(packets)
        print(frames)
        assert type(frames) is libspdl.FFmpegImageFrames
        buffer = await spdl.async_convert_cpu(frames)
        print(buffer)
        arr = spdl.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (3, 240, 320)

    asyncio.run(_test(sample.path))
