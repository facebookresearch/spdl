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


async def _decode_packet(packets):
    frames = await spdl.async_decode(packets)
    print(frames)
    buffer = await spdl.async_convert(frames)
    print(buffer)
    array = spdl.to_numpy(buffer)
    print(array.shape, array.dtype)
    return array


async def _test_async_decode(generator, N):
    # There was a case where the underlying file device was delayed, and the
    # generated sample file is not ready when the test is started, so
    # sleeping here for 1 second to make sure the file is ready.
    await asyncio.sleep(1)

    tasks = []
    async for packets in generator:
        print(packets)
        tasks.append(asyncio.create_task(_decode_packet(packets)))
    assert len(tasks) == N

    done, pending = await asyncio.wait(tasks)
    assert len(pending) == 0
    assert len(done) == len(tasks) == N
    return [t.result() for t in tasks]


def test_decode_audio_clips(get_sample):
    """Can decode audio clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=12' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)
    N = 10

    async def _test():
        timestamps = [(i, i + 1) for i in range(N)]
        gen = spdl.async_demux_audio(sample.path, timestamps=timestamps)
        arrays = await _test_async_decode(gen, N)
        assert len(arrays) == N
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (49152, 1)
            assert arr.dtype == np.int16

    asyncio.run(_test())


def test_decode_video_clips(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)
    N = 10

    async def _test():
        timestamps = [(i, i + 1) for i in range(N)]
        gen = spdl.async_demux_video(sample.path, timestamps=timestamps)
        arrays = await _test_async_decode(gen, N)
        assert len(arrays) == N
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (26, 3, 240, 320)
            assert arr.dtype == np.uint8

    asyncio.run(_test())


async def _decode_image(path):
    packets = await spdl.async_demux_image(path)
    print(packets)
    frames = await spdl.async_decode(packets)
    print(frames)
    assert type(frames) is libspdl.FFmpegImageFrames
    return frames


async def _batch_decode_image(paths):
    return await asyncio.gather(*[_decode_image(path) for path in paths])


def test_decode_image(get_sample):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        frames = await _decode_image(src)
        print(frames)
        buffer = await spdl.async_convert(frames)
        array = spdl.to_numpy(buffer)
        print(array.shape, array.dtype)
        assert array.dtype == np.uint8
        assert array.shape == (3, 240, 320)

    asyncio.run(_test(sample.path))


def test_batch_decode_image(get_samples):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG"] + samples

    async def _test():
        decoding = [asyncio.create_task(_decode_image(path)) for path in flist]
        frames = []
        await asyncio.wait(decoding)
        for i, result in enumerate(decoding):
            if i == 0:
                assert result.exception() is not None
            else:
                frames.append(result.result())

        buffer = await spdl.async_convert(frames)
        assert buffer.shape == [250, 3, 240, 320]

    asyncio.run(_test())


def test_cancellation():
    """Async task is cancellable"""

    async def _test():
        task = asyncio.create_task(spdl._async.async_sleep(3000))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


def test_cancellation_wait_for():
    """Task awaited with `wait_for` are cancelled simultaneously"""

    async def _test():
        task = asyncio.create_task(spdl._async.async_sleep(1000))
        with pytest.raises(asyncio.exceptions.TimeoutError):
            await asyncio.wait_for(task, timeout=0.1)

    asyncio.run(_test())


def test_cancellation_multi_gather():
    """Multiple tasks awaited with `gather` are cancelled simultaneously"""

    async def _test(N: int):
        tasks = [asyncio.create_task(spdl._async.async_sleep(3000)) for _ in range(N)]
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
        gen = spdl.async_demux_audio(src, timestamps=ts)
        arrays = await _test_async_decode(gen, 1)
        array = arrays[0]
        print(array.dtype, array.shape)
        assert array.shape == (480000, 1)

    asyncio.run(_test(sample.path))


def test_async_decode_audio_bytes(get_sample):
    """audio can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=10' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    ts = [(0, float("inf"))]

    async def _decode(src):
        gen = spdl.async_demux_audio(src, timestamps=ts)
        arrays = await _test_async_decode(gen, 1)
        return arrays[0]

    async def _decode_bytes(src):
        assert src != b"\x00" * len(src)
        gen = spdl.async_demux_audio(src, timestamps=ts, _zero_clear=True)
        arrays = await _test_async_decode(gen, 1)
        assert src == b"\x00" * len(src)
        return arrays[0]

    async def _test(path):
        ref = await _decode(path)
        with open(path, "rb") as f:
            hyp = await _decode_bytes(f.read())

        assert hyp.shape == (480000, 1)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))


def test_async_convert_video_cpu(get_sample):
    """async_convert_cpu can convert FFmpegVideoFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        ts = [(0, float("inf"))]
        gen = spdl.async_demux_video(src, timestamps=ts)
        arrays = await _test_async_decode(gen, 1)
        array = arrays[0]
        print(array.dtype, array.shape)
        assert array.shape == (1000, 3, 240, 320)

    asyncio.run(_test(sample.path))


def test_async_decode_video_bytes(get_sample):
    """video can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    ts = [(0, float("inf"))]

    async def _decode(src):
        gen = spdl.async_demux_video(src, timestamps=ts)
        arrays = await _test_async_decode(gen, 1)
        return arrays[0]

    async def _decode_bytes(data):
        assert data
        assert data != b"\x00" * len(data)
        gen = spdl.async_demux_video(data, timestamps=ts, _zero_clear=True)
        arrays = await _test_async_decode(gen, 1)
        assert data == b"\x00" * len(data)
        return arrays[0]

    async def _test(path):
        ref = await _decode(path)
        with open(path, "rb") as f:
            hyp = await _decode_bytes(f.read())

        assert hyp.shape == (1000, 3, 240, 320)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))


def test_async_convert_image_cpu(get_sample):
    """async_convert_cpu can convert FFmpegImageFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        frames = await _decode_image(src)
        buffer = await spdl.async_convert_cpu(frames)
        print(buffer)
        arr = spdl.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (3, 240, 320)

    asyncio.run(_test(sample.path))


def test_demux_image_bytes(get_sample):
    """Image (gray) can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png"
    sample = get_sample(cmd, width=320, height=240)

    async def _decode(src):
        packets = await spdl.async_demux_image(src)
        frames = await spdl.async_decode(packets)
        buffer = await spdl.async_convert(frames)
        return spdl.to_numpy(buffer)

    async def _decode_bytes(data):
        assert data != b"\x00" * len(data)
        packets = await spdl.async_demux_image(data, _zero_clear=True)
        assert data == b"\x00" * len(data)
        frames = await spdl.async_decode(packets)
        buffer = await spdl.async_convert(frames)
        return spdl.to_numpy(buffer)

    async def _test(path):
        ref = await _decode(path)
        with open(sample.path, "rb") as f:
            hyp = await _decode_bytes(f.read())

        assert hyp.shape == (1, 240, 320)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))


def test_async_convert_batch_image_cpu(get_samples):
    """async_convert_cpu can convert List[FFmpegImageFrames] to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 4 sample_%03d.jpg"
    flist = get_samples(cmd)

    async def _test(flist):
        frames = await _batch_decode_image(flist)
        buffer = await spdl.async_convert_cpu(frames)
        print(buffer)
        arr = spdl.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (4, 3, 240, 320)

    asyncio.run(_test(flist))
