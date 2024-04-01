import asyncio

import numpy as np

import pytest

import spdl.io
from spdl.lib import _libspdl


def test_failure():
    """demux async functoins fails normally if the input does not exist"""

    ts = [(0, 1)]

    async def _test_audio():
        async for packets in spdl.io.async_demux_audio(
            "FOO.mp3",
            timestamps=ts,
        ):
            pass

    async def _test_video():
        async for packets in spdl.io.async_demux_video(
            "FOOBAR.mp4",
            timestamps=ts,
        ):
            pass

    async def _test_image():
        await spdl.io.async_demux_image("FOO.jpg")

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_audio())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_video())

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        asyncio.run(_test_image())


async def _decode_packet(packets):
    frames = await spdl.io.async_decode_packets(packets)
    print(frames)
    buffer = await spdl.io.async_convert_frames(frames)
    print(buffer)
    array = spdl.io.to_numpy(buffer)
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
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)
    N = 2

    async def _test():
        timestamps = [(i, i + 1) for i in range(N)]
        gen = spdl.io.async_demux_audio(sample.path, timestamps=timestamps)
        arrays = await _test_async_decode(gen, N)
        assert len(arrays) == N
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (48000, 1)
            assert arr.dtype == np.int16

    asyncio.run(_test())


def test_decode_audio_clips_num_frames(get_sample):
    """Can decode audio clips with padding/dropping."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=16000:duration=1' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _decode(src, num_frames=None):
        async for packets in spdl.io.async_demux_audio(src, timestamps=[(0, 1)]):
            frames = await spdl.io.async_decode_packets(packets, num_frames=num_frames)
            buffer = await spdl.io.async_convert_frames_cpu(frames)
            return spdl.io.to_numpy(buffer)

    async def _test(src):
        arr0 = await _decode(src)
        assert arr0.dtype == np.int16
        assert arr0.shape == (16000, 1)

        num_frames = 8000
        arr1 = await _decode(src, num_frames=num_frames)
        assert arr1.dtype == np.int16
        assert arr1.shape == (num_frames, 1)
        assert np.all(arr1 == arr0[:num_frames])

        num_frames = 32000
        arr2 = await _decode(src, num_frames=num_frames)
        assert arr2.dtype == np.int16
        assert arr2.shape == (num_frames, 1)
        assert np.all(arr2[:16000] == arr0)
        assert np.all(arr2[16000:] == 0)

    asyncio.run(_test(sample.path))


def test_decode_video_clips(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)
    N = 10

    async def _test():
        timestamps = [(i, i + 1) for i in range(N)]
        gen = spdl.io.async_demux_video(sample.path, timestamps=timestamps)
        arrays = await _test_async_decode(gen, N)
        assert len(arrays) == N
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (25, 3, 240, 320)
            assert arr.dtype == np.uint8

    asyncio.run(_test())


def test_decode_video_clips_num_frames(get_sample):
    """Can decode video clips with padding/dropping."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 50 sample.mp4"
    sample = get_sample(cmd)

    async def _decode(src, pix_fmt="rgb24", **kwargs):
        async for packets in spdl.io.async_demux_video(src, timestamps=[(0, 2)]):
            frames = await spdl.io.async_decode_packets(
                packets, pix_fmt=pix_fmt, **kwargs
            )
            buffer = await spdl.io.async_convert_frames_cpu(frames)
            return spdl.io.to_numpy(buffer)

    async def _test(src):
        arr0 = await _decode(src)
        assert arr0.dtype == np.uint8
        assert arr0.shape == (50, 240, 320, 3)

        num_frames = 25
        arr1 = await _decode(src, num_frames=num_frames)
        assert arr1.dtype == np.uint8
        assert arr1.shape == (num_frames, 240, 320, 3)
        assert np.all(arr1 == arr0[:num_frames])

        num_frames = 100
        arr2 = await _decode(src, num_frames=num_frames)
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == arr2[50])

        num_frames = 100
        arr2 = await _decode(src, num_frames=num_frames, pad_mode="black")
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == 0)

        num_frames = 100
        arr2 = await _decode(src, num_frames=num_frames, pad_mode="white")
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == 255)

    asyncio.run(_test(sample.path))


async def _decode_image(path):
    packets = await spdl.io.async_demux_image(path)
    print(packets)
    frames = await spdl.io.async_decode_packets(packets)
    print(frames)
    assert type(frames) is _libspdl.FFmpegImageFrames
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
        buffer = await spdl.io.async_convert_frames(frames)
        array = spdl.io.to_numpy(buffer)
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

        buffer = await spdl.io.async_convert_frames(frames)
        assert buffer.shape == [250, 3, 240, 320]

    asyncio.run(_test())


def test_cancellation():
    """Async task is cancellable"""

    async def _test():
        task = asyncio.create_task(spdl.io._async_sleep(3000))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


def test_cancellation_wait_for():
    """Task awaited with `wait_for` are cancelled simultaneously"""

    async def _test():
        task = asyncio.create_task(spdl.io._async_sleep(1000))
        with pytest.raises(asyncio.exceptions.TimeoutError):
            await asyncio.wait_for(task, timeout=0.1)

    asyncio.run(_test())


def test_cancellation_multi_gather():
    """Multiple tasks awaited with `gather` are cancelled simultaneously"""

    async def _test(N: int):
        tasks = [asyncio.create_task(spdl.io._async_sleep(3000)) for _ in range(N)]
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
    """async_convert_frames_cpu can convert FFmpegAudioFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _test(src):
        ts = [(1, 2)]
        gen = spdl.io.async_demux_audio(src, timestamps=ts)
        arrays = await _test_async_decode(gen, 1)
        array = arrays[0]
        print(array.dtype, array.shape)
        assert array.shape == (48000, 1)

    asyncio.run(_test(sample.path))


def test_async_decode_audio_bytes(get_sample):
    """audio can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    ts = [(1, 2)]

    async def _decode(src):
        gen = spdl.io.async_demux_audio(src, timestamps=ts)
        arrays = await _test_async_decode(gen, 1)
        return arrays[0]

    async def _decode_bytes(src):
        assert src != b"\x00" * len(src)
        gen = spdl.io.async_demux_audio(src, timestamps=ts, _zero_clear=True)
        arrays = await _test_async_decode(gen, 1)
        assert src == b"\x00" * len(src)
        return arrays[0]

    async def _test(path):
        ref = await _decode(path)
        with open(path, "rb") as f:
            hyp = await _decode_bytes(f.read())

        assert hyp.shape == (48000, 1)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))


def test_async_convert_video_cpu(get_sample):
    """async_convert_frames_cpu can convert FFmpegVideoFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        ts = [(0, float("inf"))]
        gen = spdl.io.async_demux_video(src, timestamps=ts)
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
        gen = spdl.io.async_demux_video(src, timestamps=ts)
        arrays = await _test_async_decode(gen, 1)
        return arrays[0]

    async def _decode_bytes(data):
        assert data
        assert data != b"\x00" * len(data)
        gen = spdl.io.async_demux_video(data, timestamps=ts, _zero_clear=True)
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
    """async_convert_frames_cpu can convert FFmpegImageFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        frames = await _decode_image(src)
        buffer = await spdl.io.async_convert_frames_cpu(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (3, 240, 320)

    asyncio.run(_test(sample.path))


def test_demux_image_bytes(get_sample):
    """Image (gray) can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png"
    sample = get_sample(cmd, width=320, height=240)

    async def _decode(src):
        packets = await spdl.io.async_demux_image(src)
        frames = await spdl.io.async_decode_packets(packets)
        buffer = await spdl.io.async_convert_frames(frames)
        return spdl.io.to_numpy(buffer)

    async def _decode_bytes(data):
        assert data != b"\x00" * len(data)
        packets = await spdl.io.async_demux_image(data, _zero_clear=True)
        assert data == b"\x00" * len(data)
        frames = await spdl.io.async_decode_packets(packets)
        buffer = await spdl.io.async_convert_frames(frames)
        return spdl.io.to_numpy(buffer)

    async def _test(path):
        ref = await _decode(path)
        with open(sample.path, "rb") as f:
            hyp = await _decode_bytes(f.read())

        assert hyp.shape == (1, 240, 320)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))


def test_async_convert_batch_image_cpu(get_samples):
    """async_convert_frames_cpu can convert List[FFmpegImageFrames] to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 4 sample_%03d.jpg"
    flist = get_samples(cmd)

    async def _test(flist):
        frames = await _batch_decode_image(flist)
        buffer = await spdl.io.async_convert_frames_cpu(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (4, 3, 240, 320)

    asyncio.run(_test(flist))
