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


def test_decode_audio_clips(get_sample):
    """Can decode audio clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=10' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        decode_tasks = []
        async for packets in spdl.demux_audio_async(sample.path, timestamps=timestamps):
            print(packets)
            decode_tasks.append(spdl.decode_audio_async(packets))
        results = await asyncio.gather(*decode_tasks)
        for r in results:
            print(r)

    asyncio.run(_test())


def test_decode_video_clips(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        decode_tasks = []
        async for packets in spdl.demux_video_async(sample.path, timestamps=timestamps):
            print(packets)
            decode_tasks.append(spdl.decode_video_async(packets))
        results = await asyncio.gather(*decode_tasks)
        for r in results:
            print(r)

    asyncio.run(_test())


def test_decode_image(get_sample):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        packets = await spdl.demux_image_async(sample.path)
        print(packets)
        result = await spdl.decode_image_async(packets)
        print(result)

    asyncio.run(_test())


def test_batch_decode_image(get_samples):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = samples + ["NON_EXISTING_FILE.JPG"]

    async def _test():
        demuxing = [spdl.demux_image_async(path) for path in flist]
        decoding = []
        for result in await asyncio.gather(*demuxing, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"@@@ Demuxing failed! {type(result).__name__}:{result}")
                continue
            print(f"    {result}")
            decoding.append(asyncio.create_task(spdl.decode_image_async(result)))

        done, _ = await asyncio.wait(decoding, return_when=asyncio.ALL_COMPLETED)
        for result in done:
            if err := result.exception():
                print(f"    Task: {result.get_name()} failed with error: {err}")
            else:
                print(f"    Task: {result.get_name()}: {result.result()}")

    asyncio.run(_test())
