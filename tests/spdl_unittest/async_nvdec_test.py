import asyncio

import pytest

import spdl.io

DEFAULT_CUDA = 0


def test_decode_video_nvdec(get_sample):
    """Can decode video with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        decode_tasks = []
        conversion_tasks = []
        async for packets in spdl.io.async_streaming_demux(
            sample.path, "video", timestamps=timestamps
        ):
            print(packets)
            decode_tasks.append(
                spdl.io.async_decode_packets_nvdec(
                    packets, cuda_device_index=DEFAULT_CUDA
                )
            )
        results = await asyncio.gather(*decode_tasks)
        for frames in results:
            print(frames)
            conversion_tasks.append(spdl.io.async_convert_frames(frames))
        results = await asyncio.gather(*conversion_tasks)
        for buffer in results:
            tensor = spdl.io.to_torch(buffer)
            print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")

    asyncio.run(_test())


async def _decode_image(path):
    packets = await spdl.io.async_demux(path, "image")
    print(packets)
    frames = await spdl.io.async_decode_packets_nvdec(
        packets, cuda_device_index=DEFAULT_CUDA
    )
    print(frames)
    return frames


def test_decode_image_nvdec(get_sample):
    """Can decode image with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        import torch

        frames = await _decode_image(sample.path)
        buffer = await spdl.io.async_convert_frames(frames)
        tensor = spdl.io.to_torch(buffer)
        print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")
        assert tensor.shape == torch.Size([4, 240, 320])
        assert tensor.dtype == torch.uint8
        assert tensor.device == torch.device("cuda", DEFAULT_CUDA)

    asyncio.run(_test())


def test_batch_decode_image(get_samples):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG"] + samples

    async def _test():
        demuxing = [spdl.io.async_demux(path, "image") for path in flist]
        decoding = []
        frames = []
        for i, result in enumerate(
            await asyncio.gather(*demuxing, return_exceptions=True)
        ):
            print(result)
            if i == 0:
                assert isinstance(result, Exception)
                continue
            coro = spdl.io.async_decode_packets_nvdec(
                result, cuda_device_index=DEFAULT_CUDA, pix_fmt="rgba"
            )
            decoding.append(asyncio.create_task(coro))

        done, _ = await asyncio.wait(decoding, return_when=asyncio.ALL_COMPLETED)
        for result in done:
            print(result)
            frames.append(result.result())

        buffer = await spdl.io.async_convert_frames(frames)
        assert buffer.shape == [250, 4, 240, 320]

    asyncio.run(_test())


def test_convert_cpu_video_fail(get_sample):
    """Can decode video with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        packets = await spdl.io.async_demux(sample.path, "video")
        frames = await spdl.io.async_decode_packets_nvdec(
            packets, cuda_device_index=DEFAULT_CUDA
        )
        print(frames)
        with pytest.raises(TypeError):
            await spdl.io.async_convert_frames_cpu(frames)

    asyncio.run(_test())


def test_convert_cpu_image_fail(get_sample):
    """Passing NVDEC frames to async_convert_frames_cpu should fail"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        frames = await _decode_image(sample.path)

        with pytest.raises(TypeError):
            await spdl.io.async_convert_frames_cpu(frames)

    asyncio.run(_test())


def test_convert_cpu_batch_image_fail(get_samples):
    """Passing NVDEC frames to async_convert_frames_cpu should fail"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = samples

    async def _test():
        frames = await asyncio.gather(*[_decode_image(path) for path in flist])

        with pytest.raises(TypeError):
            await spdl.io.async_convert_frames_cpu(frames)

    asyncio.run(_test())
