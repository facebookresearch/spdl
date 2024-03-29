import asyncio

import pytest

import spdl

DEFAULT_CUDA = 0


def test_decode_video_nvdec(get_sample):
    """Can decode video with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        decode_tasks = []
        conversion_tasks = []
        async for packets in spdl.async_demux_video(sample.path, timestamps=timestamps):
            print(packets)
            filtered = await spdl.async_apply_bsf(packets)
            print(filtered)
            decode_tasks.append(
                spdl.async_decode_nvdec(filtered, cuda_device_index=DEFAULT_CUDA)
            )
        results = await asyncio.gather(*decode_tasks)
        for frames in results:
            print(frames)
            conversion_tasks.append(spdl.async_convert(frames))
        results = await asyncio.gather(*conversion_tasks)
        for buffer in results:
            tensor = spdl.to_torch(buffer)
            print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")

    asyncio.run(_test())


async def _decode_image(path):
    packets = await spdl.async_demux_image(path)
    print(packets)
    frames = await spdl.async_decode_nvdec(packets, cuda_device_index=DEFAULT_CUDA)
    print(frames)
    return frames


def test_decode_image_nvdec(get_sample):
    """Can decode image with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        import torch

        frames = await _decode_image(sample.path)
        buffer = await spdl.async_convert(frames)
        tensor = spdl.to_torch(buffer)
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
            coro = spdl.async_decode_nvdec(
                result, cuda_device_index=DEFAULT_CUDA, pix_fmt="rgba"
            )
            decoding.append(asyncio.create_task(coro))

        done, _ = await asyncio.wait(decoding, return_when=asyncio.ALL_COMPLETED)
        for result in done:
            print(result)
            frames.append(result.result())

        buffer = await spdl.async_convert(frames)
        assert buffer.shape == [250, 4, 240, 320]

    asyncio.run(_test())


def test_convert_cpu_video_fail(get_sample):
    """Can decode video with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(0, float("inf"))]

    async def _test():
        packets = None
        async for packets in spdl.async_demux_video(sample.path, timestamps=timestamps):
            print(packets)
            break
        filtered = await spdl.async_apply_bsf(packets)
        print(filtered)
        frames = await spdl.async_decode_nvdec(filtered, cuda_device_index=DEFAULT_CUDA)
        print(frames)
        with pytest.raises(TypeError):
            await spdl.async_convert_cpu(frames)

    asyncio.run(_test())


def test_convert_cpu_image_fail(get_sample):
    """Passing NVDEC frames to async_convert_cpu should fail"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        frames = await _decode_image(sample.path)

        with pytest.raises(TypeError):
            await spdl.async_convert_cpu(frames)

    asyncio.run(_test())


def test_convert_cpu_batch_image_fail(get_samples):
    """Passing NVDEC frames to async_convert_cpu should fail"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = samples

    async def _test():
        frames = await asyncio.gather(*[_decode_image(path) for path in flist])

        with pytest.raises(TypeError):
            await spdl.async_convert_cpu(frames)

    asyncio.run(_test())
