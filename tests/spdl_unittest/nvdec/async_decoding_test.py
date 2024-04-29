import asyncio

import pytest

import spdl.io
import spdl.utils

DEFAULT_CUDA = 0

if not spdl.utils.is_nvcodec_available():
    pytest.skip("SPDL is not compiled with NVCODEC support", allow_module_level=True)


def test_decode_video_nvdec(get_sample):
    """Can decode video with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    async def _test():
        decode_tasks = []
        async for packets in spdl.io.async_streaming_demux(
            "video", sample.path, timestamps=timestamps
        ):
            print(packets)
            decode_tasks.append(
                spdl.io.async_decode_packets_nvdec(
                    packets, cuda_device_index=DEFAULT_CUDA
                )
            )
        results = await asyncio.gather(*decode_tasks)
        for buffer in results:
            tensor = spdl.io.to_torch(buffer)
            print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")

    asyncio.run(_test())


async def _decode_image(path):
    packets = await spdl.io.async_demux_media("image", path)
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
        tensor = spdl.io.to_torch(frames)
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
        demuxing = [spdl.io.async_demux_media("image", path) for path in flist]
        packets = []
        for i, result in enumerate(
            await asyncio.gather(*demuxing, return_exceptions=True)
        ):
            if i == 0:
                print(result)
                assert isinstance(result, Exception)
                continue
            packets.append(result)

        assert len(packets) == 250
        frames = await spdl.io.async_decode_packets_nvdec(
            packets, cuda_device_index=DEFAULT_CUDA, pix_fmt="rgba"
        )
        assert frames.shape == [250, 4, 240, 320]

    asyncio.run(_test())
