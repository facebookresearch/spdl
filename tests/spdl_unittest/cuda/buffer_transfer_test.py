import asyncio
import gc

import pytest
import spdl.io
import spdl.utils
import torch


if not spdl.utils.is_cuda_available():
    pytest.skip("SPDL is not compiled with NVCODEC support", allow_module_level=True)


DEFAULT_CUDA = 0

CMDS = {
    "audio": "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav",
    "video": "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4",
    "image": "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_async_transfer_buffer_to_cuda(media_type, get_sample):
    """smoke test for transfer_buffer_to_cuda function"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    async def _test(src):
        _ = torch.zeros([0], device=torch.device(f"cuda:{DEFAULT_CUDA}"))

        demux_func = {
            "audio": spdl.io.async_demux_audio,
            "video": spdl.io.async_demux_video,
            "image": spdl.io.async_demux_image,
        }[media_type]

        frames = await spdl.io.async_decode_packets(await demux_func(src))
        buffer = await spdl.io.async_convert_frames(frames)
        cpu_tensor = spdl.io.to_torch(buffer).clone()
        cuda_tensor = spdl.io.to_torch(
            await spdl.io.async_transfer_buffer(
                buffer,
                transfer_config=spdl.io.transfer_config(
                    device_index=DEFAULT_CUDA,
                ),
            )
        )

        assert cuda_tensor.is_cuda
        assert cuda_tensor.device == torch.device(f"cuda:{DEFAULT_CUDA}")

        assert torch.allclose(cpu_tensor, cuda_tensor.cpu())

    asyncio.run(_test(sample.path))


@pytest.mark.parametrize("media_type", ["audio", "video", "image"])
def test_async_transfer_buffer_to_cuda_with_pytorch_allocator(media_type, get_sample):
    """smoke test for transfer_buffer_to_cuda function"""
    cmd = CMDS[media_type]
    sample = get_sample(cmd)

    allocator_called, deleter_called = False, False

    def allocator(size, device, stream):
        print("Calling allocator", flush=True)
        ptr = torch.cuda.caching_allocator_alloc(size, device, stream)
        nonlocal allocator_called
        allocator_called = True
        return ptr

    def deleter(ptr):
        print("Calling deleter", flush=True)
        torch.cuda.caching_allocator_delete(ptr)
        nonlocal deleter_called
        deleter_called = True

    demux_func = {
        "audio": spdl.io.async_demux_audio,
        "video": spdl.io.async_demux_video,
        "image": spdl.io.async_demux_image,
    }[media_type]

    async def _test(src):
        frames = await spdl.io.async_decode_packets(await demux_func(src))
        buffer = await spdl.io.async_convert_frames(frames)
        cpu_tensor = spdl.io.to_torch(buffer).clone()

        assert not allocator_called
        cuda_buffer = await spdl.io.async_transfer_buffer(
            buffer,
            transfer_config=spdl.io.transfer_config(
                device_index=DEFAULT_CUDA,
                allocator=(allocator, deleter),
            ),
        )
        assert allocator_called
        cuda_tensor = spdl.io.to_torch(cuda_buffer)

        assert cuda_tensor.is_cuda
        assert cuda_tensor.device == torch.device(f"cuda:{DEFAULT_CUDA}")
        assert torch.allclose(cpu_tensor, cuda_tensor.cpu())

        assert not deleter_called

    asyncio.run(_test(sample.path))

    gc.collect()
    assert deleter_called
