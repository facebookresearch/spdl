import asyncio
import gc

import numpy as np

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
                cuda_config=spdl.io.cuda_config(
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
            cuda_config=spdl.io.cuda_config(
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


def test_array_transfer_numpy():
    """smoke test for transfer_buffer function"""

    dtypes = {
        np.dtype("uint8"): torch.uint8,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
    }

    async def test(array):
        buffer = await spdl.io.async_transfer_buffer(
            array, cuda_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        )
        tensor = spdl.io.to_torch(buffer)

        device = torch.device(f"cuda:{DEFAULT_CUDA}")
        assert tensor.dtype == dtypes[array.dtype]
        assert tensor.shape == torch.Size(array.shape)
        assert tensor.device == torch.device(device)
        assert torch.allclose(tensor, torch.from_numpy(array).to(device))

    for dtype in [np.uint8, np.int32, np.int64]:
        max_val = np.iinfo(dtype).max
        array = np.random.randint(0, max_val, size=(1, 128_000), dtype=dtype)
        asyncio.run(test(array))


def test_array_transfer_torch():
    """smoke test for transfer_buffer function"""

    cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

    async def test(cpu_tensor):
        buffer = await spdl.io.async_transfer_buffer(
            cpu_tensor, cuda_config=cuda_config
        )
        cuda_tensor = spdl.io.to_torch(buffer)

        device = torch.device(f"cuda:{DEFAULT_CUDA}")
        assert cpu_tensor.dtype == cpu_tensor.dtype
        assert cpu_tensor.shape == cpu_tensor.shape
        assert cpu_tensor.device == cpu_tensor.device
        assert torch.allclose(cuda_tensor, cpu_tensor.to(device))

    for dtype in [np.uint8, np.int32, np.int64]:
        max_val = np.iinfo(dtype).max
        array = np.random.randint(0, max_val, size=(1, 128_000), dtype=dtype)
        asyncio.run(test(torch.from_numpy(array)))


def test_array_transfer_non_contiguous_torch():
    """passing noncontiguous array/tensor to transfer_buffer works."""

    cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

    async def test():
        cpu_tensor = torch.arange(24).reshape(6, 4).T[::2, :]
        assert not cpu_tensor.is_contiguous()
        buffer = await spdl.io.async_transfer_buffer(
            cpu_tensor, cuda_config=cuda_config
        )
        cuda_tensor = spdl.io.to_torch(buffer)

        device = torch.device(f"cuda:{DEFAULT_CUDA}")
        assert cpu_tensor.dtype == cpu_tensor.dtype
        assert cpu_tensor.shape == cpu_tensor.shape
        assert cpu_tensor.device == cpu_tensor.device
        assert torch.allclose(cuda_tensor, cpu_tensor.to(device))

    asyncio.run(test())


def test_array_transfer_non_contiguous_numpy():
    """passing noncontiguous array/tensor to transfer_buffer works"""

    cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

    async def test():
        array0 = np.arange(24)
        assert array0.data.contiguous
        arr = array0.reshape(6, 4).T[:, ::2]
        assert not arr.data.contiguous
        buffer = await spdl.io.async_transfer_buffer(arr, cuda_config=cuda_config)
        tensor = spdl.io.to_torch(buffer)

        device = torch.device(f"cuda:{DEFAULT_CUDA}")
        assert tensor.dtype == torch.int64
        assert tensor.shape == torch.Size(arr.shape)
        assert tensor.device == torch.device(device)
        assert torch.allclose(tensor, torch.from_numpy(arr).to(device))

    asyncio.run(test())


def test_array_transfer_concurrent():
    """smoke test for transfering multiple arrays concurrently"""

    async def test(array):
        cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        tasks = [
            spdl.io.async_transfer_buffer(array, cuda_config=cuda_config)
            for _ in range(100)
        ]
        await asyncio.wait(tasks)

    array = np.random.randint(0, 256, size=(1, 64_000), dtype=np.uint8)
    asyncio.run(test(array))
