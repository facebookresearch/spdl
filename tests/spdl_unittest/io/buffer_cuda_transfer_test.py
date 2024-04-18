import asyncio

import pytest
import spdl.io
import torch

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
        # Till we have a clone method, we decode twice. We need a clone method
        buffer = await spdl.io.async_load_media(media_type, src)
        cpu_tensor = spdl.io.to_torch(buffer)

        buffer = await spdl.io.async_load_media(media_type, src)
        buffer = await spdl.io.async_transfer_buffer_to_cuda(buffer, DEFAULT_CUDA)
        cuda_tensor = spdl.io.to_torch(buffer)

        assert cuda_tensor.is_cuda
        assert cuda_tensor.device == torch.device(f"cuda:{DEFAULT_CUDA}")

        assert torch.allclose(cpu_tensor, cuda_tensor.cpu())

    asyncio.run(_test(sample.path))
