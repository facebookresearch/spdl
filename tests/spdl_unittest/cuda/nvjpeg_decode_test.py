import asyncio
import concurrent.futures
from random import randbytes

import pytest

import spdl.io
import torch

DEFAULT_CUDA = 0


def test_decode_pix_fmt(get_sample):
    """"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(data, pix_fmt):
        buffer = await spdl.io.async_decode_image_nvjpeg(
            data, cuda_device_index=DEFAULT_CUDA, pix_fmt=pix_fmt
        )
        tensor = spdl.io.to_torch(buffer)
        assert tensor.dtype == torch.uint8
        assert tensor.shape == torch.Size([3, 240, 320])
        assert tensor.device == torch.device("cuda", DEFAULT_CUDA)
        assert not torch.equal(tensor[0], tensor[1])
        assert not torch.equal(tensor[1], tensor[2])
        assert not torch.equal(tensor[2], tensor[0])
        return tensor

    with open(sample.path, "rb") as f:
        data = f.read()

    rgb_tensor = asyncio.run(_test(data, "rgb"))
    bgr_tensor = asyncio.run(_test(data, "bgr"))

    assert torch.equal(rgb_tensor[0], bgr_tensor[2])
    assert torch.equal(rgb_tensor[1], bgr_tensor[1])
    assert torch.equal(rgb_tensor[2], bgr_tensor[0])


def test_decode_rubbish(get_sample):
    """When decoding fails, it should raise an error instead of segfault then,
    subsequent valid decodings should succeed"""

    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    executor = spdl.io.Executor(1, "SingleDecoderExecutor")

    async def _test(data):
        for _ in range(10):
            rubbish = randbytes(2096)
            with pytest.raises(RuntimeError):
                await spdl.io.async_decode_image_nvjpeg(
                    rubbish, cuda_device_index=DEFAULT_CUDA, executor=executor
                )

        for _ in range(10):
            buffer = await spdl.io.async_decode_image_nvjpeg(
                data, cuda_device_index=DEFAULT_CUDA, executor=executor
            )

            tensor = spdl.io.to_torch(buffer)
            assert tensor.dtype == torch.uint8
            assert tensor.shape == torch.Size([3, 240, 320])
            assert tensor.device == torch.device("cuda", DEFAULT_CUDA)
            assert not torch.equal(tensor[0], tensor[1])
            assert not torch.equal(tensor[1], tensor[2])
            assert not torch.equal(tensor[2], tensor[0])
        return tensor

    with open(sample.path, "rb") as f:
        data = f.read()

    asyncio.run(_test(data))
