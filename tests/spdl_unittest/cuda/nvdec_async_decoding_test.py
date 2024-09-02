# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import gc

import pytest
import spdl.io
import spdl.utils
import torch

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
        demuxer = spdl.io.Demuxer(sample.path)
        for ts in timestamps:
            packets = demuxer.demux_video(ts)
            print(packets)
            decode_tasks.append(
                spdl.io.async_decode_packets_nvdec(
                    packets,
                    device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
                )
            )
        results = await asyncio.gather(*decode_tasks)
        for buffer in results:
            tensor = spdl.io.to_torch(buffer)
            print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")

    asyncio.run(_test())


async def _decode_image(path):
    packets = await spdl.io.async_demux_image(path)
    print(packets)
    frames = await spdl.io.async_decode_packets_nvdec(
        packets, device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA)
    )
    print(frames)
    return frames


def test_decode_image_nvdec(get_sample):
    """Can decode image with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test():
        frames = await _decode_image(sample.path)
        tensor = spdl.io.to_torch(frames)
        print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")
        assert tensor.shape == torch.Size([4, 240, 320])
        assert tensor.dtype == torch.uint8
        assert tensor.device == torch.device("cuda", DEFAULT_CUDA)

    asyncio.run(_test())


def test_batch_decode_image(get_samples):
    """batch loading can handle non-existing file."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG", *samples]

    async def _test():
        buffer = await spdl.io.async_load_image_batch_nvdec(
            flist,
            device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
            pix_fmt="rgba",
            width=320,
            height=240,
            strict=False,
        )

        assert buffer.__cuda_array_interface__["shape"] == (10, 4, 240, 320)

        with pytest.raises(RuntimeError):
            await spdl.io.async_load_image_batch_nvdec(
                flist,
                device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
                width=320,
                height=240,
                strict=True,
            )

    asyncio.run(_test())


def test_batch_decode_torch_allocator(get_samples):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample_%03d.jpg"
    flist = get_samples(cmd)

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

    async def _test():
        assert not allocator_called
        assert not deleter_called
        buffer = await spdl.io.async_load_image_batch_nvdec(
            flist,
            device_config=spdl.io.cuda_config(
                device_index=DEFAULT_CUDA,
                allocator=(
                    allocator,
                    deleter,
                ),
            ),
            pix_fmt="rgba",
            width=320,
            height=240,
        )
        assert buffer.__cuda_array_interface__["shape"] == (10, 4, 240, 320)

        assert allocator_called
        assert not deleter_called

    asyncio.run(_test())

    gc.collect()
    assert deleter_called
