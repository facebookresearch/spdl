# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest
import spdl.io
import spdl.io.utils
import torch

DEFAULT_CUDA = 0

if not spdl.io.utils.is_nvcodec_available():
    pytest.skip("SPDL is not compiled with NVCODEC support", allow_module_level=True)


def test_decode_video_nvdec(get_sample):
    """Can decode video with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    timestamps = [(i, i + 1) for i in range(10)]

    demuxer = spdl.io.Demuxer(sample.path)
    for ts in timestamps:
        packets = demuxer.demux_video(ts)
        print(packets)
        buffer = spdl.io.decode_packets_nvdec(
            packets,
            device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
        )

        tensor = spdl.io.to_torch(buffer)
        print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")


def _decode_image(path):
    packets = spdl.io.demux_image(path)
    print(packets)
    frames = spdl.io.decode_packets_nvdec(
        packets, device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA)
    )
    print(frames)
    return frames


def test_decode_image_nvdec(get_sample):
    """Can decode image with NVDEC"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    frames = _decode_image(sample.path)
    tensor = spdl.io.to_torch(frames)
    print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")
    assert tensor.shape == torch.Size([4, 240, 320])
    assert tensor.dtype == torch.uint8
    assert tensor.device == torch.device("cuda", DEFAULT_CUDA)


def test_batch_decode_image(get_samples):
    """batch loading can handle non-existing file."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG", *samples]

    buffer = spdl.io.load_image_batch_nvdec(
        flist,
        device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
        pix_fmt="rgba",
        width=320,
        height=240,
        strict=False,
    )

    assert buffer.__cuda_array_interface__["shape"] == (10, 4, 240, 320)

    with pytest.raises(RuntimeError):
        spdl.io.load_image_batch_nvdec(
            flist,
            device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
            width=320,
            height=240,
            strict=True,
        )


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

    def test():
        assert not allocator_called
        assert not deleter_called
        buffer = spdl.io.load_image_batch_nvdec(
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

    test()

    gc.collect()
    assert deleter_called
