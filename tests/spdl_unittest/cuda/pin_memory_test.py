# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import spdl.io
import torch

from ..fixture import FFMPEG_CLI, get_sample


@pytest.mark.parametrize(
    "pin_memory",
    [
        True,
        False,
    ],
)
def test_pin_memory_smoke_test(pin_memory):
    """can obtain pinned memory storage"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd)

    packets = spdl.io.demux_image(sample.path)
    frames = spdl.io.decode_packets(packets)

    size = frames.width * frames.height * 3
    storage = spdl.io.cpu_storage(size, pin_memory=pin_memory)

    buffer = spdl.io.convert_frames(frames, storage=storage)
    stream = torch.cuda.Stream(device=0)
    cuda_config = spdl.io.cuda_config(device_index=0, stream=stream.cuda_stream)
    buffer = spdl.io.transfer_buffer(buffer, cuda_config=cuda_config)
    tensor = spdl.io.to_torch(buffer)

    print(tensor.device)


def test_pin_memory_small():
    """convert_frames fails if the size of memory region is too small"""
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd)

    packets = spdl.io.demux_image(sample.path)
    frames = spdl.io.decode_packets(packets)
    storage = spdl.io.cpu_storage(1, pin_memory=False)

    with pytest.raises(RuntimeError):
        spdl.io.convert_frames(frames, storage=storage)


@pytest.mark.parametrize(
    "pin_memory",
    [
        (True),
        (False),
    ],
)
def test_pin_memory_invalid_size(pin_memory):
    """convert_frames fails if size is invalid."""
    with pytest.raises(RuntimeError):
        spdl.io.cpu_storage(0, pin_memory=pin_memory)

    with pytest.raises(TypeError):
        spdl.io.cpu_storage(-1, pin_memory=pin_memory)


def test_pin_memory_convert_array():
    """convert_array can handle pinned memory."""
    vals = np.arange(0, 10)
    storage = spdl.io.cpu_storage(vals.nbytes, pin_memory=True)
    buffer = spdl.io.convert_array(vals, storage=storage)
    buffer = spdl.io.transfer_buffer(
        buffer,
        device_config=spdl.io.cuda_config(device_index=0),
    )
    tensor = spdl.io.to_torch(buffer)

    assert tensor.shape == (10,)
    assert tensor.dtype == torch.int64
    assert tensor.device == torch.device("cuda:0")
    assert (vals == tensor.cpu().numpy()).all()


def test_pin_memory_convert_array_invalid_size():
    """convert_array fails if storage is small."""
    vals = np.arange(0, 10)
    storage = spdl.io.cpu_storage(vals.nbytes // 2, pin_memory=True)
    with pytest.raises(RuntimeError):
        spdl.io.convert_array(vals, storage=storage)
