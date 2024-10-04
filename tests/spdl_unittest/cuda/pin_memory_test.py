# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import spdl.io
import spdl.lib
import torch


@pytest.mark.parametrize(
    "pin_memory",
    [
        True,
        False,
    ],
)
def test_pin_memory_smoke_test(get_sample, pin_memory):
    """can obtain pinned memory storage"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd, width=320, height=240)

    packets = spdl.io.demux_image(sample.path)
    frames = spdl.io.decode_packets(packets)

    size = frames.width * frames.height * 3
    storage = spdl.lib._libspdl.cpu_storage(size, pin_memory=pin_memory)

    buffer = spdl.io.convert_frames(frames, storage=storage)
    stream = torch.cuda.Stream(device=0)
    cuda_config = spdl.io.cuda_config(device_index=0, stream=stream.cuda_stream)
    buffer = spdl.io.transfer_buffer(buffer, cuda_config=cuda_config)
    tensor = spdl.io.to_torch(buffer)

    print(tensor.device)


def test_pin_memory_small(get_sample):
    """convert_frames fails if the size of memory region is too small"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv422p -frames:v 1 sample.jpeg"
    sample = get_sample(cmd, width=320, height=240)

    packets = spdl.io.demux_image(sample.path)
    frames = spdl.io.decode_packets(packets)

    storage = spdl.lib._libspdl.cpu_storage(1, pin_memory=False)

    with pytest.raises(RuntimeError):
        spdl.io.convert_frames(frames, storage=storage)


@pytest.mark.parametrize(
    "size,pin_memory",
    [
        (0, True),
        (-1, False),
        (0, True),
        (-1, False),
    ],
)
def test_pin_memory_invalid_size(size, pin_memory):
    """convert_frames fails if size is invalid."""
    with pytest.raises(RuntimeError):
        spdl.lib._libspdl.cpu_storage(size, pin_memory=pin_memory)
