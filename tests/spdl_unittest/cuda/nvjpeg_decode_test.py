# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from random import randbytes

import pytest
import spdl.io
import spdl.io.utils
import torch

DEFAULT_CUDA = 0


if not spdl.io.utils.is_nvjpeg_available():
    pytest.skip("SPDL is not compiled with NVJPEG support", allow_module_level=True)


def test_decode_pix_fmt(get_sample):
    """"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    def _test(data, pix_fmt):
        buffer = spdl.io.decode_image_nvjpeg(
            data,
            device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
            pix_fmt=pix_fmt,
        )
        tensor = spdl.io.to_torch(buffer)
        assert tensor.dtype == torch.uint8
        assert tensor.shape == torch.Size([3, 240, 320])
        assert tensor.device == torch.device("cuda", DEFAULT_CUDA)
        assert not torch.equal(tensor[0], tensor[1])
        assert not torch.equal(tensor[1], tensor[2])
        assert not torch.equal(tensor[2], tensor[0])
        return tensor

    rgb_tensor = _test(sample.path, "rgb")
    bgr_tensor = _test(sample.path, "bgr")

    assert torch.equal(rgb_tensor[0], bgr_tensor[2])
    assert torch.equal(rgb_tensor[1], bgr_tensor[1])
    assert torch.equal(rgb_tensor[2], bgr_tensor[0])


def test_decode_rubbish(get_samples):
    """When decoding fails, it should raise an error instead of segfault then,
    subsequent valid decodings should succeed"""

    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 10 sample_%10d.jpg"
    flist = get_samples(cmd)

    for _ in range(10):
        rubbish = randbytes(2096)
        with pytest.raises(RuntimeError):
            spdl.io.decode_image_nvjpeg(
                rubbish,
                device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
            )

    for path in flist:
        buffer = spdl.io.decode_image_nvjpeg(
            path,
            device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
        )

        tensor = spdl.io.to_torch(buffer)
        assert tensor.dtype == torch.uint8
        assert tensor.shape == torch.Size([3, 240, 320])
        assert tensor.device == torch.device("cuda", DEFAULT_CUDA)
        assert not torch.equal(tensor[0], tensor[1])
        assert not torch.equal(tensor[1], tensor[2])
        assert not torch.equal(tensor[2], tensor[0])


def test_decode_resize(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    buffer = spdl.io.decode_image_nvjpeg(
        sample.path,
        device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
        scale_width=160,
        scale_height=120,
    )
    tensor = spdl.io.to_torch(buffer)
    assert tensor.dtype == torch.uint8
    assert tensor.shape == torch.Size([3, 120, 160])
    assert tensor.device == torch.device("cuda", DEFAULT_CUDA)
    assert not torch.equal(tensor[0], tensor[1])
    assert not torch.equal(tensor[1], tensor[2])
    assert not torch.equal(tensor[2], tensor[0])


def _is_all_zero(arr):
    return all(int(v) == 0 for v in arr)


def test_decode_zero_clear(get_sample):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    with open(sample.path, "rb") as f:
        data = f.read()

    assert not _is_all_zero(data)

    buffer = spdl.io.decode_image_nvjpeg(
        data,
        device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
        scale_width=160,
        scale_height=120,
        _zero_clear=True,
    )
    tensor = spdl.io.to_torch(buffer)
    assert tensor.dtype == torch.uint8
    assert tensor.shape == torch.Size([3, 120, 160])
    assert tensor.device == torch.device("cuda", DEFAULT_CUDA)

    assert _is_all_zero(data)


def test_batch_decode_zero_clear(get_samples):
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 100 sample_%03d.jpg"
    flist = get_samples(cmd)

    dataset = []
    for path in flist:
        with open(path, "rb") as f:
            dataset.append(f.read())

    assert all(not _is_all_zero(data) for data in dataset)

    buffer = spdl.io.load_image_batch_nvjpeg(
        dataset,
        device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
        width=160,
        height=120,
        _zero_clear=True,
    )
    tensor = spdl.io.to_torch(buffer)
    assert tensor.dtype == torch.uint8
    assert tensor.shape == torch.Size([100, 3, 120, 160])
    assert tensor.device == torch.device("cuda", DEFAULT_CUDA)

    assert all(_is_all_zero(data) for data in dataset)
