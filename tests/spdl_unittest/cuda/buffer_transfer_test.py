# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import unittest

import numpy as np
import spdl.io
import spdl.io.utils
import torch
from parameterized import parameterized

from ..fixture import FFMPEG_CLI, get_sample

if not spdl.io.utils.built_with_cuda():
    raise unittest.SkipTest("SPDL is not compiled with CUDA support")


DEFAULT_CUDA = 0

CMDS = {
    "audio": f'{FFMPEG_CLI} -hide_banner -y -f lavfi -i "sine=frequency=1000:sample_rate=48000:duration=3" -c:a pcm_s16le sample.wav',
    "video": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4",
    "image": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


class TestTransferBufferToCuda(unittest.TestCase):
    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_transfer_buffer_to_cuda(self, media_type: str) -> None:
        """smoke test for transfer_buffer_to_cuda function"""
        cmd = CMDS[media_type]
        sample = get_sample(cmd)

        demux_func = {
            "audio": spdl.io.demux_audio,
            "video": spdl.io.demux_video,
            "image": spdl.io.demux_image,
        }[media_type]

        _ = torch.zeros([0], device=torch.device(f"cuda:{DEFAULT_CUDA}"))

        packets = demux_func(sample.path)
        # pyre-ignore[6]: Union type from demux_func is valid for decode_packets
        frames = spdl.io.decode_packets(packets)
        buffer = spdl.io.convert_frames(frames)
        cpu_tensor = spdl.io.to_torch(buffer).clone()

        cuda_tensor = spdl.io.to_torch(
            spdl.io.transfer_buffer(
                buffer,
                device_config=spdl.io.cuda_config(
                    device_index=DEFAULT_CUDA,
                ),
            )
        )

        self.assertTrue(cuda_tensor.is_cuda)
        self.assertEqual(cuda_tensor.device, torch.device(f"cuda:{DEFAULT_CUDA}"))

        self.assertTrue(torch.allclose(cpu_tensor, cuda_tensor.cpu()))

    @parameterized.expand(
        [
            ("audio",),
            ("video",),
            ("image",),
        ]
    )
    def test_transfer_buffer_to_cuda_with_pytorch_allocator(
        self, media_type: str
    ) -> None:
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
            "audio": spdl.io.demux_audio,
            "video": spdl.io.demux_video,
            "image": spdl.io.demux_image,
        }[media_type]

        def test():
            packets = demux_func(sample.path)
            frames = spdl.io.decode_packets(packets)
            buffer = spdl.io.convert_frames(frames)
            cpu_tensor = spdl.io.to_torch(buffer).clone()

            print("Asserting allocator was not yet called")
            self.assertFalse(allocator_called)
            print("Transferring")
            cuda_buffer = spdl.io.transfer_buffer(
                buffer,
                device_config=spdl.io.cuda_config(
                    device_index=DEFAULT_CUDA,
                    allocator=(allocator, deleter),
                ),
            )
            print("Asserting allocator was called")
            self.assertTrue(allocator_called)

            cuda_tensor = spdl.io.to_torch(cuda_buffer)
            self.assertTrue(cuda_tensor.is_cuda)
            self.assertEqual(cuda_tensor.device, torch.device(f"cuda:{DEFAULT_CUDA}"))
            self.assertTrue(torch.allclose(cpu_tensor, cuda_tensor.cpu()))

            print("Asserting deleter was not yet called")
            self.assertFalse(deleter_called)

        test()

        print("Calling GC")
        gc.collect()
        print("Asserting deleter was called")
        self.assertTrue(deleter_called)


class TestArrayTransfer(unittest.TestCase):
    def test_array_transfer_numpy(self) -> None:
        """smoke test for transfer_buffer function"""

        dtypes = {
            np.dtype("uint8"): torch.uint8,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
        }

        def test(array):
            buffer = spdl.io.transfer_buffer(
                array, device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA)
            )
            tensor = spdl.io.to_torch(buffer)

            device = torch.device(f"cuda:{DEFAULT_CUDA}")
            self.assertEqual(tensor.dtype, dtypes[array.dtype])
            self.assertEqual(tensor.shape, torch.Size(array.shape))
            self.assertEqual(tensor.device, torch.device(device))
            self.assertTrue(torch.allclose(tensor, torch.from_numpy(array).to(device)))

        for dtype in [np.uint8, np.int32, np.int64]:
            max_val = np.iinfo(dtype).max
            array = np.random.randint(0, max_val, size=(1, 128_000), dtype=dtype)
            test(array)

    def test_array_transfer_torch(self) -> None:
        """smoke test for transfer_buffer function"""

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        def test(cpu_tensor):
            buffer = spdl.io.transfer_buffer(cpu_tensor, device_config=device_config)
            cuda_tensor = spdl.io.to_torch(buffer)

            device = torch.device(f"cuda:{DEFAULT_CUDA}")
            self.assertEqual(cuda_tensor.dtype, cpu_tensor.dtype)
            self.assertEqual(cuda_tensor.shape, cpu_tensor.shape)
            self.assertEqual(cuda_tensor.device, device)
            self.assertTrue(torch.allclose(cuda_tensor, cpu_tensor.to(device)))

        for dtype in [np.uint8, np.int32, np.int64]:
            max_val = np.iinfo(dtype).max
            array = np.random.randint(0, max_val, size=(1, 128_000), dtype=dtype)
            test(torch.from_numpy(array))

    def test_array_transfer_non_contiguous_torch(self) -> None:
        """passing noncontiguous array/tensor to transfer_buffer works."""

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        cpu_tensor = torch.arange(24).reshape(6, 4).T[::2, :]
        self.assertFalse(cpu_tensor.is_contiguous())
        # pyre-ignore[6]: Tensor is a valid input type for transfer_buffer
        buffer = spdl.io.transfer_buffer(cpu_tensor, device_config=device_config)
        cuda_tensor = spdl.io.to_torch(buffer)

        device = torch.device(f"cuda:{DEFAULT_CUDA}")
        self.assertEqual(cuda_tensor.dtype, cpu_tensor.dtype)
        self.assertEqual(cuda_tensor.shape, cpu_tensor.shape)
        self.assertEqual(cuda_tensor.device, device)
        self.assertTrue(torch.allclose(cuda_tensor, cpu_tensor.to(device)))

    def test_array_transfer_non_contiguous_numpy(self) -> None:
        """passing noncontiguous array/tensor to transfer_buffer works"""

        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        array0 = np.arange(24)
        self.assertTrue(array0.data.contiguous)
        arr = array0.reshape(6, 4).T[:, ::2]
        self.assertFalse(arr.data.contiguous)
        # pyre-ignore[6]: ndarray is a valid input type for transfer_buffer
        buffer = spdl.io.transfer_buffer(arr, device_config=device_config)
        tensor = spdl.io.to_torch(buffer)

        device = torch.device(f"cuda:{DEFAULT_CUDA}")
        self.assertEqual(tensor.dtype, torch.int64)
        self.assertEqual(tensor.shape, torch.Size(arr.shape))
        self.assertEqual(tensor.device, torch.device(device))
        self.assertTrue(torch.allclose(tensor, torch.from_numpy(arr).to(device)))

    def test_array_transfer_smoke_test(self) -> None:
        """smoke test for transferring multiple arrays concurrently"""

        array = np.random.randint(0, 256, size=(1, 64_000), dtype=np.uint8)
        device_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        for _ in range(100):
            spdl.io.transfer_buffer(array, device_config=device_config)


class TestTransferCpu(unittest.TestCase):
    def test_transfer_cpu(self) -> None:
        """smoke test for transfer_buffer function"""

        for dtype in [np.uint8, np.int32, np.int64]:
            max_val = np.iinfo(dtype).max
            array = np.random.randint(0, max_val, size=(1, 128_000), dtype=dtype)
            ref = torch.from_numpy(array)
            cuda_tensor = ref.cuda(device=DEFAULT_CUDA)
            # pyre-ignore[6]: Tensor is a valid input type for transfer_buffer_cpu
            buffer = spdl.io.transfer_buffer_cpu(cuda_tensor)
            cpu_tensor = spdl.io.to_torch(buffer)

            self.assertEqual(ref.dtype, cpu_tensor.dtype)
            self.assertEqual(ref.shape, cpu_tensor.shape)
            self.assertEqual(ref.device, cpu_tensor.device)
            self.assertTrue(torch.allclose(ref, cpu_tensor))
