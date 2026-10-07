# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from spdl.io import transfer_tensor, transfer_tensor_d2h
from spdl.io._transfer import _THREAD_LOCAL


def _clear_transfer_cache() -> None:
    """Clear all thread-local transfer caches."""
    for name in ("transfer", "d2h_transfer", "d2h_pinned_memory"):
        if hasattr(_THREAD_LOCAL, name):
            delattr(_THREAD_LOCAL, name)


class AsyncTransferD2HTest(unittest.TestCase):
    """Tests for transfer_tensor_d2h (GPU to CPU)."""

    def setUp(self) -> None:
        """Clear transfer cache before each test."""
        _clear_transfer_cache()

    def test_single_tensor(self) -> None:
        """Test D2H transfer of a single tensor."""
        ref_cuda = torch.randn(16, 3, 224, 224, device="cuda:0")
        ref_cuda_copy = ref_cuda.clone()  # Save copy to verify input is not modified
        ref_cpu = ref_cuda.cpu()
        cpu = transfer_tensor_d2h(ref_cuda)

        self.assertEqual(cpu.device.type, "cpu")
        self.assertEqual(cpu.shape, ref_cuda.shape)
        self.assertEqual(cpu.dtype, ref_cuda.dtype)
        torch.testing.assert_close(cpu, ref_cpu)

        # Verify input tensor is not modified
        self.assertTrue(torch.equal(ref_cuda, ref_cuda_copy))

    def test_large_tensor(self) -> None:
        """Test D2H transfer of a large tensor."""
        ref_cuda = torch.randint(
            256, (16, 3, 4608, 5328), dtype=torch.uint8, device="cuda"
        )
        ref_cpu = ref_cuda.cpu()
        cpu = transfer_tensor_d2h(ref_cuda)

        self.assertEqual(cpu.device.type, "cpu")
        self.assertTrue(torch.equal(cpu, ref_cpu))

    def test_various_dtypes(self) -> None:
        """Test D2H transfer with various data types."""
        dtypes = [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.int8,
            torch.float16,
        ]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                if dtype in [torch.float32, torch.float64, torch.float16]:
                    ref_cuda = torch.randn(32, 64, dtype=dtype, device="cuda")
                else:
                    ref_cuda = torch.randint(127, (32, 64), dtype=dtype, device="cuda")
                ref_cpu = ref_cuda.cpu()
                cpu = transfer_tensor_d2h(ref_cuda)

                self.assertEqual(cpu.device.type, "cpu")
                self.assertEqual(cpu.dtype, dtype)
                self.assertTrue(torch.equal(cpu, ref_cpu))

    def test_list_of_tensors(self) -> None:
        """Test D2H transfer with a list of tensors."""
        refs_cuda = [torch.randn(8, 16, device="cuda") for _ in range(4)]
        refs_cpu = [t.cpu() for t in refs_cuda]
        cpus = transfer_tensor_d2h(refs_cuda)

        self.assertIsInstance(cpus, list)
        self.assertEqual(len(cpus), len(refs_cuda))
        for cpu, ref_cpu in zip(cpus, refs_cpu, strict=True):
            self.assertEqual(cpu.device.type, "cpu")
            torch.testing.assert_close(cpu, ref_cpu)

    def test_tuple_of_tensors(self) -> None:
        """Test D2H transfer with a tuple of tensors."""
        refs_cuda = (
            torch.randn(8, 16, device="cuda"),
            torch.randn(32, 64, device="cuda"),
        )
        refs_cpu = tuple(t.cpu() for t in refs_cuda)
        cpus = transfer_tensor_d2h(refs_cuda)

        self.assertIsInstance(cpus, tuple)
        self.assertEqual(len(cpus), len(refs_cuda))
        for cpu, ref_cpu in zip(cpus, refs_cpu, strict=True):
            self.assertEqual(cpu.device.type, "cpu")
            torch.testing.assert_close(cpu, ref_cpu)

    def test_dict_of_tensors(self) -> None:
        """Test D2H transfer with a dict of tensors."""
        refs_cuda: dict[str, torch.Tensor] = {
            "image": torch.randn(4, 3, 224, 224, device="cuda"),
            "label": torch.randint(10, (4,), device="cuda"),
        }
        refs_cpu = {k: v.cpu() for k, v in refs_cuda.items()}
        cpus = transfer_tensor_d2h(refs_cuda)

        self.assertIsInstance(cpus, dict)
        self.assertEqual(set(cpus.keys()), set(refs_cuda.keys()))
        self.assertEqual(cpus["image"].device.type, "cpu")
        self.assertEqual(cpus["label"].device.type, "cpu")
        torch.testing.assert_close(cpus["image"], refs_cpu["image"])
        self.assertTrue(torch.equal(cpus["label"], refs_cpu["label"]))

    def test_nested_dict_list_structure(self) -> None:
        """Test D2H transfer with nested dict and list structure."""
        images_cuda = [torch.randn(4, 3, 224, 224, device="cuda") for _ in range(2)]
        images_cpu = [t.cpu() for t in images_cuda]
        labels_cuda = torch.randint(10, (4,), device="cuda")
        ids_cuda = torch.arange(4, device="cuda")

        refs_cuda: dict[str, Any] = {
            "images": images_cuda,
            "metadata": {
                "labels": labels_cuda,
                "ids": ids_cuda,
            },
        }
        cpus = transfer_tensor_d2h(refs_cuda)

        self.assertIsInstance(cpus, dict)
        images_result = cast(list[torch.Tensor], cpus["images"])
        self.assertIsInstance(images_result, list)
        for cpu, ref_cpu in zip(images_result, images_cpu, strict=True):
            self.assertEqual(cpu.device.type, "cpu")
            torch.testing.assert_close(cpu, ref_cpu)

        metadata_result = cast(dict[str, torch.Tensor], cpus["metadata"])
        self.assertIsInstance(metadata_result, dict)
        self.assertEqual(metadata_result["labels"].device.type, "cpu")
        self.assertEqual(metadata_result["ids"].device.type, "cpu")

    def test_dataclass(self) -> None:
        """Test D2H transfer with a dataclass."""

        @dataclass
        class Batch:
            image: torch.Tensor
            label: torch.Tensor

        ref_cuda = Batch(
            image=torch.randn(4, 3, 224, 224, device="cuda"),
            label=torch.randint(10, (4,), device="cuda"),
        )
        # Save copies to verify input is not modified
        ref_image_copy = ref_cuda.image.clone()
        ref_label_copy = ref_cuda.label.clone()

        cpu = transfer_tensor_d2h(ref_cuda)

        self.assertIsInstance(cpu, Batch)
        self.assertEqual(cpu.image.device.type, "cpu")
        self.assertEqual(cpu.label.device.type, "cpu")
        torch.testing.assert_close(cpu.image, ref_cuda.image.cpu())
        self.assertTrue(torch.equal(cpu.label, ref_cuda.label.cpu()))

        # Verify input tensors are not modified
        self.assertTrue(torch.equal(ref_cuda.image, ref_image_copy))
        self.assertTrue(torch.equal(ref_cuda.label, ref_label_copy))

    def test_empty_batch(self) -> None:
        """Test D2H transfer with empty containers."""
        result = transfer_tensor_d2h([])
        self.assertEqual(result, [])

        result = transfer_tensor_d2h({})
        self.assertEqual(result, {})

    def test_zero_sized_tensor(self) -> None:
        """Test D2H transfer of zero-sized tensors.

        Zero-sized tensors should still be transferred to the CPU,
        not returned as-is. The device of the output tensor should be CPU.
        """
        # Single zero-sized tensor
        ref = torch.randn(0, 3, 224, 224, device="cuda")
        self.assertEqual(ref.device.type, "cuda")
        cpu = transfer_tensor_d2h(ref)

        self.assertEqual(cpu.device.type, "cpu")
        self.assertEqual(cpu.shape, ref.shape)
        self.assertEqual(cpu.dtype, ref.dtype)

    def test_zero_sized_tensor_in_batch(self) -> None:
        """Test D2H transfer with zero-sized tensors mixed with regular tensors.

        All tensors including zero-sized ones should be transferred to CPU.
        """
        refs = {
            "empty": torch.randn(0, 3, 224, 224, device="cuda"),
            "normal": torch.randn(4, 3, 224, 224, device="cuda"),
        }
        cpus = transfer_tensor_d2h(refs)

        self.assertEqual(cpus["empty"].device.type, "cpu")
        self.assertEqual(cpus["empty"].shape, refs["empty"].shape)
        self.assertEqual(cpus["normal"].device.type, "cpu")
        torch.testing.assert_close(cpus["normal"], refs["normal"].cpu())

    def test_all_zero_sized_tensors(self) -> None:
        """Test D2H transfer when all tensors are zero-sized.

        Even when total data size is zero, the tensors should be transferred
        to CPU.
        """
        refs = [
            torch.randn(0, 3, 224, 224, device="cuda"),
            torch.randn(4, 0, 224, 224, device="cuda"),
            torch.randn(4, 3, 0, 224, device="cuda"),
        ]
        cpus = transfer_tensor_d2h(refs)

        self.assertIsInstance(cpus, list)
        self.assertEqual(len(cpus), len(refs))
        for cpu, ref in zip(cpus, refs, strict=True):
            self.assertEqual(cpu.device.type, "cpu")
            self.assertEqual(cpu.shape, ref.shape)
        self.assertFalse(hasattr(_THREAD_LOCAL, "d2h_pinned_memory"))

    def test_mixed_device_tensors(self) -> None:
        """Test D2H transfer with tensors on different devices.

        Only tensors on the specified device should be transferred to CPU.
        Tensors on other devices (like CPU) should be left untouched.
        """
        device = torch.device("cuda:0")
        cpu_tensor = torch.randn(4, 3, 224, 224)  # Already on CPU
        cuda_tensor = torch.randn(4, 3, 224, 224, device=device)

        refs = {
            "cpu": cpu_tensor,
            "cuda": cuda_tensor,
        }
        result = transfer_tensor_d2h(refs, device=device)

        # CPU tensor should be unchanged (same object)
        self.assertIs(result["cpu"], cpu_tensor)
        self.assertEqual(result["cpu"].device.type, "cpu")

        # CUDA tensor should be transferred to CPU (different object)
        self.assertIsNot(result["cuda"], cuda_tensor)
        self.assertEqual(result["cuda"].device.type, "cpu")
        torch.testing.assert_close(result["cuda"], cuda_tensor.cpu())

    def test_mixed_device_in_nested_structure(self) -> None:
        """Test D2H transfer with mixed device tensors in nested structure.

        Only tensors on the specified device should be transferred to CPU.
        Tensors on other devices should be left untouched, even in nested
        structures.
        """
        device = torch.device("cuda:0")
        cpu_tensor1 = torch.randn(4, 3, 224, 224)
        cpu_tensor2 = torch.randint(10, (4,))
        cuda_tensor1 = torch.randn(8, 8, device=device)
        cuda_tensor2 = torch.randn(16, 16, device=device)

        refs: dict[str, Any] = {
            "features": [cpu_tensor1, cuda_tensor1],
            "metadata": {
                "labels": cpu_tensor2,
                "scores": cuda_tensor2,
            },
        }
        result = transfer_tensor_d2h(refs, device=device)

        # CPU tensors should be unchanged (same objects)
        features_result = cast(list[torch.Tensor], result["features"])
        metadata_result = cast(dict[str, torch.Tensor], result["metadata"])

        self.assertIs(features_result[0], cpu_tensor1)
        self.assertIs(metadata_result["labels"], cpu_tensor2)

        # CUDA tensors should be transferred to CPU (different objects)
        self.assertIsNot(features_result[1], cuda_tensor1)
        self.assertEqual(features_result[1].device.type, "cpu")
        torch.testing.assert_close(features_result[1], cuda_tensor1.cpu())

        self.assertIsNot(metadata_result["scores"], cuda_tensor2)
        self.assertEqual(metadata_result["scores"].device.type, "cpu")
        torch.testing.assert_close(metadata_result["scores"], cuda_tensor2.cpu())

    def test_mixed_dtypes_use_aligned_offsets(self) -> None:
        """Test that each packed tensor starts at a dtype-aligned offset."""
        refs = [
            torch.tensor([1], dtype=torch.uint8, device="cuda"),
            torch.tensor([2.5], dtype=torch.float32, device="cuda"),
            torch.tensor([3.5], dtype=torch.float64, device="cuda"),
        ]

        result = transfer_tensor_d2h(refs)

        for actual, expected in zip(result, refs, strict=True):
            self.assertEqual(actual.dtype, expected.dtype)
            torch.testing.assert_close(actual, expected.cpu())

    def test_dataclass_init_false_field_preserves_tensor_order(self) -> None:
        """Test dataclass fields use one traversal order for packing and rebuilding."""

        @dataclass
        class Batch:
            deferred: torch.Tensor = field(init=False)
            regular: torch.Tensor

        batch = Batch(regular=torch.full((4,), 2.0, device="cuda"))
        batch.deferred = torch.full((4,), 1.0, device="cuda")

        result = transfer_tensor_d2h(batch)

        torch.testing.assert_close(result.regular, torch.full((4,), 2.0))
        torch.testing.assert_close(result.deferred, torch.full((4,), 1.0))

    def test_non_contiguous_tensor(self) -> None:
        """Test D2H transfer of a tensor whose logical view is not contiguous."""
        ref = torch.arange(24, device="cuda").reshape(4, 6).transpose(0, 1)
        self.assertFalse(ref.is_contiguous())

        result = transfer_tensor_d2h(ref)

        self.assertTrue(result.is_contiguous())
        torch.testing.assert_close(result, ref.cpu())

    def test_preserves_autograd(self) -> None:
        """Test tensors requiring gradients use a differentiable transfer path."""
        ref = torch.randn(8, device="cuda", requires_grad=True)

        result = transfer_tensor_d2h(ref)
        result.square().sum().backward()

        self.assertIsNotNone(ref.grad)
        assert ref.grad is not None
        torch.testing.assert_close(ref.grad.cpu(), 2 * ref.detach().cpu())

    def test_packed_tensors_have_independent_version_counters(self) -> None:
        """Test mutating one packed tensor does not invalidate another in autograd."""
        refs = [
            torch.zeros(8, device="cuda"),
            torch.arange(8.0, device="cuda"),
        ]
        result = transfer_tensor_d2h(refs)
        weight = torch.ones_like(result[1], requires_grad=True)
        expected = result[1].detach().clone()
        loss = (result[1] * weight).sum()

        result[0].add_(1)
        loss.backward()

        self.assertIsNotNone(weight.grad)
        assert weight.grad is not None
        torch.testing.assert_close(weight.grad, expected)

    def test_waits_for_current_producer_stream(self) -> None:
        """Test that the copy stream waits for work on the caller's stream."""
        producer = torch.cuda.Stream()
        ref = torch.zeros(1024, device="cuda")
        producer.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            ref.fill_(42)
            result = transfer_tensor_d2h(ref)

        torch.testing.assert_close(result, torch.full_like(result, 42))

    def test_unindexed_cuda_device(self) -> None:
        """Test that an unindexed CUDA device resolves successfully."""
        ref = torch.randn(8, device="cuda")

        result = transfer_tensor_d2h(ref, device="cuda")

        torch.testing.assert_close(result, ref.cpu())

    @unittest.skipUnless(torch.cuda.device_count() > 1, "requires two CUDA devices")
    def test_unindexed_cuda_device_uses_current_index(self) -> None:
        """Test that an unindexed CUDA device resolves to the current device."""
        with torch.cuda.device(1):
            ref = torch.randn(8, device="cuda")
            result = transfer_tensor_d2h(ref, device="cuda")

        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, ref.cpu())

    @unittest.skipUnless(torch.cuda.device_count() > 1, "requires two CUDA devices")
    def test_transfer_handlers_are_cached_per_device(self) -> None:
        """Test that an explicit device never reuses another device's stream."""
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]

        for device in devices:
            ref = torch.randn(8, device=device)
            result = transfer_tensor_d2h(ref, device=device)
            torch.testing.assert_close(result, ref.cpu())

        transfers = _THREAD_LOCAL.d2h_transfer
        self.assertEqual(set(transfers), set(devices))

    def test_with_custom_stream(self) -> None:
        """Test D2H transfer with a custom CUDA stream."""
        self.assertFalse(hasattr(_THREAD_LOCAL, "d2h_transfer"))

        device = torch.device("cuda:0")
        stream = torch.cuda.Stream(device)
        ref_cuda = torch.randn(16, 3, 224, 224, device=device)
        cpu = transfer_tensor_d2h(ref_cuda, device=device, stream=stream)

        self.assertEqual(cpu.device.type, "cpu")
        torch.testing.assert_close(cpu, ref_cuda.cpu())

        # When custom stream is provided, cache should NOT be populated
        # because a new handler is created for the custom stream
        self.assertFalse(hasattr(_THREAD_LOCAL, "d2h_transfer"))

    def test_cpu_tensor_passthrough(self) -> None:
        """Test that D2H transfer passes through CPU tensors unchanged."""
        ref = torch.randn(16, 3, 224, 224)  # CPU tensor
        result = transfer_tensor_d2h(ref)

        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, ref)


class PinnedMemoryCacheD2HTest(unittest.TestCase):
    """Tests for pinned-memory cache behavior for D2H transfers."""

    def setUp(self) -> None:
        """Clear transfer cache before each test."""
        _clear_transfer_cache()

    def test_d2h_cache_reuse_when_sufficient(self) -> None:
        """Test that PinnedMemory is reused when capacity is sufficient for D2H."""
        # First transfer - creates cached memory
        small_tensor = torch.randn(8, 8, device="cuda")
        transfer_tensor_d2h(small_tensor)

        self.assertTrue(hasattr(_THREAD_LOCAL, "d2h_pinned_memory"))
        initial_mem = _THREAD_LOCAL.d2h_pinned_memory
        initial_capacity = initial_mem.numel()

        # Second transfer with same or smaller size - should reuse
        same_size_tensor = torch.randn(8, 8, device="cuda")
        transfer_tensor_d2h(same_size_tensor)

        # The cached object should be the same (same id)
        self.assertIs(_THREAD_LOCAL.d2h_pinned_memory, initial_mem)
        self.assertEqual(_THREAD_LOCAL.d2h_pinned_memory.numel(), initial_capacity)

    def test_d2h_cache_reallocation_when_insufficient(self) -> None:
        """Test that new PinnedMemory is allocated when cached one is too small for D2H."""
        # First transfer with small tensor - creates small cached memory
        small_tensor = torch.randn(8, 8, device="cuda")
        small_size = small_tensor.numel() * small_tensor.element_size()
        transfer_tensor_d2h(small_tensor)

        self.assertTrue(hasattr(_THREAD_LOCAL, "d2h_pinned_memory"))
        initial_mem = _THREAD_LOCAL.d2h_pinned_memory
        initial_capacity = initial_mem.numel()
        self.assertGreaterEqual(initial_capacity, small_size)

        # Second transfer with much larger tensor - should reallocate
        large_tensor = torch.randn(256, 256, 64, device="cuda")
        large_size = large_tensor.numel() * large_tensor.element_size()
        self.assertGreater(large_size, initial_capacity)

        cpu_large = transfer_tensor_d2h(large_tensor)

        # Verify transfer worked
        self.assertEqual(cpu_large.device.type, "cpu")
        torch.testing.assert_close(cpu_large, large_tensor.cpu())

        # Verify cache was updated with new larger memory
        new_capacity = _THREAD_LOCAL.d2h_pinned_memory.numel()
        self.assertGreaterEqual(new_capacity, large_size)
        # The capacity should be larger than before
        self.assertGreater(new_capacity, initial_capacity)


class RoundTripTransferD2HTest(unittest.TestCase):
    """Tests D2H transfers of tensors produced by the existing H2D helper."""

    def setUp(self) -> None:
        """Clear transfer cache before each test."""
        _clear_transfer_cache()

    def test_h2d_then_d2h(self) -> None:
        """Test round-trip: CPU -> GPU -> CPU."""
        original = torch.randn(16, 3, 224, 224)

        # CPU -> GPU
        cuda = transfer_tensor(original)
        self.assertEqual(cuda.device.type, "cuda")

        # GPU -> CPU
        back_to_cpu = transfer_tensor_d2h(cuda)
        self.assertEqual(back_to_cpu.device.type, "cpu")

        torch.testing.assert_close(back_to_cpu, original)

    def test_round_trip_dataclass(self) -> None:
        """Test round-trip with dataclass structure."""

        @dataclass
        class Batch:
            images: list[torch.Tensor]
            labels: torch.Tensor

        original = Batch(
            images=[torch.randn(4, 3, 224, 224) for _ in range(2)],
            labels=torch.randint(10, (4,)),
        )

        # CPU -> GPU
        cuda_batch = transfer_tensor(original)
        self.assertIsInstance(cuda_batch, Batch)
        for img in cuda_batch.images:
            self.assertEqual(img.device.type, "cuda")
        self.assertEqual(cuda_batch.labels.device.type, "cuda")

        # GPU -> CPU
        cpu_batch = transfer_tensor_d2h(cuda_batch)
        self.assertIsInstance(cpu_batch, Batch)
        for i, img in enumerate(cpu_batch.images):
            self.assertEqual(img.device.type, "cpu")
            torch.testing.assert_close(img, original.images[i])
        self.assertEqual(cpu_batch.labels.device.type, "cpu")
        self.assertTrue(torch.equal(cpu_batch.labels, original.labels))

    def test_round_trip_dataclass_with_nested_dict(self) -> None:
        """Test round-trip with dataclass containing nested dict.

        This test specifically covers the crash case where a dataclass with
        multiple tensors and nested dict caused SIGSEGV during D2H transfer.
        The bug was that after H2D transfer, GPU tensors become views into a
        shared buffer. The D2H transfer code incorrectly used storage.data_ptr()
        and storage.nbytes() which returned the entire buffer instead of the
        individual tensor's data, causing memory corruption during memcpy.
        """

        @dataclass
        class ImageBatch:
            images: torch.Tensor
            labels: torch.Tensor
            metadata: dict[str, torch.Tensor]

        original = ImageBatch(
            images=torch.randn(32, 3, 224, 224),
            labels=torch.randint(0, 1000, (32,)),
            metadata={
                "ids": torch.arange(32),
                "scores": torch.rand(32),
            },
        )

        # Save copies to verify input is not modified
        original_images_copy = original.images.clone()
        original_labels_copy = original.labels.clone()
        original_ids_copy = original.metadata["ids"].clone()
        original_scores_copy = original.metadata["scores"].clone()

        # CPU -> GPU
        cuda_batch = transfer_tensor(original)
        self.assertIsInstance(cuda_batch, ImageBatch)
        self.assertEqual(cuda_batch.images.device.type, "cuda")
        self.assertEqual(cuda_batch.labels.device.type, "cuda")
        self.assertEqual(cuda_batch.metadata["ids"].device.type, "cuda")
        self.assertEqual(cuda_batch.metadata["scores"].device.type, "cuda")

        # Verify original input is not modified after H2D transfer
        self.assertTrue(torch.equal(original.images, original_images_copy))
        self.assertTrue(torch.equal(original.labels, original_labels_copy))
        self.assertTrue(torch.equal(original.metadata["ids"], original_ids_copy))
        self.assertTrue(torch.equal(original.metadata["scores"], original_scores_copy))

        # Save copies of CUDA tensors to verify they are not modified
        cuda_images_copy = cuda_batch.images.clone()
        cuda_labels_copy = cuda_batch.labels.clone()
        cuda_ids_copy = cuda_batch.metadata["ids"].clone()
        cuda_scores_copy = cuda_batch.metadata["scores"].clone()

        # GPU -> CPU - this is where the crash occurred before the fix
        cpu_batch = transfer_tensor_d2h(cuda_batch)
        self.assertIsInstance(cpu_batch, ImageBatch)

        # Verify CUDA input is not modified after D2H transfer
        self.assertTrue(torch.equal(cuda_batch.images, cuda_images_copy))
        self.assertTrue(torch.equal(cuda_batch.labels, cuda_labels_copy))
        self.assertTrue(torch.equal(cuda_batch.metadata["ids"], cuda_ids_copy))
        self.assertTrue(torch.equal(cuda_batch.metadata["scores"], cuda_scores_copy))

        # Verify all tensors are on CPU and data is correct
        self.assertEqual(cpu_batch.images.device.type, "cpu")
        self.assertEqual(cpu_batch.labels.device.type, "cpu")
        self.assertEqual(cpu_batch.metadata["ids"].device.type, "cpu")
        self.assertEqual(cpu_batch.metadata["scores"].device.type, "cpu")

        torch.testing.assert_close(cpu_batch.images, original.images)
        self.assertTrue(torch.equal(cpu_batch.labels, original.labels))
        self.assertTrue(
            torch.equal(cpu_batch.metadata["ids"], original.metadata["ids"])
        )
        self.assertTrue(
            torch.allclose(cpu_batch.metadata["scores"], original.metadata["scores"])
        )
