# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""H2D error-path ownership with ordinary Torch storage and simulated CUDA streams."""

from __future__ import annotations

import gc
import threading
import unittest
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import spdl.io._transfer as module
import torch


class _Stream:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.failures: list[BaseException | None] = []

    def synchronize(self) -> None:
        self.events.append("synchronize")
        if self.failures and (error := self.failures.pop(0)) is not None:
            raise error


class _Cuda:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.transfer_stream = _Stream(self.events)
        self.current_device: object = "cuda:0"

    def Stream(self, device: object) -> _Stream:
        return self.transfer_stream

    def stream(self, stream: object):
        return nullcontext()

    @contextmanager
    def device(self, device: object):
        previous = self.current_device
        self.current_device = device
        try:
            yield
        finally:
            self.current_device = previous

    def default_stream(self, device: object) -> object:
        return ("default", device)


class _Tensor:
    def __init__(self, cuda: _Cuda, *, owner: object = None, cpu: bool = True) -> None:
        self.cuda = cuda
        self.owner = owner
        self.is_cpu = cpu
        self.device: object = "cpu" if cpu else "cuda:1"
        self.fail_pin = False
        self.pinned_on: object = None
        self.recorded_on: object = None

    def pin_memory(self) -> _Tensor:
        if self.fail_pin:
            raise ValueError("traversal root")
        self.pinned_on = self.cuda.current_device
        return self

    def to(self, device: object, *, non_blocking: bool) -> _Tensor:
        result = _Tensor(self.cuda, cpu=False)
        result.device = device
        self.cuda.events.append("enqueue")
        return result

    def record_stream(self, stream: object) -> None:
        self.recorded_on = stream
        self.cuda.events.append("record_default")


class H2DTransferLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cuda = _Cuda()
        self.initial_entries = tuple(module._FAILED_H2D)
        self.enterContext(
            patch.object(
                module, "torch", SimpleNamespace(Tensor=_Tensor, cuda=self.cuda)
            )
        )
        self.transfer = module._DataTransfer(cast(Any, "cuda:1"), 0)
        self.addCleanup(self._cleanup_completed_test_transfers)

    def _cleanup_completed_test_transfers(self) -> None:
        self.cuda.transfer_stream.failures.clear()
        self.cuda.transfer_stream.synchronize()
        with module._FAILED_H2D_LOCK:
            module._FAILED_H2D[:] = list(self.initial_entries)

    def test_success_selects_device_and_records_default_consumer(self) -> None:
        """Successful uncached transfer finishes before recording destination lifetime."""
        source = _Tensor(self.cuda, owner=torch.arange(16))
        result = cast(_Tensor, self.transfer(source))
        self.assertEqual(source.pinned_on, "cuda:1")
        self.assertEqual(result.recorded_on, ("default", "cuda:1"))
        self.assertEqual(self.cuda.events, ["enqueue", "synchronize", "record_default"])
        self.assertEqual(self.transfer._batch_cache, [])

    def test_later_leaf_failure_drains_before_preserving_primary_error(self) -> None:
        """Earlier copies complete before a later recursive leaf exception unwinds."""
        first, later = _Tensor(self.cuda), _Tensor(self.cuda)
        later.fail_pin = True
        with self.assertRaisesRegex(ValueError, "traversal root"):
            self.transfer([first, later])
        self.assertEqual(self.cuda.events, ["enqueue", "synchronize"])
        self.assertIsNone(self.transfer._pending)

    def test_failed_sync_retains_ordinary_storage_until_successful_retry(self) -> None:
        """Unknown completion retains ordinary Torch source storage and destinations."""
        storage = torch.arange(64, dtype=torch.uint8)
        storage_ref = weakref.ref(storage)
        source = _Tensor(self.cuda, owner=storage)
        source_ref = weakref.ref(source)
        self.cuda.transfer_stream.failures = [RuntimeError("synchronize root")]
        with self.assertRaisesRegex(RuntimeError, "synchronize root"):
            self.transfer(source)
        pending = self.transfer._pending
        self.assertIsNotNone(pending)
        assert pending is not None
        destination_ref = weakref.ref(next(iter(pending.destinations)))
        del source, storage, pending
        gc.collect()
        self.assertIsNotNone(source_ref())
        self.assertIsNotNone(storage_ref())
        self.assertIsNotNone(destination_ref())
        self.transfer([])
        gc.collect()
        self.assertIsNone(source_ref())
        self.assertIsNone(storage_ref())
        self.assertIsNone(destination_ref())
        self.assertIsNone(self.transfer._pending)

    def test_secondary_sync_failure_does_not_replace_traversal_root(self) -> None:
        """A failed failure-drain preserves the original traversal exception and owners."""
        first, later = _Tensor(self.cuda), _Tensor(self.cuda)
        later.fail_pin = True
        self.cuda.transfer_stream.failures = [RuntimeError("secondary sync")]
        with self.assertRaisesRegex(ValueError, "traversal root") as caught:
            self.transfer([first, later])
        self.assertIn("secondary sync", caught.exception.__notes__[0])
        self.assertIsNotNone(self.transfer._pending)
        self.transfer([])
        self.assertIsNone(self.transfer._pending)

    def test_cached_public_retries_do_not_admit_new_copies(self) -> None:
        """Cached retries retain one failed batch and reject new pin/enqueue work."""
        storage = torch.arange(64)
        storage_ref = weakref.ref(storage)
        source = _Tensor(self.cuda, owner=storage)
        self.enterContext(patch.object(module, "_THREAD_LOCAL", threading.local()))
        self.enterContext(
            patch.object(module, "_normalize_cuda_device", return_value="cuda:1")
        )
        self.cuda.transfer_stream.failures = [RuntimeError("initial sync failure")]
        with self.assertRaisesRegex(RuntimeError, "initial sync failure"):
            module.transfer_tensor_h2d(source, num_caches=0)
        transfer = module._get_transfer_func(0)
        pending = transfer._pending
        self.assertIsNotNone(pending)
        assert pending is not None
        destination_ref = weakref.ref(next(iter(pending.destinations)))
        del source, storage, pending
        for attempt in range(3):
            rejected = _Tensor(self.cuda)
            error = RuntimeError(f"retry sync {attempt}")
            self.cuda.transfer_stream.failures = [error]
            with self.assertRaises(RuntimeError) as caught:
                module.transfer_tensor_h2d(rejected, num_caches=0)
            self.assertIs(caught.exception, error)
            self.assertIs(module._get_transfer_func(0), transfer)
            self.assertIsNone(rejected.pinned_on)
            self.assertEqual(self.cuda.events.count("enqueue"), 1)
            self.assertEqual(len(module._FAILED_H2D), len(self.initial_entries) + 1)
            self.assertIsNotNone(storage_ref())
            self.assertIsNotNone(destination_ref())
        self.assertEqual(module.transfer_tensor_h2d([], num_caches=0), [])
        gc.collect()
        self.assertIsNone(transfer._pending)
        self.assertIsNone(storage_ref())
        self.assertIsNone(destination_ref())
        self.assertEqual(tuple(module._FAILED_H2D), self.initial_entries)

    def test_quarantine_outlives_thread_local_helper(self) -> None:
        """Destroying a failed helper cannot drop unconfirmed ordinary source storage."""
        storage = torch.arange(8)
        reference = weakref.ref(storage)
        source = _Tensor(self.cuda, owner=storage)
        self.cuda.transfer_stream.failures = [RuntimeError("sync root")]
        with self.assertRaisesRegex(RuntimeError, "sync root"):
            self.transfer(source)
        pending = self.transfer._pending
        assert pending is not None
        transfer_ref = weakref.ref(self.transfer)
        del self.transfer, source, storage
        gc.collect()
        self.assertIsNone(transfer_ref())
        self.assertIsNotNone(reference())
        self.assertIn(pending, module._FAILED_H2D)
        self.cuda.transfer_stream.synchronize()
        module._FAILED_H2D.remove(pending)
        del pending
        gc.collect()
        self.assertIsNone(reference())

    def test_negative_cache_count_fails_before_stream_creation(self) -> None:
        """An invalid cache count does not allocate a stream or enqueue a transfer."""
        with patch.object(self.cuda, "Stream", wraps=self.cuda.Stream) as create:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                module._DataTransfer(cast(Any, "cuda:1"), -1)
        create.assert_not_called()

    def test_cached_helper_follows_normalized_device(self) -> None:
        """Changing the normalized CUDA device replaces only the current helper."""
        self.enterContext(patch.object(module, "_THREAD_LOCAL", threading.local()))
        with patch.object(
            module, "_normalize_cuda_device", side_effect=["cuda:1", "cuda:2"]
        ):
            first = module._get_transfer_func(0)
            second = module._get_transfer_func(0)
        self.assertIsNot(first, second)
        self.assertEqual(first._device, "cuda:1")
        self.assertEqual(second._device, "cuda:2")
