# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from spdl.pipeline._arena._pool import SharedMemorySegmentPool


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


class H2DLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cuda = _Cuda()
        fake = SimpleNamespace(Tensor=_Tensor, cuda=self.cuda)
        self.patch = patch.object(module, "torch", fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.transfer = module._DataTransfer(cast(Any, "cuda:1"), 0)

    def test_persistent_sync_failure_blocks_new_cached_public_transfers(self) -> None:
        """Same-helper retries retain one pending batch and admit no new CUDA copies."""
        pool = SharedMemorySegmentPool(64, 1)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 64)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        source = _Tensor(self.cuda, owner=view)
        source_ref = weakref.ref(source)
        initial_entries = tuple(module._FAILED_H2D)
        self.enterContext(patch.object(module, "_THREAD_LOCAL", threading.local()))
        self.enterContext(
            patch.object(module, "_normalize_cuda_device", return_value="cuda:1")
        )
        stream_factory = self.enterContext(
            patch.object(self.cuda, "Stream", wraps=self.cuda.Stream)
        )
        self.cuda.transfer_stream.failures = [RuntimeError("initial sync failure")]
        with self.assertRaisesRegex(RuntimeError, "initial sync failure"):
            module.transfer_tensor_h2d(source, num_caches=0)
        transfer = module._get_transfer_func(0)
        pending = transfer._pending
        self.assertIsNotNone(pending)
        self.assertEqual(len(pending.sources), 1)
        self.assertEqual(len(pending.destinations), 1)
        destination_ref = weakref.ref(next(iter(pending.destinations)))
        self.assertIs(pending.stream, self.cuda.transfer_stream)
        self.assertEqual(tuple(module._FAILED_H2D), (*initial_entries, pending))
        self.assertEqual(transfer._batch_cache, [])
        del source, view, pending
        pool.close()
        pool.unlink()
        gc.collect()
        for attempt in range(5):
            rejected = _Tensor(self.cuda)
            error = RuntimeError(f"retry sync {attempt}")
            self.cuda.transfer_stream.failures = [error]
            with self.assertRaises(RuntimeError) as raised:
                module.transfer_tensor_h2d(rejected, num_caches=0)
            self.assertIs(raised.exception, error)
            self.assertIs(module._get_transfer_func(0), transfer)
            self.assertIsNone(rejected.pinned_on)
            self.assertEqual(self.cuda.events.count("enqueue"), 1)
            self.assertEqual(len(module._FAILED_H2D), len(initial_entries) + 1)
            self.assertIs(module._FAILED_H2D[-1], transfer._pending)
            self.assertIsNotNone(source_ref())
            self.assertIsNotNone(destination_ref())
            self.assertIsNotNone(pool._segs)
            gc.collect()
        stream_factory.assert_called_once()
        self.cuda.transfer_stream.failures = []
        self.assertEqual(module.transfer_tensor_h2d([], num_caches=0), [])
        gc.collect()
        self.assertEqual(tuple(module._FAILED_H2D), initial_entries)
        self.assertIsNone(transfer._pending)
        self.assertIsNone(source_ref())
        self.assertIsNone(destination_ref())
        self.assertIsNone(pool._segs)
        self.assertEqual(self.cuda.events.count("enqueue"), 1)

    def test_failed_sync_keeps_external_storage_until_proven_retry(self) -> None:
        """Unknown DMA completion retains shared storage independently of dropped inputs."""
        pool = SharedMemorySegmentPool(64, 1)
        self.addCleanup(pool.unlink)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        off, size = writer.write_binary(b"a" * 64)
        span = writer.commit_unit()
        view = reader.read_binary(off, size)
        reader.end_unit(span, [])
        source = _Tensor(self.cuda, owner=view)
        reference = weakref.ref(source)
        self.cuda.transfer_stream.failures = [
            RuntimeError("synchronize root"),
            None,
            None,
        ]
        with self.assertRaisesRegex(RuntimeError, "synchronize root"):
            self.transfer(source)
        pending = self.transfer._pending
        self.assertIsNotNone(pending)
        self.assertIn(pending, module._FAILED_H2D)
        del source, view
        pool.close()
        pool.unlink()
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertIsNotNone(pool._segs)
        del pending
        self.transfer([])
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(pool._segs)
        self.assertIsNone(self.transfer._pending)
