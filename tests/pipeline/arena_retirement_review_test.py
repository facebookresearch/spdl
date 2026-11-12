# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import threading
import unittest
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Any
from unittest.mock import patch

from parameterized import parameterized
from spdl.pipeline import SharedMemoryRingBuffer, SharedMemorySegmentPool
from spdl.pipeline._arena import _pool, _ring


def _arena(kind: str, *, timeout: float = 0) -> Any:
    return (
        SharedMemorySegmentPool(64, 1, acquire_timeout=timeout)
        if kind == "pool"
        else SharedMemoryRingBuffer(64, acquire_timeout=timeout)
    )


def _holder(owner: Any) -> Any:
    holder = object.__new__(type(owner))
    holder.__setstate__(owner.__getstate__())
    return holder


class ArenaRetirementReviewTest(unittest.TestCase):
    @parameterized.expand([("pool",), ("ring",)])
    def test_export_blocked_retirement_keeps_names_and_supports_safe_retry(
        self, kind: str
    ) -> None:
        """Export-blocked retirement retains names and retries safely after release."""
        arena = _arena(kind)
        name = arena._ctrl_name if kind == "pool" else arena._shm_name
        mapped = arena._ctrl if kind == "pool" else arena._shm
        export = memoryview(mapped.buf)
        try:
            self.assertIsNone(arena.close())
            self.assertIsNone(arena.shutdown_arena())
            with self.assertRaises(BufferError):
                arena.unlink()
            attachment = shared_memory.SharedMemory(name=name)
            attachment.close()
            self.assertGreater(export.nbytes, 0)
            export.release()
            arena.close()
            arena.unlink()
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)
        finally:
            export.release()
            arena.close()
            arena.unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_empty_copied_handle_closes_after_owner_unlink(self, kind: str) -> None:
        """An unused local copy closes without reattaching removed creator names."""
        owner = _arena(kind)
        holder = _holder(owner)
        owner.close()
        owner.unlink()
        self.assertIsNone(holder.close())
        self.assertIsNone(holder.shutdown_arena())
        self.assertTrue(holder._closed)

    @parameterized.expand([("pool",), ("ring",)])
    def test_completed_shutdown_does_not_reacquire_control(self, kind: str) -> None:
        """Repeated shutdown on a retired handle never reaccesses released control."""
        owner = _arena(kind)
        try:
            owner.close()
            with patch.object(
                owner, "_condition", side_effect=AssertionError("retired")
            ):
                self.assertIsNone(owner.shutdown_arena())
                self.assertIsNone(owner.shutdown_arena())
            self.assertTrue(owner._closed)
        finally:
            owner.unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_local_reader_close_preserves_unit_accounting_and_owner_capacity(
        self, kind: str
    ) -> None:
        """Local close preserves final unit accounting and another handle’s capacity."""
        owner = _arena(kind)
        holder = _holder(owner)
        writer, reader = owner.open_writer(), holder.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"old!" * 16)
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        try:
            holder.close()
            self.assertFalse(owner._shutdown)
            reader.end_unit(span)
            self.assertEqual(bytes(view), b"old!" * 16)
            view.release()
            holder.close()
            writer.begin_unit()
            next_offset, next_size = writer.write_binary(b"next")
            next_span = writer.commit_unit()
            next_reader = owner.open_reader()
            next_view = next_reader.read_binary(next_offset, next_size)
            next_reader.end_unit(next_span)
            self.assertEqual(bytes(next_view), b"next")
            next_view.release()
        finally:
            view.release()
            holder.close()
            owner.close()
            owner.unlink()

    def test_rejected_ring_read_does_not_block_retirement(self) -> None:
        """A rejected read records no unit, so close retires and unlink succeeds."""
        ring = _arena("ring")
        reader = ring.open_reader()
        with self.assertRaises(ValueError):
            reader.read_binary(ring._capacity, 1)
        self.assertFalse(reader.pending_unit)
        ring.close()
        self.assertTrue(ring._closed)
        ring.unlink()

    def test_rejected_ring_read_keeps_accepted_unit_pending(self) -> None:
        """A failed read cannot clear an accepted unit that still holds the ring."""
        ring = _arena("ring")
        writer, reader = ring.open_writer(), ring.open_reader()
        try:
            writer.begin_unit()
            offset, size = writer.write_binary(b"unit")
            span = writer.commit_unit()
            self.assertEqual(bytes(reader.read_binary(offset, size)), b"unit")
            with self.assertRaises(ValueError):
                reader.read_binary(ring._capacity, 1)
            self.assertTrue(reader.pending_unit)
            ring.close()
            self.assertFalse(ring._closed)
            with self.assertRaises(BufferError):
                ring.unlink()
            reader.end_unit(span)
            self.assertFalse(reader.pending_unit)
            ring.close()
            self.assertTrue(ring._closed)
        finally:
            ring.close()
            ring.unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_failed_copied_close_does_not_cancel_shared_owner(self, kind: str) -> None:
        """A failed local retirement leaves shared creator work uncancelled."""
        owner = _arena(kind)
        holder = _holder(owner)
        holder.open_reader()
        held, release = threading.Event(), threading.Event()

        def occupy() -> None:
            with owner._space_cv:
                held.set()
                release.wait(10)

        worker = threading.Thread(target=occupy)
        worker.start()
        module = _pool if kind == "pool" else _ring
        try:
            self.assertTrue(held.wait(10))
            with patch.object(module, "_CONTROL_LOCK_TIMEOUT", 0.02):
                with self.assertRaises(RuntimeError):
                    holder.close()
            self.assertFalse(owner._shutdown)
            release.set()
            worker.join(10)
            writer = owner.open_writer()
            writer.begin_unit()
            self.assertEqual(writer.write_binary(b"live"), (0, 4))
            writer.abort_unit()
        finally:
            release.set()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            # The test-owned lock is known healthy again; no production error
            # reset is implied by clearing an older implementation's test state.
            holder._retirement_error = None
            holder.close()
            owner.close()
            owner.unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_two_close_callers_recheck_completed_retirement(self, kind: str) -> None:
        """Concurrent close callers recheck physical retirement under the final lock."""
        owner = _arena(kind)
        entered, resume = threading.Event(), threading.Event()
        failures: list[Exception] = []
        original = owner._condition

        def finish() -> None:
            try:
                owner.close()
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=finish)

        @contextmanager
        def delayed(**kwargs: Any):
            if threading.current_thread() is worker:
                entered.set()
                self.assertTrue(resume.wait(10))
            with original(**kwargs):
                yield

        try:
            with patch.object(owner, "_condition", delayed):
                worker.start()
                self.assertTrue(entered.wait(10))
                owner.close()
                resume.set()
                worker.join(10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(owner._closed)
        finally:
            resume.set()
            worker.join(10)
            owner.close()
            owner.unlink()

    def test_shutdown_rechecks_export_blocked_retirement(self) -> None:
        """Shutdown rechecks the no-control state after overlapping deferred close."""
        owner = _arena("ring")
        export = memoryview(owner._shm.buf)
        entered, resume = threading.Event(), threading.Event()
        failures: list[Exception] = []
        original = owner._condition

        def finish() -> None:
            try:
                owner.shutdown_arena()
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=finish)

        @contextmanager
        def delayed(**kwargs: Any):
            if threading.current_thread() is worker:
                entered.set()
                self.assertTrue(resume.wait(10))
            with original(**kwargs):
                yield

        try:
            with patch.object(owner, "_condition", delayed):
                worker.start()
                self.assertTrue(entered.wait(10))
                owner.close()
                resume.set()
                worker.join(10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertGreater(export.nbytes, 0)
            with self.assertRaises(BufferError):
                owner.unlink()
        finally:
            resume.set()
            worker.join(10)
            export.release()
            owner.close()
            owner.unlink()

    def test_reclaim_failure_is_observable_and_wakes_waiting_producer(self) -> None:
        """Native reclaim failure is reported and wakes the blocked producer safely."""
        pool = _arena("pool", timeout=10)
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"retained")
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        reader.end_unit(span)
        waiting = threading.Event()
        observed: list[BaseException] = []
        original_wait = pool._wait_space

        def wait(deadline: float, control_deadline: float) -> bool:
            waiting.set()
            return original_wait(deadline, control_deadline)

        def produce() -> None:
            try:
                writer.begin_unit()
            except Exception as error:
                observed.append(error)

        failure = RuntimeError("reclaim publication failed")
        original = type(pool)._reclaimed
        failed = property(
            original.fget, lambda self, value: (_ for _ in ()).throw(failure)
        )
        worker = threading.Thread(target=produce)
        try:
            with patch.object(pool, "_wait_space", wait):
                worker.start()
                self.assertTrue(waiting.wait(10))
                with self.assertLogs(_pool._LG.name, level="ERROR"):
                    with patch.object(type(pool), "_reclaimed", failed):
                        view.release()
                worker.join(2)
                self.assertFalse(worker.is_alive())
            self.assertEqual(observed, [failure])
            self.assertEqual(pool._reclaimed, 0)
            self.assertIn(0, reader._released)
        finally:
            pool._space_signal.release()
            if worker.ident is not None:
                worker.join(10)
            view.release()
            pool._retirement_error = None
            reader.unit_released(0)
            pool.close()
            pool.unlink()
