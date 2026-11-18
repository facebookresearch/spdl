# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Caller-owned allocation tickets over ordinary pickle and public process regions."""

from __future__ import annotations

import copy
import ctypes
import gc
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import pickle
import queue
import sys
import sysconfig
import threading
import time
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
from multiprocessing import shared_memory
from pathlib import Path
from types import CodeType
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import spdl.pipeline as pipeline_module
import torch
from spdl.pipeline import (
    config,
    PipelineBuilder,
    PipelineFailure,
    SharedMemorySegmentPool,
    StatsQueue,
    TaskStatsHook,
)
from spdl.pipeline._arena import _direct, _pool as pool_module
from spdl.pipeline._arena._direct import _ArenaAllocationTicket, ArenaReservation
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

from ._defaults import _DefaultsTestCase


def _cpu_defaults() -> None:
    config.set_default_build_callback(None)
    config.set_default_background_tasks([])
    config.set_default_hook_class(TaskStatsHook)
    config.set_default_queue_class(StatsQueue)
    torch.set_num_threads(1)


def _foreign_producer(pool: SharedMemorySegmentPool, result: Any) -> None:
    _cpu_defaults()
    try:
        with pool.reserve():
            result.put("unexpected reservation")
    except RuntimeError as error:
        result.put(str(error))


def _queue_producer(pool: SharedMemorySegmentPool, result: Any, go: Any) -> None:
    _cpu_defaults()
    for sequence in range(3):
        with pool.reserve() as unit:
            data = unit.allocate_bytes(16 * torch.int64.itemsize)
            out = torch.frombuffer(data.obj, dtype=torch.int64)
            out.fill_(sequence + 3)
            ticket = unit.seal()
            metadata = {"sequence": sequence, "cuda": torch.cuda.is_initialized()}
        del data, out, unit
        result.put((ticket, metadata))
        if not go.wait(30):
            raise TimeoutError("caller did not release output")
        go.clear()


def _remove_os_names(names: list[str]) -> None:
    """Test-only cleanup of names a deliberately unrecoverable pool retains."""
    for name in names:
        try:
            segment = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            continue
        try:
            segment.unlink()
        finally:
            segment.close()


def _die_holding_condition(pool: SharedMemorySegmentPool, ready: Any) -> None:
    pool._space_cv.acquire()
    ready.set()
    os._exit(17)


def _fork_try(pool: SharedMemorySegmentPool, result: Any) -> None:
    try:
        with pool.reserve():
            result.put("unexpected")
    except RuntimeError as error:
        result.put(str(error))


def _fork_resources(unit: Any, context: Any, result: Any) -> None:
    errors = []
    for action in (
        lambda: unit.allocate_bytes(1),
        unit.__enter__,
        lambda: unit.__exit__(None, None, None),
        unit.seal,
        context.__enter__,
        lambda: context.__exit__(None, None, None),
    ):
        try:
            action()
        except RuntimeError as error:
            errors.append(str(error))
    unit._finalizer()
    result.put(errors)


_OPERATION: Any = None


def _initialize(
    pools: tuple[SharedMemorySegmentPool, ...],
    counter: Any,
    receipts: Any,
    fail: bool = False,
) -> None:
    global _OPERATION
    _cpu_defaults()
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    pool = pools[len(pools) - 1 - index]
    receipts.put((os.getpid(), pool.arena_id, torch.cuda.is_initialized()))

    def collate(sequence: int) -> tuple[_ArenaAllocationTicket, dict[str, Any]]:
        with pool.reserve() as unit:
            data = unit.allocate_bytes(2 * 16 * torch.int64.itemsize)
            out = torch.frombuffer(data.obj, dtype=torch.int64).reshape(2, 16)
            if fail:
                raise ValueError("deliberate direct stage failure")
            parts = [
                torch.full((16,), sequence + i, dtype=torch.int64) for i in range(2)
            ]
            torch.stack(parts, out=out)
            return unit.seal(), {
                "sequence": sequence,
                "shape": (2, 16),
                "dtype": torch.int64,
                "cuda": torch.cuda.is_initialized(),
            }

    _OPERATION = collate


def _collate(sequence: int) -> tuple[_ArenaAllocationTicket, dict[str, Any]]:
    return _OPERATION(sequence)


class DirectPoolTest(_DefaultsTestCase):
    def setUp(self) -> None:
        super().setUp()
        _cpu_defaults()
        self.pool = self._pool()

    def _pool(self, count: int = 1, timeout: float = 0) -> SharedMemorySegmentPool:
        pool = SharedMemorySegmentPool(65536, count, acquire_timeout=timeout)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        return pool

    def _ticket(
        self, pool: SharedMemorySegmentPool | None = None
    ) -> _ArenaAllocationTicket:
        pool = self.pool if pool is None else pool
        with pool.reserve() as unit:
            view = unit.allocate_bytes(32)
            view[:] = b"a" * 32
            ticket = unit.seal()
        view.release()
        return ticket

    def test_public_reservation_and_opaque_ticket(self) -> None:
        """Pool acquisition exposes the reservation while its ticket stays opaque."""
        with self.pool.reserve() as unit:
            self.assertIsInstance(unit, pipeline_module.ArenaReservation)
            view = unit.allocate_bytes(32)
            view[:] = b"a" * 32
            ticket = unit.seal()
        view.release()
        self.assertIs(pipeline_module.ArenaReservation, ArenaReservation)
        self.assertIn("ArenaReservation", pipeline_module.__all__)
        self.assertEqual(ticket.arena_id, self.pool.arena_id)
        self.assertFalse(hasattr(ticket, "__dict__"))
        self.assertFalse(hasattr(ticket, "_owner"))
        self.assertFalse(hasattr(pipeline_module, "ArenaAllocationTicket"))
        self.assertFalse(hasattr(pipeline_module, "_ArenaAllocationTicket"))
        with self.assertRaises(FrozenInstanceError):
            cast(Any, ticket).extent = 0
        with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
            cast(Any, ticket).arena_id = "different"
        self.pool.discard(ticket)

    def test_pickle_all_protocols_are_observational(self) -> None:
        """Repeated standard pickle, deepcopy and reduction never change ownership."""
        ticket = self._ticket()
        before = self.pool._prepare_direct().counts()
        copies = [copy.deepcopy(ticket)]
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            packet = pickle.dumps(ticket, protocol=protocol)
            self.assertLessEqual(len(packet), 65536)
            copies.append(pickle.loads(packet))
        for _ in range(3):
            ticket.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
        self.assertEqual(self.pool._prepare_direct().counts(), before)
        self.pool.discard(copies[0])
        for duplicate in copies[1:] + [ticket]:
            with self.assertRaises(RuntimeError):
                self.pool.discard(duplicate)
            with self.assertRaises(RuntimeError):
                self.pool.recover(duplicate)

    def test_dropped_ticket_does_not_release(self) -> None:
        """Losing a sealed packet retains capacity until explicit discard or cleanup."""
        ticket = self._ticket()
        saved = pickle.dumps(ticket)
        del ticket
        gc.collect()
        with self.assertRaises(BufferError):
            self.pool.reserve()
        self.pool.discard(pickle.loads(saved))
        with self.pool.reserve() as unit:
            self.assertIsInstance(unit, ArenaReservation)

    def test_context_holds_empty_extent_until_exit(self) -> None:
        """A caller context retains the logical claim even without native bytes."""
        with self.pool.reserve() as unit:
            ticket = unit.seal()
        context = self.pool.recover(ticket)
        with self.assertRaises(RuntimeError):
            self.pool.discard(ticket)
        with self.assertRaises(BufferError):
            self.pool.reserve()
        with context as batch:
            self.assertIsInstance(batch, memoryview)
            self.assertEqual(batch.nbytes, 0)
            with self.assertRaises(BufferError):
                self.pool.reserve()
        with self.assertRaises(RuntimeError):
            context.__enter__()
        with self.pool.reserve() as unit:
            self.assertGreater(unit._unit, ticket.unit)

    def test_context_exception_releases_claim_but_preserves_alias(self) -> None:
        """Caller exceptions end the claim while escaped native storage still gates reuse."""
        ticket = self._ticket()
        with self.assertRaisesRegex(ValueError, "caller failed"):
            with self.pool.recover(ticket) as batch:
                alias = memoryview(batch)
                raise ValueError("caller failed")
        del batch
        with self.assertRaises(BufferError):
            self.pool.reserve()
        self.assertEqual(bytes(alias), b"a" * 32)
        alias.release()
        with self.pool.reserve() as unit:
            self.assertEqual(unit._unit, 1)

    def test_two_sided_torch_storage_holds(self) -> None:
        """Producer and consumer storage aliases independently prevent FIFO reuse."""
        with self.pool.reserve() as unit:
            data = unit.allocate_bytes(16 * torch.int64.itemsize)
            out = torch.frombuffer(data.obj, dtype=torch.int64)
            out.fill_(9)
            producer = out.detach().untyped_storage()
            ticket = unit.seal()
        data.release()
        with self.pool.recover(ticket) as batch:
            consumer = torch.frombuffer(batch.obj, dtype=torch.int64).detach()
        batch.release()
        del data, batch, out, unit
        with self.assertRaises(BufferError):
            self.pool.reserve()
        del producer
        with self.assertRaises(BufferError):
            self.pool.reserve()
        self.assertTrue(torch.equal(consumer, torch.full((16,), 9, dtype=torch.int64)))
        del consumer
        with self.pool.reserve() as unit:
            self.assertEqual(unit._unit, 1)

    def test_abort_preserves_numpy_alias(self) -> None:
        """An unsealed exception cannot recycle escaped producer array storage."""
        with self.assertRaisesRegex(ValueError, "abort"):
            with self.pool.reserve() as unit:
                data = unit.allocate_bytes(16 * np.dtype(np.int64).itemsize)
                out = np.frombuffer(data, dtype=np.int64)
                out.fill(13)
                raise ValueError("abort")
        data.release()
        del data
        with self.assertRaises(BufferError):
            self.pool.reserve()
        self.assertTrue(np.array_equal(out, np.full(16, 13, dtype=np.int64)))
        del out, unit
        with self.pool.reserve() as unit:
            self.assertEqual(unit._unit, 1)

    def test_discard_does_not_release_producer_storage(self) -> None:
        """Unread discard releases exactly one transfer claim, not live producer storage."""
        with self.pool.reserve() as unit:
            out = unit.allocate_bytes(16)
            out[:] = b"x" * 16
            ticket = unit.seal()
        self.pool.discard(ticket)
        with self.assertRaises(RuntimeError):
            self.pool.discard(ticket)
        with self.assertRaises(BufferError):
            self.pool.reserve()
        self.assertEqual(bytes(out), b"x" * 16)
        out.release()
        with self.pool.reserve() as unit:
            self.assertEqual(unit._unit, 1)

    def test_wrong_arena_does_not_claim_or_switch_pool(self) -> None:
        """Routing to the wrong pool fails before touching its mode or a live claim."""
        other = self._pool()
        ticket = self._ticket()
        with self.assertRaisesRegex(RuntimeError, "different pool"):
            other.recover(ticket)
        self.assertEqual(other._mode, 0)
        with self.pool.recover(ticket) as batch:
            self.assertEqual(bytes(batch), b"a" * 32)

    def test_invalid_identity_extent_hash_and_generation(self) -> None:
        """Tampered byte descriptors cannot consume the genuine sealed allocation."""
        ticket = self._ticket()
        variants = [
            replace(ticket, _arena_id="foreign"),
            replace(ticket, unit=-1),
            replace(ticket, unit=True),
            replace(ticket, unit=2**64),
            replace(ticket, unit=ticket.unit + 1),
            replace(ticket, producer_pid=os.getpid() + 1),
            replace(ticket, producer_pid=0),
            replace(ticket, producer_pid=True),
            replace(ticket, extent=-1),
            replace(ticket, extent=True),
            replace(ticket, extent=65537),
            replace(ticket, extent=33),
            replace(ticket, extent=0),
        ]
        for invalid in variants:
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((RuntimeError, ValueError)),
            ):
                self.pool.recover(invalid)
        with self.pool.recover(ticket) as batch:
            self.assertEqual(bytes(batch), b"a" * 32)
        del batch
        with self.pool.reserve() as unit:
            new = unit.seal()
        with self.assertRaises(RuntimeError):
            self.pool.recover(ticket)
        self.pool.discard(new)

    def test_raced_recover_discard_has_single_winner(self) -> None:
        """Duplicate recover and discard callers compete for one shared claim."""
        for _ in range(10):
            with self.pool.reserve() as unit:
                ticket = unit.seal()
            start = threading.Barrier(3)

            def claim(
                recover: bool,
                ticket: _ArenaAllocationTicket = ticket,
                start: threading.Barrier = start,
            ) -> bool:
                start.wait(10)
                try:
                    if recover:
                        with self.pool.recover(copy.deepcopy(ticket)) as value:
                            self.assertEqual(value.nbytes, 0)
                    else:
                        self.pool.discard(pickle.loads(pickle.dumps(ticket)))
                    return True
                except RuntimeError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(claim, True)
                second = executor.submit(claim, False)
                start.wait(10)
                self.assertEqual(sum([first.result(10), second.result(10)]), 1)
        self.assertEqual(self.pool._published, 10)
        self.assertEqual(self.pool._reclaimed, 10)

    def test_post_claim_exception_has_preexisting_cleanup_guard(self) -> None:
        """Failure immediately after the shared claim store retires only its own guard."""
        ticket = self._ticket()
        owner = self.pool._prepare_direct()
        original = owner._claim

        def fail(descriptor: Any, claim: bytes) -> None:
            original(descriptor, claim)
            raise MemoryError("after claim store")

        with patch.object(owner, "_claim", fail):
            with self.assertRaisesRegex(MemoryError, "after claim store"):
                self.pool.recover(ticket)
        self.assertEqual(self.pool._reclaimed, 1)
        with self.pool.reserve() as unit:
            self.assertEqual(unit._unit, 1)

    def test_seal_header_failure_before_latch_rolls_back_publication(self) -> None:
        """A failed pre-latch counter write aborts without inventing a publication."""
        original = _direct._HEADER
        failed = []

        class HeaderOnce:
            size = original.size

            def unpack_from(self, *args: Any) -> Any:
                return original.unpack_from(*args)

            def pack_into(self, *args: Any) -> None:
                original.pack_into(*args)
                if args[4] == 1 and not failed:
                    failed.append(True)
                    raise MemoryError("before seal latch")

        with patch.object(_direct, "_HEADER", HeaderOnce()):
            with self.assertRaises(MemoryError):
                with self.pool.reserve() as unit:
                    unit.seal()
        self.assertEqual(self.pool._published, 0)
        self.assertEqual(self.pool._reclaimed, 0)
        with self.pool.reserve() as next_unit:
            self.assertEqual(next_unit._unit, 1)

    def test_seal_slot_failure_after_latch_retains_exact_transfer(self) -> None:
        """Failure after the slot latch preserves one publication until explicit discard."""
        owner = self.pool._prepare_direct()
        original_slot = _direct._SLOT
        original_seal = owner.seal
        saved = []
        failed = []

        class SlotOnce:
            size = original_slot.size

            def unpack_from(self, *args: Any) -> Any:
                return original_slot.unpack_from(*args)

            def pack_into(self, *args: Any) -> None:
                original_slot.pack_into(*args)
                if args[5] == _direct._PUBLISHED and not failed:
                    failed.append(True)
                    raise MemoryError("after seal slot latch")

        def save(ticket: _ArenaAllocationTicket) -> None:
            saved.append(ticket)
            original_seal(ticket)

        with (
            patch.object(_direct, "_SLOT", SlotOnce()),
            patch.object(owner, "seal", save),
        ):
            with self.assertRaises(MemoryError):
                with self.pool.reserve() as unit:
                    unit.seal()
        self.assertEqual(self.pool._published, 1)
        self.assertEqual(self.pool._reclaimed, 0)
        with self.assertRaises(BufferError):
            self.pool.reserve()
        self.pool.discard(saved[0])
        self.assertEqual(self.pool._reclaimed, 1)

    def test_post_seal_exception_never_aborts_transfer(self) -> None:
        """Failure after the seal latch loses a ticket without inferring nondelivery."""
        saved: list[_ArenaAllocationTicket] = []
        owner = self.pool._prepare_direct()
        original = owner.seal

        def fail(ticket: _ArenaAllocationTicket) -> None:
            saved.append(ticket)
            original(ticket)
            raise MemoryError("after seal latch")

        with patch.object(owner, "seal", fail):
            with self.assertRaises(MemoryError):
                with self.pool.reserve() as unit:
                    unit.seal()
        del unit
        gc.collect()
        with self.assertRaises(BufferError):
            self.pool.reserve()
        self.pool.discard(saved[0])
        self.assertEqual(self.pool._reclaimed, 1)

    def test_reservation_constructor_failure_aborts(self) -> None:
        """A failed reservation lifetime guard returns the tentative slot safely."""
        with patch.object(
            _direct.weakref, "finalize", side_effect=MemoryError("guard failed")
        ):
            with self.assertRaises(MemoryError):
                self.pool.reserve()
        with self.pool.reserve() as unit:
            self.assertEqual(unit._unit, 1)

    def _failed_native_handoff(self, consumer: bool) -> None:
        owner = self.pool._prepare_direct()
        unit = self.pool.reserve()
        first = unit.allocate_bytes(32)
        first[:] = b"a" * 32
        ticket = unit.seal() if consumer else None
        root = MemoryError("after native handoff")
        captured: list[Any] = []
        self.addCleanup(setattr, root, "__traceback__", None)
        self.addCleanup(captured.clear)

        def fail_cast(type_: Any, value: Any) -> Any:
            captured.append(value[1:])
            raise root

        with patch.object(_direct, "cast", fail_cast):
            try:
                if consumer:
                    assert ticket is not None
                    self.pool.recover(ticket)
                else:
                    owner.export(unit._unit, 0, 32, 0)
            except MemoryError as error:
                self.assertIs(error, root)
            else:
                self.fail("native return did not fail")
        if not consumer:
            unit.__exit__(None, None, None)
        first.release()
        del first, unit
        gc.collect()
        self.assertEqual(bytes(captured[0]), b"a" * 31)
        with self.assertRaises(BufferError):
            self.pool.reserve()
        root.__traceback__ = None
        gc.collect()
        with self.assertRaises(BufferError):
            self.pool.reserve()
        captured.clear()
        gc.collect()
        with self.pool.reserve() as unit:
            self.assertEqual(unit._unit, 1)

    def test_producer_failed_handoff_retains_traceback_and_alias(self) -> None:
        """Aborted producer storage survives failed native handoff and escaped aliases."""
        self._failed_native_handoff(False)

    def test_consumer_failed_handoff_retains_traceback_and_alias(self) -> None:
        """Failed recovery ends its claim without retiring adopted native storage."""
        self._failed_native_handoff(True)

    def test_live_views_survive_close_unlink(self) -> None:
        """Deferred close preserves live storage until an explicit quiescent retry."""
        with self.pool.reserve() as unit:
            data = unit.allocate_bytes(16 * np.dtype(np.int64).itemsize)
            producer = np.frombuffer(data, dtype=np.int64)
            producer.fill(29)
            ticket = unit.seal()
        data.release()
        context = self.pool.recover(ticket)
        with context as batch:
            consumer = np.frombuffer(batch, dtype=np.int64)
            batch.release()
            self.pool.close()
            self.assertFalse(self.pool._closed)
            self.assertIsNotNone(self.pool._ctrl_buf)
            self.assertTrue(np.array_equal(producer, np.full(16, 29, dtype=np.int64)))
            self.assertTrue(np.array_equal(consumer, np.full(16, 29, dtype=np.int64)))
        self.pool.close()
        self.assertFalse(self.pool._closed)
        self.assertTrue(np.array_equal(consumer, np.full(16, 29, dtype=np.int64)))
        del data, batch, consumer, producer, context, unit
        gc.collect()
        self.pool.close()
        self.assertTrue(self.pool._closed)
        self.assertIsNone(self.pool._ctrl_buf)
        self.assertIsNone(self.pool._segs)
        self.pool.unlink()

    def test_completion_order_does_not_break_fifo(self) -> None:
        """Out-of-order seals and discards preserve the oldest live slot watermark."""
        pool = self._pool(2)
        first, second = pool.reserve(), pool.reserve()
        b = second.seal()
        a = first.seal()
        pool.discard(b)
        with self.assertRaises(BufferError):
            pool.reserve()
        pool.discard(a)
        self.assertEqual(pool._reclaimed, 2)
        with pool.reserve() as unit:
            self.assertEqual(unit._unit, 2)

    def test_caller_replacement_storage_does_not_change_sealed_bytes(self) -> None:
        """Caller tensor rebinding cannot change the arena's sealed byte extent."""
        with self.pool.reserve() as unit:
            data = unit.allocate_bytes(16 * torch.int64.itemsize)
            out = torch.frombuffer(data.obj, dtype=torch.int64)
            out.fill_(17)
            out.set_(torch.full((16,), 23, dtype=torch.int64).untyped_storage())
            ticket = unit.seal()
        with self.pool.recover(ticket) as batch:
            actual = torch.frombuffer(batch.obj, dtype=torch.int64)
            self.assertEqual(actual.tolist(), [17] * 16)
            self.assertEqual(out.tolist(), [23] * 16)

    def test_byte_extent_and_alignment_limits(self) -> None:
        """Positive byte sizes and aligned allocations must fit the reserved extent."""
        with self.pool.reserve(max_bytes=64) as unit:
            for size in (0, -1, True, 1.5):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    unit.allocate_bytes(cast(Any, size))
            for alignment in (0, -1, 3, True, 1.5):
                with self.subTest(alignment=alignment), self.assertRaises(ValueError):
                    unit.allocate_bytes(1, alignment=cast(Any, alignment))
            with self.assertRaises(BufferError):
                unit.allocate_bytes(65)
            out = unit.allocate_bytes(32)
            out[:] = bytes(range(32))
            tail = unit.allocate_bytes(32, alignment=32)
            tail[:] = bytes(range(32, 64))
            ticket = unit.seal()
            with self.assertRaises(RuntimeError):
                unit.allocate_bytes(1)
            with self.assertRaises(RuntimeError):
                unit.seal()
        with self.pool.recover(ticket) as batch:
            self.assertEqual(batch.nbytes, 64)
            self.assertEqual(bytes(batch), bytes(range(64)))

    def _misaligned(self) -> int:
        with self.pool.reserve() as probe:
            view = probe.allocate_bytes(1)
            base = ctypes.addressof(ctypes.c_char.from_buffer(view))
            view.release()
        return (base & -base) << 1

    def test_rejected_alignment_releases_export_while_error_is_held(self) -> None:
        """A rejected allocation drops its export even while its traceback is kept."""
        alignment = self._misaligned()
        owner = self.pool._prepare_direct()
        with self.pool.reserve(max_bytes=64) as unit:
            before = owner._record(unit._unit)
            # assertRaises clears traceback frames, which would hide a retained view.
            try:
                unit.allocate_bytes(1, alignment=alignment)
            except ValueError as error:
                rejected = error
            else:
                self.fail("misaligned allocation was accepted")
            self.assertIsNotNone(rejected.__traceback__)
            self.assertEqual(owner._record(unit._unit), before)
            self.assertEqual(sum(self.pool._mapping_exports.values()), 0)
            out = unit.allocate_bytes(64)
            out[:] = bytes(range(64))
            alias = out[32:]
            ticket = unit.seal()
        out.release()
        self.assertEqual(bytes(alias), bytes(range(32, 64)))
        alias.release()
        with self.pool.recover(ticket) as batch:
            self.assertEqual(bytes(batch), bytes(range(64)))
        batch.release()
        self.assertEqual(sum(self.pool._mapping_exports.values()), 0)
        self.assertIsInstance(rejected, ValueError)

    def test_rejected_alignment_keeps_hold_of_retained_alias(self) -> None:
        """An alias kept by a failed address check still owns the rejected hold."""
        alignment = self._misaligned()
        owner = self.pool._prepare_direct()
        retained: list[object] = []
        address = ctypes.addressof

        def retain(obj: Any) -> int:
            retained.append(obj)
            return address(obj)

        with self.pool.reserve(max_bytes=64) as unit:
            before = owner._record(unit._unit)
            with (
                patch.object(_direct.ctypes, "addressof", retain),
                self.assertRaises(ValueError),
            ):
                unit.allocate_bytes(1, alignment=alignment)
            self.assertEqual(owner._record(unit._unit)[1], before[1] + 1)
            retained.clear()
            self.assertEqual(owner._record(unit._unit), before)
            self.assertEqual(sum(self.pool._mapping_exports.values()), 0)

    def test_multiple_dtypes_alignment_and_caller_repeated_leaves(self) -> None:
        """Caller-computed mixed layouts preserve full values and repeated-leaf identity."""
        dtypes = (torch.float16, torch.float64, torch.uint8)
        offsets = (0, 64, 256)
        array_offset, extent = 320, 344
        with self.pool.reserve() as unit:
            data = unit.allocate_bytes(extent)
            tensors = [
                torch.frombuffer(data.obj, dtype=dtype, count=17, offset=offset)
                for dtype, offset in zip(dtypes, offsets, strict=True)
            ]
            array = np.frombuffer(data, dtype=np.float64, count=3, offset=array_offset)
            array.fill(3.25)
            for index, tensor in enumerate(tensors):
                tensor.fill_(index + 2)
                self.assertEqual(tensor.data_ptr() % 64, 0)
            envelope = unit.seal(), (dtypes, offsets, array_offset)
        ticket, metadata = pickle.loads(pickle.dumps(envelope))
        with self.pool.recover(ticket) as batch:
            recovered = [
                torch.frombuffer(batch.obj, dtype=dtype, count=17, offset=offset)
                for dtype, offset in zip(metadata[0], metadata[1], strict=True)
            ]
            repeated = [recovered, recovered[0]]
            self.assertIs(repeated[0][0], repeated[1])
            for expected, actual in zip(tensors, recovered, strict=True):
                self.assertTrue(torch.equal(expected, actual))
            actual_array = np.frombuffer(
                batch, dtype=np.float64, count=3, offset=metadata[2]
            )
            self.assertTrue(np.array_equal(actual_array, array))

    def test_caller_numpy_dtype_metadata_is_not_interpreted_by_arena(self) -> None:
        """Caller-owned structured and custom dtype metadata travel outside tickets."""
        dtypes = (
            np.dtype([("x", "i4")]),
            np.dtype((np.int32, (2,))),
            np.dtype("U4"),
            cast(Any, np.dtype)("f8", metadata={"x": 1}),
        )
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                with self.pool.reserve() as unit:
                    data = unit.allocate_bytes(2 * dtype.itemsize)
                    data[:] = bytes(data.nbytes)
                    expected = np.frombuffer(data, dtype=dtype)
                    ticket, metadata = pickle.loads(pickle.dumps((unit.seal(), dtype)))
                with self.pool.recover(ticket) as batch:
                    actual = np.frombuffer(batch, dtype=metadata)
                    self.assertTrue(np.array_equal(expected, actual))
                    self.assertEqual(actual.dtype.metadata, expected.dtype.metadata)
                del data, batch, expected, actual, unit

    def test_ordinary_and_direct_modes_are_exclusive(self) -> None:
        """Direct control preparation does not disable an unused ordinary pool."""
        ordinary = self._pool()
        writer, reader = ordinary.open_writer(), ordinary.open_reader()
        writer.begin_unit()
        offset, nbytes = writer.write_binary(b"normal")
        span = writer.commit_unit()
        view = reader.read_binary(offset, nbytes)
        reader.end_unit(span, [])
        self.assertEqual(bytes(view), b"normal")
        view.release()
        with self.assertRaises(RuntimeError):
            ordinary.reserve()
        ticket = self._ticket()
        with self.assertRaises(RuntimeError):
            self.pool.open_writer()
        with self.assertRaises(RuntimeError):
            self.pool.open_reader()
        self.pool.discard(ticket)

    def test_pool_handles_not_item_queue_packets(self) -> None:
        """Pool synchronization handles are spawn arguments, never ordinary pickle payloads."""
        with self.assertRaises(RuntimeError):
            pickle.dumps(self.pool)
        ticket = self._ticket()
        self.assertIsInstance(
            pickle.loads(pickle.dumps(ticket)), _ArenaAllocationTicket
        )
        self.pool.discard(ticket)

    def test_queue_full_retry_reuses_one_metadata_envelope(self) -> None:
        """Ordinary Queue Full retries and feeder pickle never create another ownership hold."""
        ctx = mp.get_context("spawn")
        raw = ctx.Queue(1)
        self.addCleanup(raw.join_thread)
        self.addCleanup(raw.close)
        ticket = self._ticket()
        raw.put_nowait(None)
        for _ in range(3):
            with self.assertRaises(queue.Full):
                raw.put_nowait(ticket)
        self.assertIsNone(raw.get(timeout=30))
        raw.put_nowait(ticket)
        restored = raw.get(timeout=30)
        self.assertEqual(self.pool._published, 1)
        self.pool.discard(restored)
        with self.assertRaises(RuntimeError):
            self.pool.recover(ticket)

    def test_feeder_late_failure_does_not_retire_sealed_ticket(self) -> None:
        """A real feeder reduction failure leaves the ticket occupied until explicit discard."""
        ctx = mp.get_context("spawn")
        raw = ctx.Queue(1)
        self.addCleanup(raw.join_thread)
        self.addCleanup(raw.close)
        ticket = self._ticket()
        failed = threading.Event()
        errors: list[BaseException] = []

        def observe(error: BaseException, obj: object) -> None:
            errors.append(error)
            failed.set()

        cast(Any, raw)._on_queue_feeder_error = observe
        raw.put((ticket, lambda: None))
        self.assertTrue(failed.wait(10))
        self.assertEqual(len(errors), 1)
        with self.assertRaises(BufferError):
            self.pool.reserve()
        raw.put(ticket)
        self.pool.discard(raw.get(timeout=30))
        self.assertEqual(self.pool._reclaimed, 1)

    def test_actual_spawn_control_exists_before_serialization(self) -> None:
        """A never-reserving parent recovers spawn output and unblocks shared reuse."""
        ctx = mp.get_context("spawn")
        raw, go = ctx.Queue(), ctx.Event()
        process = ctx.Process(target=_queue_producer, args=(self.pool, raw, go))
        process.start()
        try:
            for sequence in range(3):
                ticket, metadata = raw.get(timeout=30)
                self.assertEqual(ticket.arena_id, self.pool.arena_id)
                self.assertEqual(ticket.producer_pid, process.pid)
                with self.pool.recover(ticket) as data:
                    batch = torch.frombuffer(data.obj, dtype=torch.int64)
                    self.assertFalse(metadata["cuda"])
                    self.assertEqual(metadata["sequence"], sequence)
                    self.assertTrue(
                        torch.equal(
                            batch,
                            torch.full((16,), sequence + 3, dtype=torch.int64),
                        )
                    )
                del data, batch
                go.set()
            process.join(30)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join(10)
            raw.close()
            raw.join_thread()
        self.assertEqual(self.pool._reclaimed, 3)

    def test_second_producer_pid_rejected(self) -> None:
        """A foreign process fails producer ownership before waiting for a full pool."""
        with self.pool.reserve() as unit:
            view = unit.allocate_bytes(16)
            ctx = mp.get_context("spawn")
            result = ctx.Queue()
            process = ctx.Process(target=_foreign_producer, args=(self.pool, result))
            process.start()
            try:
                self.assertIn("another process", result.get(timeout=30))
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            finally:
                if process.is_alive():
                    process.kill()
                    process.join(10)
                result.close()
                result.join_thread()
            view.release()

    @unittest.skipUnless(
        "fork" in mp.get_all_start_methods(), "requires the fork start method"
    )
    def test_fork_handle_inheritance_fails_safe(self) -> None:
        """Fork inheritance cannot impersonate a spawn-attached pool owner."""
        ctx = mp.get_context("fork")
        result = ctx.Queue()
        process = ctx.Process(target=_fork_try, args=(self.pool, result))
        process.start()
        try:
            self.assertIn("fork", result.get(timeout=10))
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join(10)
            result.close()
            result.join_thread()

    def test_close_wakes_capacity_waiter_without_deadlock(self) -> None:
        """Condition-first close wakes a blocked reserve without local-lock inversion."""
        pool = self._pool(timeout=10)
        with pool.reserve() as unit:
            held = unit.allocate_bytes(16)
        entered = threading.Event()
        original = pool._wait_space

        def wait(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            return original(*args, **kwargs)

        with (
            patch.object(pool, "_wait_space", wait),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            pending = executor.submit(pool.reserve)
            self.assertTrue(entered.wait(10))
            pool.close()
            with self.assertRaises(BufferError):
                pending.result(10)
        self.assertEqual(bytes(held), bytes(16))
        held.release()
        pool.close()
        self.assertTrue(pool._closed)
        self.assertIsNone(pool._ctrl_buf)

    def test_ordinary_writer_close_wakes_same_pid_waiter(self) -> None:
        """Creator close wakes an ordinary same-PID capacity waiter with BufferError."""
        pool = self._pool(timeout=10)
        writer = pool.open_writer()
        writer.begin_unit()
        writer.write_binary(b"a" * 32)
        writer.commit_unit()
        entered = threading.Event()
        original = pool._wait_space

        def wait(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            return original(*args, **kwargs)

        with (
            patch.object(pool, "_wait_space", wait),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            pending = executor.submit(writer.begin_unit)
            self.assertTrue(entered.wait(timeout=pool._acquire_timeout))
            self.assertFalse(pending.done())
            pool.close()
            with self.assertRaises(BufferError):
                pending.result(timeout=pool._acquire_timeout)
        self.assertTrue(pool._closed)
        self.assertIsNone(pool._ctrl_buf)

    def test_stranded_condition_close_is_bounded_and_retains_mapping(self) -> None:
        """Confirmed producer death while holding a nonrobust mutex retires conservatively."""
        ctx = mp.get_context("spawn")
        pool = SharedMemorySegmentPool(65536, 1, acquire_timeout=0)
        with pool.reserve() as unit:
            alias = unit.allocate_bytes(16)
            alias[:] = b"z" * 16
        ready = ctx.Event()
        process = ctx.Process(target=_die_holding_condition, args=(pool, ready))
        process.start()
        self.assertTrue(ready.wait(30))
        process.join(30)
        self.assertEqual(process.exitcode, 17)
        start = time.monotonic()
        with patch.object(pool_module, "_CONTROL_LOCK_TIMEOUT", 0.05):
            with self.assertRaisesRegex(RuntimeError, "retired and mappings retained"):
                pool.close()
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(bytes(alias), b"z" * 16)
        self.assertIn(pool, pool_module._REGISTRATION_QUARANTINE)
        names = [pool.arena_id, *pool._seg_names]
        self.addCleanup(_remove_os_names, names)
        start = time.monotonic()
        alias.release()
        self.assertLess(time.monotonic() - start, 2)
        # The stranded control lock blocks physical retirement, so names stay.
        with self.assertRaises(BufferError):
            pool.unlink()
        self.assertIn(pool, pool_module._REGISTRATION_QUARANTINE)
        with self.assertRaisesRegex(RuntimeError, "retired"):
            pool.reserve()

    def _failing_drain(self, failures: set[str]) -> tuple[Any, list[str], ExitStack]:
        """Queue two aborts and one export; fail the named retirements."""
        pool = SharedMemorySegmentPool(65536, 2, acquire_timeout=0)
        self.addCleanup(pool.close)
        with pool.reserve():
            pass
        direct = pool._direct
        assert direct is not None
        calls: list[str] = []

        def abort(unit: int) -> None:
            calls.append(f"abort{unit}")
            if f"abort{unit}" in failures:
                raise RuntimeError(f"deliberate abort{unit} failure")

        def retire(unit: int, side: int, name: str) -> None:
            calls.append(f"export{unit}")
            if f"export{unit}" in failures:
                raise RuntimeError(f"deliberate export{unit} failure")

        direct._deferred_aborts.extend([7, 8])
        direct._deferred_exports.append((9, 0, "export"))
        stack = ExitStack()
        stack.enter_context(patch.object(direct, "_abort_unit", abort))
        stack.enter_context(patch.object(direct, "_retire_export", retire))
        self.addCleanup(stack.close)
        return direct, calls, stack

    def test_failed_deferred_retirement_still_drains_the_rest(self) -> None:
        """The first drain failure is raised after every queued retirement runs."""
        direct, calls, stack = self._failing_drain({"abort8", "export9"})
        with self.assertRaises(RuntimeError) as caught:
            with direct._control():
                pass
        stack.close()
        self.assertEqual(calls, ["abort8", "abort7", "export9"])
        self.assertEqual(str(caught.exception), "deliberate abort8 failure")
        self.assertEqual(
            caught.exception.__notes__,
            [
                "Later deferred arena retirement also failed: "
                "RuntimeError('deliberate export9 failure')"
            ],
        )
        self.assertEqual((direct._deferred_aborts, direct._deferred_exports), ([], []))
        self.assertFalse(direct._draining_exports)
        self.assertEqual(direct._control_depth, 0)

    def test_update_failure_is_not_replaced_by_drain_failure(self) -> None:
        """An update's own error propagates; drain failures are attached as notes."""
        direct, calls, stack = self._failing_drain({"abort8"})
        with self.assertRaisesRegex(ValueError, "deliberate update failure") as caught:
            with direct._control():
                raise ValueError("deliberate update failure")
        stack.close()
        self.assertEqual(calls, ["abort8", "abort7", "export9"])
        self.assertEqual(
            caught.exception.__notes__,
            [
                "Deferred arena retirement also failed: "
                "RuntimeError('deliberate abort8 failure')"
            ],
        )
        self.assertEqual((direct._deferred_aborts, direct._deferred_exports), ([], []))
        self.assertFalse(direct._draining_exports)

    def test_ticket_does_not_retain_owner_after_terminal_cleanup(self) -> None:
        """A metadata packet remains picklable without retaining its closed pool."""
        pool = SharedMemorySegmentPool(65536, 1, acquire_timeout=0)
        ticket = self._ticket(pool)
        packet = pickle.dumps(ticket)
        pool.close()
        pool.unlink()
        reference = weakref.ref(pool)
        del pool
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(pickle.dumps(ticket), packet)

    def test_native_retirement_is_deferred_past_control_snapshot(self) -> None:
        """Reentrant native release cannot resurrect a stale producer or mapping hold."""
        unit = self.pool.reserve()
        owner = self.pool._prepare_direct()
        pending = [unit.allocate_bytes(32)]
        original = owner._record
        released = []

        def record(index: int) -> Any:
            result = original(index)
            if pending:
                pending.pop().release()
                released.append(True)
            return result

        with patch.object(owner, "_record", record):
            fresh = owner.export(unit._unit, 0, 32, 0)
        self.assertEqual(released, [True])
        self.assertEqual(owner._record(unit._unit)[1], 2)
        self.assertEqual(self.pool._mapping_exports[self.pool._seg_names[0]], 1)
        unit.__exit__(None, None, None)
        fresh.release()
        with self.pool.reserve() as next_unit:
            self.assertEqual(next_unit._unit, 1)

    def test_retirement_never_holds_lease_while_waiting_for_condition(self) -> None:
        """Another thread can inspect the lease while its retirement waits on shared control."""
        unit = self.pool.reserve()
        owner = self.pool._prepare_direct()
        entered = threading.Event()
        errors = []
        original = self.pool._space_cv.acquire
        main = threading.current_thread()

        def acquire(*args: Any, **kwargs: Any) -> Any:
            if threading.current_thread() is not main:
                entered.set()
            return original(*args, **kwargs)

        def retire() -> None:
            try:
                unit._finalizer()
            except Exception as error:
                errors.append(error)

        with patch.object(self.pool._space_cv, "acquire", acquire):
            with owner._control():
                worker = threading.Thread(target=retire, daemon=True)
                worker.start()
                self.assertTrue(entered.wait(10))
                available = unit._lease._lock.acquire(timeout=0.5)
                if available:
                    unit._lease._lock.release()
                self.assertTrue(available)
            worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        with self.pool.reserve() as next_unit:
            self.assertEqual(next_unit._unit, 1)

    @unittest.skipUnless(
        "fork" in mp.get_all_start_methods(), "requires the fork start method"
    )
    def test_fork_rejects_before_inherited_local_locks(self) -> None:
        """Forked units and contexts reject PID mismatch before inherited local locks."""
        pool = self._pool(2)
        ctx = mp.get_context("fork")
        result = ctx.Queue()
        with pool.reserve() as unit:
            with pool.reserve() as second:
                ticket = second.seal()
            context = pool.recover(ticket)
            with context:
                with unit._lock, context._lock:
                    process = ctx.Process(
                        target=_fork_resources, args=(unit, context, result)
                    )
                    process.start()
                    try:
                        errors = result.get(timeout=10)
                        self.assertEqual(len(errors), 6)
                        self.assertTrue(all("PID" in error for error in errors))
                        process.join(10)
                        self.assertEqual(process.exitcode, 0)
                    finally:
                        if process.is_alive():
                            process.kill()
                            process.join(10)
                        result.close()
                        result.join_thread()
        with pool.reserve() as next_unit:
            self.assertEqual(next_unit._unit, 2)

    def test_wrong_claim_nonce_and_generation_cannot_finish_context(self) -> None:
        """Only the current generation's exact claim nonce can release its caller hold."""
        with self.pool.reserve() as unit:
            ticket = unit.seal()
        context = self.pool.recover(ticket)
        owner = self.pool._prepare_direct()
        before = owner._record(ticket.unit)
        nonce = context.claim
        wrong = bytes([nonce[0] ^ 255]) + nonce[1:]
        owner.finish(ticket.unit, wrong)
        owner.finish(ticket.unit + 1, nonce)
        self.assertEqual(owner._record(ticket.unit), before)
        with self.assertRaises(BufferError):
            self.pool.reserve()
        with self.assertRaises(TypeError):
            pickle.dumps(context)
        with context as batch:
            self.assertEqual(batch.nbytes, 0)
        owner.finish(ticket.unit, nonce)
        self.assertEqual(self.pool._reclaimed, 1)
        with self.assertRaises(RuntimeError):
            self.pool.discard(ticket)

    def test_optional_io_absent_preserves_empty_extent_operation(self) -> None:
        """Native IO is needed only for actual views, not sealing an empty extent."""
        missing = ModuleNotFoundError(
            "optional helper absent", name="spdl.io.lib._buffer"
        )
        metadata = {"metadata": 7}
        with patch("importlib.import_module", side_effect=missing):
            with self.pool.reserve() as unit:
                ticket = unit.seal()
            ticket, received = pickle.loads(pickle.dumps((ticket, metadata)))
            with self.pool.recover(ticket) as batch:
                self.assertEqual(batch.nbytes, 0)
                self.assertEqual(received, {"metadata": 7})
            with self.assertRaisesRegex(RuntimeError, "install spdl.io"):
                with self.pool.reserve() as unit:
                    unit.allocate_bytes(32)
        self.assertTrue(
            all(value == 0 for value in self.pool._mapping_exports.values())
        )
        with self.pool.reserve() as next_unit:
            self.assertEqual(next_unit._unit, 2)


class DirectRegionTest(_DefaultsTestCase):
    def setUp(self) -> None:
        super().setUp()
        _cpu_defaults()

    def _run(
        self, batches: int, fail: bool = False, continuous: bool = False
    ) -> list[int]:
        ctx = mp.get_context("spawn")
        pools = tuple(
            SharedMemorySegmentPool(65536, 2, acquire_timeout=10) for _ in range(2)
        )
        receipts = ctx.Queue()
        for pool in pools:
            self.addCleanup(pool.unlink)
            self.addCleanup(pool.close)
        spec = ProcessPoolExecutorConfig(
            max_workers=2,
            mp_context="spawn",
            initializer=_initialize,
            initargs=(pools, ctx.Value("i", 0), receipts, fail),
        )
        lookup = {pool.arena_id: pool for pool in pools}

        def recover(envelope: tuple[_ArenaAllocationTicket, dict[str, Any]]) -> int:
            ticket, metadata = envelope
            self.assertIsInstance(ticket, _ArenaAllocationTicket)
            with lookup[ticket.arena_id].recover(ticket) as data:
                batch = torch.frombuffer(data.obj, dtype=metadata["dtype"]).reshape(
                    metadata["shape"]
                )
                expected = torch.stack(
                    [
                        torch.full((16,), metadata["sequence"] + i, dtype=torch.int64)
                        for i in range(2)
                    ]
                )
                self.assertTrue(torch.equal(batch, expected))
                self.assertFalse(metadata["cuda"])
                return metadata["sequence"]

        pipeline = (
            PipelineBuilder()
            .add_source(range(batches), continuous=continuous)
            .to(spec, buffer_size=3)
            .pipe(_collate)
            .to(MAIN_PROCESS, buffer_size=2)
            .pipe(recover)
            .add_sink(2)
            .build(num_threads=3, max_failures=0)
        )
        result: list[int] = []
        completed = False
        try:
            try:
                with pipeline.auto_stop(timeout=30):
                    try:
                        for _ in range(3 if continuous else 1):
                            epoch = list(pipeline.get_iterator(timeout=30))
                            self.assertCountEqual(epoch, range(batches))
                            result.extend(epoch)
                        completed = True
                    finally:
                        for pool in pools:
                            pool.close()
            except PipelineFailure as error:
                if not continuous or not completed:
                    raise
                self.assertTrue(error.exceptions)
                self.assertTrue(all(pool._closing for pool in pools))
                for failure in error.exceptions:
                    root: BaseException = failure
                    while root.__cause__ is not None:
                        root = root.__cause__
                    if isinstance(root, BufferError):
                        self.assertRegex(str(root), "closing|shut down")
                    else:
                        self.assertIs(type(root), RuntimeError)
                        self.assertEqual(str(root), "Arena recovery is cancelled")
            workers = [receipts.get(timeout=10) for _ in range(2)]
            self.assertEqual({row[1] for row in workers}, set(lookup))
            self.assertTrue(all(not row[2] for row in workers))
            self.assertTrue(
                all(
                    not p.is_alive()
                    for p in mp.active_children()
                    if p.pid in {row[0] for row in workers}
                )
            )
            self.assertTrue(
                all(not pool_module._process_exists(row[0]) for row in workers)
            )
            for pool in pools:
                pool.close()
            return result
        finally:
            receipts.close()
            receipts.join_thread()

    def test_public_ordinary_regions_with_caller_main_recovery(self) -> None:
        """Public regions carry opaque tickets with stable reverse-pool routing."""
        self.assertCountEqual(self._run(12), range(12))

    def test_continuous_caller_shutdown_wakes_before_reap(self) -> None:
        """Caller shutdown wakes prefetched reservations before ordinary worker reap."""
        self.assertCountEqual(self._run(12, continuous=True), list(range(12)) * 3)

    def test_empty_public_region_preserves_eof(self) -> None:
        """An empty ordinary region terminates without creating direct allocations."""
        self.assertEqual(self._run(0), [])

    def test_unsealed_stage_failure_is_bounded(self) -> None:
        """Region failure preserves its cause and reaps workers before owner cleanup."""
        with self.assertRaises(PipelineFailure) as raised:
            self._run(4, fail=True)
        self.assertIn(
            "deliberate direct stage failure", str(raised.exception.exceptions)
        )


class _NativeReleaseProbe:
    def __init__(self, pool: SharedMemorySegmentPool) -> None:
        self.pool = pool
        self.owner = pool._prepare_direct()
        self.done = threading.Event()
        self.snapshots: list[tuple[int, ...]] = []
        self._release = self.owner._release_export
        self._patch = patch.object(self.owner, "_release_export", self._observe)

    def __enter__(self) -> _NativeReleaseProbe:
        self._patch.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._patch.stop()

    def _observe(self, unit: int, side: int, name: str) -> None:
        self._release(unit, side, name)
        with self.pool._close_lock:
            native = sum(self.pool._mapping_exports.values())
        counts = self.owner.counts()
        _, producer, consumer, state = self.owner._record(unit)
        self.snapshots.append((unit, side, native, producer, consumer, state, *counts))
        if native == 0:
            self.done.set()


def _ft_code_receipt(module: Any, class_name: str) -> dict[str, Any]:
    path = Path(module.__file__)
    source = path.read_bytes()
    compiled = compile(source, module.__file__, "exec", optimize=sys.flags.optimize)
    class_code = next(
        value
        for value in compiled.co_consts
        if isinstance(value, CodeType) and value.co_name == class_name
    )
    methods = {}
    if class_name == "_DirectPool":
        names = ("reserve", "finish", "_release_export")
    elif class_name == "_PoolWriter":
        names = ("begin_unit",)
    else:
        names = ("__init__", "__setstate__", "close", "unlink")
    for name in names:
        expected = next(
            value
            for value in class_code.co_consts
            if isinstance(value, CodeType) and value.co_name == name
        )
        actual = getattr(getattr(module, class_name), name).__code__
        methods[name] = {
            "co_names": actual.co_names,
            "bytecode_sha256": hashlib.sha256(actual.co_code).hexdigest(),
            "matches_source": actual.co_code == expected.co_code
            and actual.co_names == expected.co_names
            and actual.co_consts == expected.co_consts,
        }
    cached = getattr(module, "__cached__", None)
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "cached": cached,
        "pyc_sha256": hashlib.sha256(Path(cached).read_bytes()).hexdigest()
        if cached and Path(cached).is_file()
        else None,
        "methods": methods,
    }


@unittest.skipUnless(
    sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
    "native release contracts require a free-threaded build",
)
class DirectFreeThreadingContractTest(_DefaultsTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._receipt: dict[str, Any] = {"case": self.id(), "pid": os.getpid()}
        self._weak: dict[str, Any] = {}
        self.addCleanup(self._write_receipt)
        _cpu_defaults()
        self.assertEqual(sysconfig.get_config_var("Py_GIL_DISABLED"), 1)
        code = {
            "direct": _ft_code_receipt(_direct, "_DirectPool"),
            "pool": _ft_code_receipt(pool_module, "SharedMemorySegmentPool"),
            "ordinary_writer": _ft_code_receipt(pool_module, "_PoolWriter"),
        }
        native = importlib.import_module("spdl.io.lib._buffer")
        native_file = getattr(native, "__file__", None)
        image = Path(native_file or sys.executable)
        self._receipt.update(
            {
                "python": sys.version,
                "executable": sys.executable,
                "torch": torch.__version__,
                "Py_GIL_DISABLED": sysconfig.get_config_var("Py_GIL_DISABLED"),
                "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)(),
                "effective_code": code,
                "native_module_file": native_file,
                "native_module_origin": str(
                    getattr(getattr(native, "__spec__", None), "origin", None)
                ),
                "hashed_runtime_image": str(image),
                "runtime_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "phases": [],
            }
        )
        for receipt in code.values():
            self.assertTrue(
                all(value["matches_source"] for value in receipt["methods"].values()),
                receipt,
            )
        self.assertEqual(pool_module._CTRL.size, 42)
        self.assertFalse(hasattr(pipeline_module, "ArenaAllocationTicket"))
        self.assertIs(pipeline_module.ArenaReservation, ArenaReservation)

    def _pool(self, count: int = 1) -> SharedMemorySegmentPool:
        pool = SharedMemorySegmentPool(65536, count, acquire_timeout=10)
        self._receipt["acquire_timeout"] = pool._acquire_timeout
        self._receipt["count"] = count
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        return pool

    def _record(
        self, phase: str, pool: SharedMemorySegmentPool, probe: _NativeReleaseProbe
    ) -> None:
        with pool._close_lock:
            native = sum(pool._mapping_exports.values())
        self._receipt["phases"].append(
            {
                "phase": phase,
                "native_exports": native,
                "counts": probe.owner.counts(),
                "weak_live": {
                    name: ref() is not None for name, ref in self._weak.items()
                },
                "callback_snapshots": list(probe.snapshots),
            }
        )

    def _write_receipt(self) -> None:
        directory = Path("/tmp/track_a_FT_contract_receipts")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._testMethodName}.{os.getpid()}.{time.time_ns()}.json"
        path.write_text(json.dumps(self._receipt, sort_keys=True, indent=2) + "\n")

    def _wait_native(
        self, pool: SharedMemorySegmentPool, probe: _NativeReleaseProbe
    ) -> None:
        done = probe.done.wait(timeout=pool._acquire_timeout)
        self._record("native_wait_finished", pool, probe)
        self.assertTrue(done, self._receipt)
        self.assertEqual(sum(pool._mapping_exports.values()), 0, self._receipt)
        self.assertTrue(probe.snapshots)

    def _admit(self, pool: SharedMemorySegmentPool, unit_number: int) -> None:
        start = time.monotonic()
        with pool.reserve() as unit:
            self.assertEqual(unit._unit, unit_number)
            ticket = unit.seal()
        pool.discard(ticket)
        self.assertLessEqual(time.monotonic() - start, pool._acquire_timeout)

    def test_explicit_memoryview_release_drives_native_reclaim(self) -> None:
        """Live byte aliases gate reuse until actual native release callbacks."""
        pool = self._pool()
        with _NativeReleaseProbe(pool) as probe:
            with pool.reserve() as unit:
                output = unit.allocate_bytes(32)
                output[:] = b"a" * 32
                producer = memoryview(output)
                ticket = unit.seal()
            output.release()
            with pool.recover(ticket) as batch:
                consumer = memoryview(batch)
            del output, batch, unit
            self._weak = {
                "producer": weakref.ref(producer),
                "consumer": weakref.ref(consumer),
            }
            probe.done.clear()
            self._record("both_live_after_logical_exit", pool, probe)
            self.assertEqual(sum(pool._mapping_exports.values()), 2)
            self.assertEqual(pool._reclaimed, 0)
            self.assertEqual(bytes(producer), b"a" * 32)
            self.assertEqual(bytes(consumer), b"a" * 32)
            producer.release()
            self._record("consumer_only", pool, probe)
            self.assertEqual(pool._reclaimed, 0)
            self.assertFalse(probe.done.is_set())
            consumer.release()
            self._wait_native(pool, probe)
            self.assertEqual(pool._reclaimed, 1)
            self._admit(pool, 1)
            self._record("next_admitted", pool, probe)

    def test_torch_storage_owners_release_only_at_actual_native_free(self) -> None:
        """Escaped Torch storage owners survive wrapper and logical-context release."""
        pool = self._pool()
        with _NativeReleaseProbe(pool) as probe:
            with pool.reserve() as unit:
                data = unit.allocate_bytes(1024 * torch.int64.itemsize)
                output = torch.frombuffer(data.obj, dtype=torch.int64)
                output.fill_(41)
                producer = output.untyped_storage()
                ticket = unit.seal()
            data.release()
            context = pool.recover(ticket)
            with context as batch:
                consumer_tensor = torch.frombuffer(
                    batch.obj, dtype=torch.int64
                ).detach()
                consumer = consumer_tensor.untyped_storage()
                batch.release()
                self.assertEqual(consumer_tensor.tolist(), [41] * 1024)
            self._weak = {
                "producer_tensor": weakref.ref(output),
                "producer_storage": weakref.ref(producer),
                "consumer_tensor": weakref.ref(consumer_tensor),
                "consumer_storage": weakref.ref(consumer),
            }
            del data, output, batch, consumer_tensor, context, unit
            probe.done.clear()
            self._record("storage_live_after_wrapper_exit", pool, probe)
            self.assertEqual(pool._reclaimed, 0)
            self.assertEqual(sum(pool._mapping_exports.values()), 2)
            del producer
            self._record("consumer_storage_live", pool, probe)
            self.assertEqual(pool._reclaimed, 0)
            self.assertFalse(probe.done.is_set())
            del consumer
            self._wait_native(pool, probe)
            self.assertEqual(pool._reclaimed, 1)
            self._admit(pool, 1)
            self._record("next_admitted", pool, probe)

    def test_completed_later_unit_waits_for_old_native_export(self) -> None:
        """Later units reclaim in FIFO order after the oldest native release."""
        pool = self._pool(2)
        with _NativeReleaseProbe(pool) as probe:
            aliases = []
            tickets = []
            for value in (b"a", b"b"):
                with pool.reserve() as unit:
                    output = unit.allocate_bytes(32)
                    output[:] = value * 32
                    ticket = unit.seal()
                output.release()
                with pool.recover(ticket) as batch:
                    aliases.append(memoryview(batch))
                tickets.append(ticket)
                del output, batch, unit
            self._weak = {
                "old": weakref.ref(aliases[0]),
                "later": weakref.ref(aliases[1]),
            }
            probe.done.clear()
            aliases[1].release()
            self._record("later_complete_old_live", pool, probe)
            self.assertEqual(pool._reclaimed, 0)
            self.assertEqual(probe.owner._record(tickets[1].unit)[1:3], (0, 0))
            self.assertEqual(probe.owner._record(tickets[1].unit)[3], _direct._CONSUMED)
            self.assertEqual(bytes(aliases[0]), b"a" * 32)
            self.assertFalse(probe.done.is_set())
            aliases[0].release()
            self._wait_native(pool, probe)
            self.assertEqual(probe.owner.counts(), (2, 2, 2, 2))
            self._admit(pool, 2)
            self._record("next_admitted", pool, probe)

    def test_native_free_does_not_finish_an_active_logical_claim(self) -> None:
        """An active logical claim gates admission even after native release."""
        pool = self._pool()
        with _NativeReleaseProbe(pool) as probe:
            with pool.reserve() as unit:
                output = unit.allocate_bytes(32)
                ticket = unit.seal()
            output.release()
            context = pool.recover(ticket)
            batch = context.__enter__()
            self._weak = {"consumer": weakref.ref(batch)}
            probe.done.clear()
            batch.release()
            self._wait_native(pool, probe)
            self.assertEqual(pool._reclaimed, 0)
            entered = threading.Event()
            original = pool._wait_space

            def wait(*args: Any, **kwargs: Any) -> Any:
                entered.set()
                return original(*args, **kwargs)

            with (
                patch.object(pool, "_wait_space", wait),
                ThreadPoolExecutor(max_workers=1) as executor,
            ):
                pending = executor.submit(pool.reserve)
                self.assertTrue(entered.wait(timeout=pool._acquire_timeout))
                self.assertFalse(pending.done())
                self._record("native_zero_logical_claim_live", pool, probe)
                context.__exit__(None, None, None)
                next_unit = pending.result(timeout=pool._acquire_timeout)
                with next_unit:
                    self.assertEqual(next_unit._unit, 1)
            self._record("logical_exit_next_admitted", pool, probe)
