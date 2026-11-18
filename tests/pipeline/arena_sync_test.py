# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Non-ACK wake, admission and actual dead-waiter/native-lifetime regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import os
import sys
import sysconfig
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import CodeType
from typing import Any, cast
from unittest import skipUnless
from unittest.mock import patch

import torch
from parameterized import parameterized
from spdl.pipeline import (
    config,
    SharedMemoryRingBuffer,
    SharedMemorySegmentPool,
    StatsQueue,
    TaskStatsHook,
)
from spdl.pipeline._arena import _direct, _pool, _ring, _sync

from ._defaults import _DefaultsTestCase

_Arena = SharedMemorySegmentPool | SharedMemoryRingBuffer


def _cpu_defaults() -> None:
    config.set_default_build_callback(None)
    config.set_default_background_tasks([])
    config.set_default_hook_class(TaskStatsHook)
    config.set_default_queue_class(StatsQueue)
    torch.set_num_threads(1)


def _legacy_wait(arena: _Arena, waiting: Any) -> None:
    condition = cast(Any, arena._space_cv)
    original = condition._wait_semaphore.acquire

    def acquire(*args: Any, **kwargs: Any) -> Any:
        waiting.set()
        return original(*args, **kwargs)

    with patch.object(condition._wait_semaphore, "acquire", acquire), condition:
        condition.wait(timeout=arena._acquire_timeout)
    raise AssertionError("legacy waiter unexpectedly returned")


def _writer_waiter(
    arena: _Arena, units: int, legacy: bool, result: Any, waiting: Any, finish: Any
) -> None:
    _cpu_defaults()
    writer = arena.open_writer()
    for value in range(1, units + 1):
        writer.begin_unit()
        offset, size = writer.write_binary(bytes([value]) * 8192)
        result.put((os.getpid(), offset, size, writer.commit_unit()))
    if legacy:
        _legacy_wait(arena, waiting)
    original = arena._wait_space

    def wait(*args: Any, **kwargs: Any) -> Any:
        waiting.set()
        return original(*args, **kwargs)

    with patch.object(arena, "_wait_space", wait):
        try:
            writer.begin_unit()
            if isinstance(arena, SharedMemoryRingBuffer):
                writer.write_binary(b"x" * 8192)
        except BufferError:
            result.put((os.getpid(), "BufferError"))
        else:
            raise AssertionError("full writer unexpectedly admitted")
    if not finish.wait(arena._acquire_timeout):
        raise TimeoutError("caller did not reap cancelled writer")


def _holder_writer_close(arena: _Arena, result: Any) -> None:
    writer = arena.open_writer()
    writer.begin_unit()
    offset, size = writer.write_binary(b"h" * 8192)
    span = writer.commit_unit()
    del writer
    arena.close()
    try:
        arena.unlink()
    except RuntimeError:
        rejected = True
    else:
        rejected = False
    result.put((os.getpid(), offset, size, span, rejected))


def _effective_code(module: Any, owner: str | None, method: str) -> dict[str, Any]:
    source = Path(module.__file__).read_bytes()
    container = compile(source, module.__file__, "exec", optimize=sys.flags.optimize)
    target = module
    if owner is not None:
        container = next(
            value
            for value in container.co_consts
            if isinstance(value, CodeType) and value.co_name == owner
        )
        target = getattr(module, owner)
    expected = next(
        value
        for value in container.co_consts
        if isinstance(value, CodeType) and value.co_name == method
    )
    actual = getattr(target, method).__code__
    return {
        "co_names": actual.co_names,
        "bytecode_sha256": hashlib.sha256(actual.co_code).hexdigest(),
        "matches_source": actual.co_code == expected.co_code
        and actual.co_names == expected.co_names
        and actual.co_consts == expected.co_consts,
    }


class ArenaSyncTest(_DefaultsTestCase):
    def setUp(self) -> None:
        super().setUp()
        _cpu_defaults()
        self.workers: list[Any] = []
        self.receipt: dict[str, Any] = {
            "case": self.id(),
            "pid": os.getpid(),
            "python": sys.version,
            "torch": torch.__version__,
            "Py_GIL_DISABLED": sysconfig.get_config_var("Py_GIL_DISABLED"),
            "source_sha256": {
                module.__name__: hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (_sync, _pool, _ring, _direct)
            },
            "observations": [],
        }
        self.addCleanup(self._save_receipt)
        self.receipt["effective_code"] = [
            _effective_code(_sync, None, "_signal_space"),
            _effective_code(_pool, "_PoolWriter", "begin_unit"),
            _effective_code(_ring, "_RingWriter", "write_binary"),
            _effective_code(_direct, "_DirectPool", "reserve"),
        ]
        self.assertTrue(
            all(item["matches_source"] for item in self.receipt["effective_code"])
        )
        self.assertEqual(_sync._signal_space.__code__.co_names, ("acquire", "release"))

    def _save_receipt(self) -> None:
        directory = Path("/tmp/track_a_nonack_receipts")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._testMethodName}.{os.getpid()}.{time.time_ns()}.json"
        path.write_text(json.dumps(self.receipt, indent=2, sort_keys=True) + "\n")

    def _arena(self, kind: str, count: int = 1, timeout: float = 10) -> _Arena:
        arena: _Arena = (
            SharedMemorySegmentPool(65536, count, acquire_timeout=timeout)
            if kind == "pool"
            else SharedMemoryRingBuffer(8192 * count, acquire_timeout=timeout)
        )
        self.addCleanup(self._unlink_reaped, arena)
        self.addCleanup(arena.close)
        return arena

    def _unlink_reaped(self, arena: _Arena) -> None:
        self.assertTrue(all(not p.is_alive() for p in self.workers))
        arena.unlink()

    def _queue(self) -> Any:
        result = mp.get_context("spawn").Queue()
        self.addCleanup(result.join_thread)
        self.addCleanup(result.close)
        return result

    def _spawn(self, target: Any, args: tuple[Any, ...]) -> Any:
        process = mp.get_context("spawn").Process(target=target, args=args)
        process.start()
        self.workers.append(process)

        def reap() -> None:
            if process.is_alive():
                process.kill()
            process.join(10)
            self.assertFalse(process.is_alive())

        self.addCleanup(reap)
        return process

    def _kill_reap(self, process: Any, timeout: float) -> None:
        self.assertTrue(process.is_alive())
        process.kill()
        process.join(timeout)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, -9)
        self.assertFalse(_sync._process_exists(process.pid))
        self.receipt["observations"].append(
            {"reaped_pid": process.pid, "exitcode": process.exitcode}
        )

    def _observe(self, arena: _Arena, label: str) -> None:
        if isinstance(arena, SharedMemorySegmentPool):
            state = {
                "_published": arena._published,
                "_reclaimed": arena._reclaimed,
                "delivered": arena._read_cursor,
                "native_exports": sum(arena._mapping_exports.values()),
            }
        else:
            state = {"_head": arena._head, "_tail": arena._tail}
        self.receipt["observations"].append({"label": label, **state})

    @parameterized.expand([("pool",), ("ring",)])
    def test_same_pid_holder_close_keeps_creator_and_signal_live(
        self, kind: str
    ) -> None:
        """A same-PID holder closes locally without cancelling creator admission."""
        arena = self._arena(kind)
        holder = copy.copy(arena)
        holder.close()
        self.assertTrue(holder._closed)
        self.assertFalse(arena._shutdown)
        with self.assertRaisesRegex(RuntimeError, "creator"):
            holder.unlink()
        writer = arena.open_writer()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 8192)
        span = writer.commit_unit()
        reader = arena.open_reader()
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual(bytes(view), b"a" * 8192)
        view.release()
        arena._signal_space()
        self.assertTrue(arena._space_signal.acquire(False))

    @parameterized.expand([("pool",), ("ring",)])
    def test_unobserved_wakes_coalesce_to_one_token(self, kind: str) -> None:
        """Repeated wakes without a receiver leave exactly one non-ACK token."""
        arena = self._arena(kind)
        with (
            patch.object(
                arena._space_cv, "notify", side_effect=AssertionError("ACK notify")
            ),
            patch.object(
                arena._space_cv,
                "notify_all",
                side_effect=AssertionError("ACK notify_all"),
            ),
        ):
            for _ in range(20):
                arena._signal_space()
            self.assertTrue(arena._space_signal.acquire(False))
            self.assertFalse(arena._space_signal.acquire(False))
            arena.close()
        self.assertTrue(arena._closed)

    @parameterized.expand([("pool",), ("ring",)])
    @skipUnless(os.name == "posix", "requires actual POSIX process reap")
    def test_public_close_cancels_live_spawned_writer(self, kind: str) -> None:
        """Public creator close wakes a live spawned capacity waiter without ACK."""
        arena = self._arena(kind)
        ctx = mp.get_context("spawn")
        result, waiting, finish = self._queue(), ctx.Event(), ctx.Event()
        process = self._spawn(
            _writer_waiter, (arena, 1, False, result, waiting, finish)
        )
        self.assertEqual(result.get(timeout=arena._acquire_timeout)[0], process.pid)
        self.assertTrue(waiting.wait(arena._acquire_timeout))
        arena.close()
        self.assertEqual(
            result.get(timeout=arena._acquire_timeout), (process.pid, "BufferError")
        )
        self.assertTrue(process.is_alive())
        self.assertFalse(arena._closed)
        finish.set()
        process.join(arena._acquire_timeout)
        self.assertEqual(process.exitcode, 0)
        arena.close()
        self.assertTrue(arena._closed)

    @parameterized.expand(
        [("pool", False), ("ring", False), ("pool", True), ("ring", True)]
    )
    @skipUnless(os.name == "posix", "requires actual POSIX process reap")
    def test_reaped_capacity_waiter_needs_no_close_ack(
        self, kind: str, legacy: bool
    ) -> None:
        """A killed semaphore or legacy Condition waiter cannot stall creator close."""
        arena = self._arena(kind)
        ctx = mp.get_context("spawn")
        result, waiting, finish = self._queue(), ctx.Event(), ctx.Event()
        process = self._spawn(
            _writer_waiter, (arena, 1, legacy, result, waiting, finish)
        )
        self.assertEqual(result.get(timeout=arena._acquire_timeout)[0], process.pid)
        self.assertTrue(waiting.wait(arena._acquire_timeout))
        self._kill_reap(process, arena._acquire_timeout)
        condition = cast(Any, arena._space_cv)
        before = (
            (condition._sleeping_count.get_value(), condition._woken_count.get_value())
            if legacy
            else None
        )
        start = time.monotonic()
        with (
            patch.object(condition, "notify", side_effect=AssertionError("ACK notify")),
            patch.object(
                condition, "notify_all", side_effect=AssertionError("ACK notify_all")
            ),
        ):
            arena.close()
        limit = (
            _pool._CONTROL_LOCK_TIMEOUT
            if kind == "pool"
            else _ring._CONTROL_LOCK_TIMEOUT
        )
        self.assertLessEqual(time.monotonic() - start, limit)
        self.assertTrue(arena._closed)
        if legacy:
            self.assertEqual(before, (1, 0))
            self.assertEqual(
                (
                    condition._sleeping_count.get_value(),
                    condition._woken_count.get_value(),
                ),
                before,
            )
        self.receipt["observations"].append(
            {"legacy_condition_counts": before, "close_completed": True}
        )

    @parameterized.expand([("pool", False), ("pool", True), ("ring", False)])
    def test_timeout_final_predicate_can_admit_real_reclaimed_space(
        self, kind: str, direct: bool
    ) -> None:
        """An actual timed-out wake still checks real capacity before failing admission."""
        arena = self._arena(kind, timeout=0.05)
        if direct:
            pool = cast(SharedMemorySegmentPool, arena)
            with pool.reserve() as unit:
                ticket = unit.seal()
            release = partial(pool.discard, ticket)
            admit = pool.reserve
        else:
            writer, reader = arena.open_writer(), arena.open_reader()
            writer.begin_unit()
            writer.write_binary(b"a" * 8192)
            span = writer.commit_unit()
            if isinstance(arena, SharedMemoryRingBuffer):
                writer.begin_unit()
            release = partial(reader.end_unit, span, [])
            admit = (
                writer.begin_unit
                if kind == "pool"
                else lambda: writer.write_binary(b"b" * 8192)
            )
        original = arena._wait_space
        calls = []

        def expired(deadline: float, control_deadline: float) -> bool:
            result = original(deadline, control_deadline)
            calls.append((result, deadline, control_deadline))
            if not result:
                release()
            return result

        with patch.object(arena, "_wait_space", expired):
            admitted = admit()
        self.assertEqual([value[0] for value in calls], [False])
        self.assertLessEqual(calls[0][2] - calls[0][1], _pool._CONTROL_LOCK_TIMEOUT)
        if direct:
            self.assertIsInstance(admitted, _direct.ArenaReservation)
            cast(_direct.ArenaReservation, admitted).__exit__(None, None, None)
        self._observe(arena, "actual_timeout_then_final_admission")

    @parameterized.expand([("pool",), ("ring",)])
    def test_timeout_then_new_attempt_uses_real_returned_space(self, kind: str) -> None:
        """A failed capacity deadline preserves state for a new real-space admission."""
        arena = self._arena(kind, timeout=0.05)
        writer, reader = arena.open_writer(), arena.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"a" * 8192)
        span = writer.commit_unit()
        if kind == "ring":
            writer.begin_unit()
        admit = (
            writer.begin_unit
            if kind == "pool"
            else partial(writer.write_binary, b"b" * 8192)
        )
        before = (
            (arena._published, arena._read_cursor)
            if isinstance(arena, SharedMemorySegmentPool)
            else (arena._head, arena._tail)
        )
        with self.assertRaises(BufferError):
            admit()
        after = (
            (arena._published, arena._read_cursor)
            if isinstance(arena, SharedMemorySegmentPool)
            else (arena._head, arena._tail)
        )
        self.assertEqual(after, before)
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual(bytes(view), b"a" * 8192)
        view.release()
        admit()
        self._observe(arena, "new_attempt_after_actual_consumption")

    @parameterized.expand([("pool",), ("ring",)])
    def test_fresh_writer_rejects_unread_but_existing_writer_can_queue(
        self, kind: str
    ) -> None:
        """Only fresh channels reject unread work; delivered aliases do not block rebind."""
        arena = self._arena(kind, count=2)
        writer, reader = arena.open_writer(), arena.open_reader()
        positions = []
        for byte in (b"a", b"b"):
            writer.begin_unit()
            offset, size = writer.write_binary(byte * 8192)
            positions.append((offset, size, writer.commit_unit()))
            with self.assertRaisesRegex(RuntimeError, "unread"):
                arena.open_writer()
        aliases = []
        for offset, size, span in positions:
            aliases.append(reader.read_binary(offset, size))
            reader.end_unit(span, [])
        self.assertEqual([bytes(v) for v in aliases], [b"a" * 8192, b"b" * 8192])
        fresh = arena.open_writer()
        if isinstance(arena, SharedMemorySegmentPool):
            self.assertEqual(
                (arena._published, arena._read_cursor, arena._reclaimed), (2, 2, 0)
            )
        for view in aliases:
            view.release()
        fresh.begin_unit()
        fresh.write_binary(b"unused")
        fresh.abort_unit()
        self.assertIsNotNone(arena.open_writer())
        self._observe(arena, "delivered_aliases_then_unused_abort")

    @parameterized.expand([("pool",), ("ring",)])
    @skipUnless(os.name == "posix", "requires actual POSIX process reap")
    def test_spawned_holder_close_does_not_cancel_creator(self, kind: str) -> None:
        """A spawned holder closes locally while creator delivery and reuse remain valid."""
        arena = self._arena(kind)
        result = self._queue()
        process = self._spawn(_holder_writer_close, (arena, result))
        pid, offset, size, span, rejected = result.get(timeout=arena._acquire_timeout)
        self.assertEqual(pid, process.pid)
        self.assertTrue(rejected)
        reader = arena.open_reader()
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual(bytes(view), b"h" * 8192)
        view.release()
        process.join(arena._acquire_timeout)
        self.assertEqual(process.exitcode, 0)
        self.assertFalse(arena._shutdown)
        self.assertIsNotNone(arena.open_writer())

    @parameterized.expand(
        [("bytes", False), ("torch", False), ("bytes", True), ("torch", True)]
    )
    @skipUnless(os.name == "posix", "requires actual POSIX process reap")
    def test_native_last_alias_reclaims_after_dead_writer(
        self, kind: str, legacy: bool
    ) -> None:
        """Last native release completes after a reaped waiter and admits correct new values."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool", count=2))
        ctx = mp.get_context("spawn")
        result, waiting, finish = self._queue(), ctx.Event(), ctx.Event()
        process = self._spawn(
            _writer_waiter, (pool, 2, legacy, result, waiting, finish)
        )
        positions = [result.get(timeout=pool._acquire_timeout) for _ in range(2)]
        self.assertTrue(all(row[0] == process.pid for row in positions))
        reader = pool.open_reader()
        first = reader.read_binary(positions[0][1], positions[0][2])
        reader.end_unit(positions[0][3], [])
        alias: Any = (
            torch.frombuffer(first.obj, dtype=torch.uint8)
            if kind == "torch"
            else memoryview(first)
        )
        first.release()
        native = sys.modules["spdl.io.lib._buffer"]
        native_file = getattr(native, "__file__", None)
        image = Path(native_file or sys.executable)
        self.receipt.update(
            {
                "native_module_file": native_file,
                "native_module_origin": str(
                    getattr(getattr(native, "__spec__", None), "origin", None)
                ),
                "hashed_runtime_image": str(image),
                "runtime_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)(),
            }
        )
        second = reader.read_binary(positions[1][1], positions[1][2])
        reader.end_unit(positions[1][3], [])
        self.assertEqual(bytes(second), bytes([2]) * 8192)
        second.release()
        self.assertTrue(waiting.wait(pool._acquire_timeout))
        self.assertEqual(
            (pool._published, pool._read_cursor, pool._reclaimed), (2, 2, 0)
        )
        self.assertEqual(reader._released, {1})
        self.assertEqual(
            alias.tolist() if kind == "torch" else bytes(alias),
            [1] * 8192 if kind == "torch" else bytes([1]) * 8192,
        )
        self._kill_reap(process, pool._acquire_timeout)
        native_done = threading.Event()
        original = reader.unit_released
        snapshots: list[tuple[int, ...]] = []

        def released(unit: int) -> None:
            original(unit)
            snapshots.append(
                (
                    unit,
                    pool._published,
                    pool._read_cursor,
                    pool._reclaimed,
                    sum(pool._mapping_exports.values()),
                )
            )
            if pool._reclaimed == 2:
                native_done.set()

        reference = weakref.ref(alias)
        with patch.object(reader, "unit_released", released):
            if kind == "torch":
                del alias
            else:
                alias.release()
            self.assertTrue(native_done.wait(pool._acquire_timeout))
        self.assertEqual(pool._reclaimed, 2)
        self.assertEqual(sum(pool._mapping_exports.values()), 0)
        writer = pool.open_writer()
        writer.begin_unit()
        offset, size = writer.write_binary(bytes([3]) * 8192)
        span = writer.commit_unit()
        third = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual(bytes(third), bytes([3]) * 8192)
        third.release()
        self.assertEqual(pool._reclaimed, 3)
        self.assertFalse(torch.cuda.is_initialized())
        self.receipt["observations"].append(
            {
                "native_callback_snapshots": snapshots,
                "weak_wrapper_live_after_release": reference() is not None,
                "legacy": legacy,
            }
        )

    def test_direct_multiple_waiters_receive_capacity_and_cancel_baton(self) -> None:
        """One-PID direct waiters share freed capacity and terminal cancellation wakes."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool", count=2))
        with pool.reserve() as a:
            first = a.seal()
        with pool.reserve() as b:
            second = b.seal()
        waiting: set[int] = set()
        lock = threading.Lock()
        all_waiting = threading.Event()
        admitted: set[int] = set()
        two_admitted = threading.Event()
        finish = threading.Event()
        original = pool._wait_space

        def wait(*args: Any, **kwargs: Any) -> Any:
            with lock:
                waiting.add(threading.get_ident())
                if len(waiting) == 3:
                    all_waiting.set()
            return original(*args, **kwargs)

        def reserve() -> str:
            try:
                with pool.reserve() as unit:
                    with lock:
                        admitted.add(unit._unit)
                        if len(admitted) == 2:
                            two_admitted.set()
                    if not finish.wait(pool._acquire_timeout):
                        raise TimeoutError(
                            "caller did not finish admitted reservations"
                        )
                return "admitted"
            except BufferError:
                return "cancelled"

        with (
            patch.object(pool, "_wait_space", wait),
            ThreadPoolExecutor(max_workers=3) as executor,
        ):
            futures = [executor.submit(reserve) for _ in range(3)]
            self.assertTrue(all_waiting.wait(pool._acquire_timeout))
            pool.discard(second)
            self.assertFalse(two_admitted.is_set())
            pool.discard(first)
            self.assertTrue(two_admitted.wait(pool._acquire_timeout))
            pool.close()
            finish.set()
            self.assertCountEqual(
                [future.result(timeout=pool._acquire_timeout) for future in futures],
                ["admitted", "admitted", "cancelled"],
            )
        self.assertEqual(admitted, {2, 3})
        pool.close()
        self.assertTrue(pool._closed)

    def test_exception_after_token_take_hands_real_capacity_to_peer(self) -> None:
        """A failed token recipient forwards a wake before another waiter gets capacity."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        with pool.reserve() as unit:
            ticket = unit.seal()
        first_wait, second_wait = threading.Event(), threading.Event()
        taken, release_fault, peer_took = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        take, waiting = pool._space_signal.acquire, pool._wait_space
        fault_thread: list[int] = []

        def acquire(*args: Any, **kwargs: Any) -> Any:
            result = take(*args, **kwargs)
            if kwargs.get("timeout") is not None and result:
                if threading.get_ident() in fault_thread:
                    taken.set()
                    if not release_fault.wait(pool._acquire_timeout):
                        raise TimeoutError("caller did not release injected fault")
                    raise MemoryError("after actual token take")
                peer_took.set()
            return result

        def wait(*args: Any, **kwargs: Any) -> Any:
            (first_wait if threading.get_ident() in fault_thread else second_wait).set()
            return waiting(*args, **kwargs)

        def fail() -> Any:
            fault_thread.append(threading.get_ident())
            return pool.reserve()

        with (
            patch.object(pool._space_signal, "acquire", acquire),
            patch.object(pool, "_wait_space", wait),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            bad = executor.submit(fail)
            self.assertTrue(first_wait.wait(pool._acquire_timeout))
            pool.discard(ticket)
            self.assertTrue(taken.wait(pool._acquire_timeout))
            with pool.reserve() as blocker:
                self.assertEqual(blocker._unit, 1)
                good = executor.submit(pool.reserve)
                self.assertTrue(second_wait.wait(pool._acquire_timeout))
                release_fault.set()
                with self.assertRaisesRegex(MemoryError, "actual token take"):
                    bad.result(timeout=pool._acquire_timeout)
                self.assertTrue(peer_took.wait(pool._acquire_timeout))
                self.assertFalse(good.done())
            admitted = good.result(timeout=pool._acquire_timeout)
            with admitted:
                self.assertEqual(admitted._unit, 2)
        self.assertEqual(pool._reclaimed, 1)
