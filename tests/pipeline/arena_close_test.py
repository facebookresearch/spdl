# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal cancellation, conservative retirement and reusable ordinary endpoints."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import weakref
from multiprocessing import shared_memory
from pathlib import Path
from types import CodeType, SimpleNamespace
from typing import Any, cast
from unittest import skipUnless
from unittest.mock import patch

import torch
from parameterized import parameterized
from spdl.pipeline import (
    config,
    PipelineBuilder,
    SharedMemoryRingBuffer,
    SharedMemorySegmentPool,
    StatsQueue,
    TaskStatsHook,
)
from spdl.pipeline._arena import _pool as pool_module, _ring as ring_module
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

from ._defaults import _DefaultsTestCase

_Arena = SharedMemorySegmentPool | SharedMemoryRingBuffer


def _cpu_defaults() -> None:
    config.set_default_build_callback(None)
    config.set_default_background_tasks([])
    config.set_default_hook_class(TaskStatsHook)
    config.set_default_queue_class(StatsQueue)
    torch.set_num_threads(1)


def _metadata_worker(
    pool: SharedMemorySegmentPool, result: Any, finish: Any, close_holder: bool = False
) -> None:
    _cpu_defaults()
    with pool.reserve() as unit:
        ticket = unit.seal()
    if close_holder:
        pool.close()
        if not pool._closed:
            raise AssertionError("metadata sender holder did not close locally")
    result.put(
        (os.getpid(), ticket, {"pid": os.getpid(), "cuda": torch.cuda.is_initialized()})
    )
    if not finish.wait(30):
        raise TimeoutError("caller did not finish metadata worker")


def _blocked_direct_worker(
    pool: SharedMemorySegmentPool, result: Any, waiting: Any, finish: Any
) -> None:
    _cpu_defaults()
    with pool.reserve() as unit:
        ticket = unit.seal()
    original = pool._wait_space

    def wait(*args: Any, **kwargs: Any) -> Any:
        waiting.set()
        return original(*args, **kwargs)

    result.put((os.getpid(), ticket))
    with patch.object(pool, "_wait_space", wait):
        try:
            pool.reserve()
        except BufferError as error:
            result.put((os.getpid(), type(error).__name__))
        else:
            raise AssertionError("cancelled worker acquired another reservation")
    if not finish.wait(30):
        raise TimeoutError("caller did not reap blocked worker")


def _late_endpoint(pool: SharedMemorySegmentPool, result: Any, start: Any) -> None:
    if not start.wait(30):
        raise TimeoutError("caller did not permit late endpoint")
    try:
        pool.reserve()
    except (BufferError, RuntimeError) as error:
        result.put((os.getpid(), type(error).__name__, pool._shutdown))
    else:
        raise AssertionError("late endpoint reopened cancelled pool")


def _ordinary_endpoint(arena: _Arena, side: str, result: Any, finish: Any) -> None:
    endpoint = arena.open_writer() if side == "writer" else arena.open_reader()
    result.put((os.getpid(), type(endpoint).__name__))
    if not finish.wait(30):
        raise TimeoutError("caller did not finish ordinary endpoint")


def _ordinary_publish(arena: _Arena, value: bytes, result: Any, finish: Any) -> None:
    writer = arena.open_writer()
    writer.reset()
    writer.begin_unit()
    offset, nbytes = writer.write_binary(value)
    span = writer.commit_unit()
    result.put((os.getpid(), offset, nbytes, span))
    if not finish.wait(30):
        raise TimeoutError("caller did not finish ordinary writer")


def _ordinary_consume(
    pool: SharedMemorySegmentPool, offset: int, nbytes: int, span: int, result: Any
) -> None:
    reader = pool.open_reader()
    view = reader.read_binary(offset, nbytes)
    reader.end_unit(span, [])
    result.put((os.getpid(), bytes(view), reader._next))
    view.release()


def _probe_endpoint(arena: _Arena, side: str, result: Any) -> None:
    try:
        if side == "direct":
            cast(SharedMemorySegmentPool, arena).reserve()
        elif side == "writer":
            arena.open_writer()
        else:
            arena.open_reader()
    except RuntimeError:
        result.put((os.getpid(), "RuntimeError"))
    else:
        result.put((os.getpid(), "unexpected endpoint"))


def _holder_publish_close(
    pool: SharedMemorySegmentPool, kind: str, result: Any, finish: Any
) -> None:
    _cpu_defaults()
    output: torch.Tensor | None = None
    with pool.reserve() as unit:
        data = unit.allocate_bytes(8192)
        if kind == "torch":
            output = torch.frombuffer(data.obj, dtype=torch.int64)
            output.fill_(41)
            alias = output.detach()
        else:
            data[:] = b"a" * 8192
            alias = memoryview(data)
        ticket = unit.seal()
    data.release()
    del output, data, unit
    pool.close()
    try:
        pool.unlink()
    except RuntimeError:
        unlink_rejected = True
    else:
        unlink_rejected = False
    result.put((os.getpid(), ticket, unlink_rejected, torch.cuda.is_initialized()))
    if not finish.wait(30):
        raise TimeoutError("caller did not release native sender")
    actual = alias.tolist() if kind == "torch" else bytes(alias)
    expected = [41] * 1024 if kind == "torch" else b"a" * 8192
    if actual != expected:
        raise AssertionError("holder close invalidated native sender alias")
    del alias
    pool.close()
    result.put((os.getpid(), pool._closed))


def _ring_holder_publish_close(
    ring: SharedMemoryRingBuffer, result: Any, finish: Any
) -> None:
    writer = ring.open_writer()
    writer.begin_unit()
    offset, size = writer.write_binary(b"holder data")
    span = writer.commit_unit()
    del writer
    ring.close()
    try:
        ring.unlink()
    except RuntimeError:
        unlink_rejected = True
    else:
        unlink_rejected = False
    result.put((os.getpid(), offset, size, span, unlink_rejected, ring._closed))
    if not finish.wait(30):
        raise TimeoutError("caller did not reap ring holder")


def _fork_close_inherited(arena: _Arena, result: Any) -> None:
    try:
        arena.close()
    except RuntimeError as error:
        result.put(str(error))
    else:
        result.put("unexpected inherited close")


def _hold_condition(arena: _Arena, ready: Any) -> None:
    arena._space_cv.acquire()
    ready.set()
    os._exit(17)


def _public_payload(sequence: int) -> dict[str, Any]:
    _cpu_defaults()
    return {
        "id": sequence,
        "value": torch.full((1024,), sequence, dtype=torch.int64),
        "pid": os.getpid(),
        "cuda": torch.cuda.is_initialized(),
    }


def _loaded_code_receipt(module: Any, class_name: str) -> dict[str, Any]:
    source = Path(module.__file__)
    packet = source.read_bytes()
    compiled = compile(packet, module.__file__, "exec", optimize=sys.flags.optimize)
    class_code = next(
        value
        for value in compiled.co_consts
        if isinstance(value, CodeType) and value.co_name == class_name
    )
    methods = {}
    for name in ("__init__", "__setstate__", "close", "unlink"):
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
        "source": str(source),
        "source_sha256": hashlib.sha256(packet).hexdigest(),
        "cached": cached,
        "pyc_sha256": hashlib.sha256(Path(cached).read_bytes()).hexdigest()
        if cached and Path(cached).is_file()
        else None,
        "methods": methods,
    }


class ArenaCloseTest(_DefaultsTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._workers: list[Any] = []
        self._reported_pids: set[int] = set()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        _cpu_defaults()

    def _arena(self, kind: str, timeout: float = 0) -> _Arena:
        if kind == "pool":
            arena: _Arena = SharedMemorySegmentPool(65536, 2, acquire_timeout=timeout)
        else:
            arena = SharedMemoryRingBuffer(65536, acquire_timeout=timeout)
        self.addCleanup(self._unlink_after_reap, arena)
        self.addCleanup(arena.close)
        return arena

    def _unlink_after_reap(self, arena: _Arena) -> None:
        self.assertTrue(all(not process.is_alive() for process in self._workers))
        self.assertTrue(
            all(not pool_module._process_exists(pid) for pid in self._reported_pids)
        )
        arena.unlink()

    def _queue(self) -> Any:
        result = mp.get_context("spawn").Queue()
        self.addCleanup(result.join_thread)
        self.addCleanup(result.close)
        return result

    def _spawn(self, target: Any, args: tuple[Any, ...]) -> Any:
        process = mp.get_context("spawn").Process(target=target, args=args)
        process.start()
        self._workers.append(process)

        def reap() -> None:
            if process.is_alive():
                process.kill()
            process.join(10)
            self.assertFalse(process.is_alive())

        self.addCleanup(reap)
        return process

    @staticmethod
    def _os_names(arena: _Arena) -> list[str]:
        return (
            [arena.arena_id, *arena._seg_names]
            if isinstance(arena, SharedMemorySegmentPool)
            else [arena.name]
        )

    @staticmethod
    def _remove_os_names(names: list[str]) -> None:
        """Test-only cleanup of names a deliberately unrecoverable arena retains."""
        for name in names:
            try:
                segment = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                continue
            try:
                segment.unlink()
            finally:
                segment.close()

    def _names_exist(self, arena: _Arena) -> None:
        names = self._os_names(arena)
        for name in names:
            segment = shared_memory.SharedMemory(name=name)
            try:
                self.assertEqual(segment.name, name)
            finally:
                segment.close()

    def _retired(self, arena: _Arena) -> None:
        self.assertTrue(arena._closed)
        if isinstance(arena, SharedMemorySegmentPool):
            self.assertIsNone(arena._ctrl_buf)
            self.assertIsNone(arena._ctrl)
            self.assertIsNone(arena._segs)
        else:
            self.assertIsNone(arena._buf)
            self.assertIsNone(arena._shm)

    def _retained(self, arena: _Arena) -> None:
        self.assertFalse(arena._closed)
        self.assertTrue(arena._shutdown)
        if isinstance(arena, SharedMemorySegmentPool):
            self.assertIsNotNone(arena._ctrl_buf)
            self.assertIsNotNone(arena._ctrl)
            self.assertIn(arena, pool_module._REGISTRATION_QUARANTINE)
        else:
            self.assertIsNotNone(arena._buf)
            self.assertIsNotNone(arena._shm)
            self.assertIn(arena, ring_module._RETIREMENT_ROOTS)
        self._names_exist(arena)

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_metadata_sender_holder_close_preserves_creator_claim(self) -> None:
        """A closed metadata sender preserves the creator's shared transfer hold."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        result, finish = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_metadata_worker, (pool, result, finish, True))
        pid, ticket, metadata = result.get(timeout=30)
        self.assertEqual(pid, process.pid)
        self.assertFalse(pool._shutdown)
        self._names_exist(pool)
        with pool.recover(ticket) as batch:
            self.assertEqual(batch.nbytes, 0)
            self.assertEqual(metadata, {"pid": pid, "cuda": False})
            self.assertEqual(pool._reclaimed, 0)
        self.assertEqual(pool._reclaimed, 1)
        finish.set()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        pool.close()
        self._retired(pool)

    @parameterized.expand([("bytes",), ("torch",)])
    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_native_sender_holder_close_preserves_creator_recovery(
        self, kind: str
    ) -> None:
        """Sender-holder close preserves shared transfer and its live native alias."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        result, finish = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_holder_publish_close, (pool, kind, result, finish))
        pid, ticket, unlink_rejected, cuda = result.get(timeout=30)
        self.assertEqual(pid, process.pid)
        self.assertTrue(unlink_rejected)
        self.assertFalse(cuda)
        self.assertFalse(pool._shutdown)
        self._names_exist(pool)
        actual: torch.Tensor | None = None
        with pool.recover(ticket) as batch:
            if kind == "torch":
                actual = torch.frombuffer(batch.obj, dtype=torch.int64)
                self.assertEqual(actual.tolist(), [41] * 1024)
            else:
                self.assertEqual(bytes(batch), b"a" * 8192)
            finish.set()
            self.assertEqual(result.get(timeout=30), (pid, True))
            process.join(30)
            self.assertEqual(process.exitcode, 0)
            self.assertFalse(pool._shutdown)
            self.assertEqual(pool._reclaimed, 0)
        del batch, actual
        self.assertEqual(pool._reclaimed, 1)
        pool.close()
        self._retired(pool)

    @parameterized.expand([("pool",), ("ring",)])
    def test_same_pid_copied_holder_close_does_not_cancel_creator(
        self, kind: str
    ) -> None:
        """A same-PID copied holder closes locally and cannot unlink creator names."""
        arena = self._arena(kind)
        holder = copy.copy(arena)
        self.addCleanup(holder.close)
        self.assertFalse(holder._owns_names)
        self.assertTrue(arena._owns_names)
        with self.assertRaisesRegex(RuntimeError, "creator"):
            holder.unlink()
        holder.close()
        self.assertTrue(holder._closed)
        self.assertFalse(arena._shutdown)
        self._names_exist(arena)
        with self.assertRaisesRegex(RuntimeError, "creator"):
            holder.unlink()
        writer, reader = arena.open_writer(), arena.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"creator remains usable")
        span = writer.commit_unit()
        view = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual(bytes(view), b"creator remains usable")
        view.release()
        arena.close()
        self._retired(arena)

    def test_same_pid_consumer_holder_close_retains_active_claim(self) -> None:
        """Consumer-holder close preserves a local active claim and escaped native alias."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        with pool.reserve() as unit:
            value = unit.allocate_bytes(32)
            value[:] = b"c" * 32
            ticket = unit.seal()
        value.release()
        holder = copy.copy(pool)
        self.addCleanup(holder.close)
        context = holder.recover(ticket)
        with context as batch:
            alias = memoryview(batch)
            holder.close()
            self.assertFalse(holder._closed)
            self.assertFalse(pool._shutdown)
            self.assertEqual(bytes(alias), b"c" * 32)
        del batch, context
        holder.close()
        self.assertFalse(holder._closed)
        alias.release()
        holder.close()
        self.assertTrue(holder._closed)
        self.assertFalse(pool._shutdown)
        self.assertEqual(pool._reclaimed, 1)
        with pool.reserve() as next_unit:
            self.assertEqual(next_unit._unit, 1)

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_ring_sender_holder_close_preserves_creator_read_and_reuse(self) -> None:
        """A ring holder closes locally without cancelling committed creator-visible data."""
        ring = cast(SharedMemoryRingBuffer, self._arena("ring"))
        result, finish = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_ring_holder_publish_close, (ring, result, finish))
        pid, offset, size, span, unlink_rejected, holder_closed = result.get(timeout=30)
        self.assertEqual(pid, process.pid)
        self.assertTrue(unlink_rejected)
        self.assertTrue(holder_closed)
        self.assertFalse(ring._shutdown)
        self._names_exist(ring)
        reader = ring.open_reader()
        copied = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual(bytes(copied), b"holder data")
        finish.set()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        writer = ring.open_writer()
        writer.begin_unit()
        offset, size = writer.write_binary(b"new creator data")
        span = writer.commit_unit()
        second = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual(bytes(second), b"new creator data")
        self.assertEqual(bytes(copied), b"holder data")
        ring.close()
        self._retired(ring)
        copied.release()
        second.release()

    @parameterized.expand([("pool",), ("ring",)])
    @skipUnless(os.name == "posix", "requires POSIX fork")
    def test_fork_close_rejects_before_inherited_locks(self, kind: str) -> None:
        """Inherited close rejects PID mismatch before shared or local locks and writes."""
        arena = self._arena(kind)
        context = mp.get_context("fork")
        result = context.Queue()
        self.addCleanup(result.join_thread)
        self.addCleanup(result.close)
        before = (
            bytes(arena._ctrl_view())
            if isinstance(arena, SharedMemorySegmentPool)
            else bytes(arena._ensure())
        )
        with arena._space_cv, arena._close_lock:
            process = context.Process(
                target=_fork_close_inherited, args=(arena, result)
            )
            process.start()
            self._workers.append(process)
            try:
                self.assertIn("fork", result.get(timeout=10))
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            finally:
                if process.is_alive():
                    process.kill()
                    process.join(10)
        after = (
            bytes(arena._ctrl_view())
            if isinstance(arena, SharedMemorySegmentPool)
            else bytes(arena._ensure())
        )
        self.assertEqual(after, before)
        self.assertFalse(arena._shutdown)

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_metadata_worker_close_before_join_retains_control(self) -> None:
        """Metadata-only worker control survives close until actual reap and retry."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        result, finish = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_metadata_worker, (pool, result, finish))
        pid, ticket, metadata = result.get(timeout=30)
        self.assertEqual(pid, process.pid)
        self.assertTrue(process.is_alive())
        with pool.recover(ticket) as batch:
            self.assertEqual(batch.nbytes, 0)
            self.assertEqual(metadata, {"pid": pid, "cuda": False})
        for _ in range(3):
            pool.close()
            self._retained(pool)
            self.assertTrue(process.is_alive())
        finish.set()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        pool.close()
        self._retired(pool)
        self._names_exist(pool)
        name = pool.arena_id
        pool.unlink()
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_public_close_wakes_actual_spawn_reservation(self) -> None:
        """Public close wakes a blocked spawned reserve and retains its live control."""
        pool = SharedMemorySegmentPool(65536, 1, acquire_timeout=30)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        ctx = mp.get_context("spawn")
        result, waiting, finish = self._queue(), ctx.Event(), ctx.Event()
        process = self._spawn(_blocked_direct_worker, (pool, result, waiting, finish))
        pid, ticket = result.get(timeout=30)
        self.assertEqual(pid, process.pid)
        self.assertEqual(ticket.arena_id, pool.arena_id)
        self.assertTrue(waiting.wait(30))
        start = time.monotonic()
        pool.close()
        self.assertEqual(result.get(timeout=10), (pid, "BufferError"))
        self.assertLess(time.monotonic() - start, 10)
        self._retained(pool)
        pool.close()
        self._retained(pool)
        finish.set()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        pool.close()
        self._retired(pool)

    def test_claim_started_before_close_can_enter_and_finish(self) -> None:
        """Immediate claims survive cancellation; new recoveries cannot begin afterward."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        with pool.reserve() as first:
            claimed = first.seal()
        with pool.reserve() as second:
            unread = second.seal()
        context = pool.recover(claimed)
        pool.close()
        self._retained(pool)
        with self.assertRaises((BufferError, RuntimeError)):
            pool.recover(unread)
        with self.assertRaises((BufferError, RuntimeError)):
            pool.reserve()
        with context as batch:
            self.assertEqual(batch.nbytes, 0)
            pool.close()
            self._retained(pool)
        pool.close()
        self._retired(pool)
        with self.assertRaises((BufferError, RuntimeError)):
            pool.recover(unread)
        with self.assertRaises(RuntimeError):
            context.__enter__()

    def test_native_storage_and_claim_each_gate_close(self) -> None:
        """Native aliases and the active caller claim independently defer final retirement."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        with pool.reserve() as unit:
            data = unit.allocate_bytes(16 * torch.int64.itemsize)
            tensor = torch.frombuffer(data.obj, dtype=torch.int64)
            tensor.fill_(37)
            producer = tensor.untyped_storage()
            ticket = unit.seal()
        data.release()
        context = pool.recover(ticket)
        with context as batch:
            recovered = torch.frombuffer(batch.obj, dtype=torch.int64)
            alias = torch.empty(0, dtype=torch.int64).set_(recovered.untyped_storage())
            batch.release()
            del data, tensor, producer, recovered, unit
            pool.close()
            self._retained(pool)
            self.assertTrue(
                torch.equal(alias, torch.full((16,), 37, dtype=torch.int64))
            )
        del batch, context
        pool.close()
        self._retained(pool)
        self.assertEqual(alias.tolist(), [37] * 16)
        del alias
        gc.collect()
        self.assertFalse(pool._closed)
        pool.close()
        self._retired(pool)

    def test_unsealed_local_reservation_defers_close_until_abort(self) -> None:
        """An active metadata-only reservation retains control until context exit."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        with pool.reserve() as unit:
            pool.close()
            self._retained(pool)
            self.assertFalse(unit._sealed)
        pool.close()
        self._retired(pool)

    @skipUnless(os.name == "posix", "requires names that outlive their last mapping")
    def test_dormant_spawn_argument_observes_sticky_cancel(self) -> None:
        """A dormant spawn handle attaches late and observes sticky cancellation."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        result, start = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_late_endpoint, (pool, result, start))
        self.assertIsNone(pool._direct)
        pool.close()
        self._names_exist(pool)
        start.set()
        pid, error, cancelled = result.get(timeout=30)
        self.assertEqual(pid, process.pid)
        self.assertIn(error, ("BufferError", "RuntimeError"))
        self.assertTrue(cancelled)
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        pool.close()
        self._retired(pool)

    @parameterized.expand(
        [("pool", "writer"), ("pool", "reader"), ("ring", "writer"), ("ring", "reader")]
    )
    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_live_ordinary_endpoint_defers_retirement(
        self, kind: str, side: str
    ) -> None:
        """Either live endpoint PID retains mappings across repeated public close calls."""
        arena = self._arena(kind)
        result, finish = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_ordinary_endpoint, (arena, side, result, finish))
        pid, endpoint = result.get(timeout=30)
        self.assertEqual(pid, process.pid)
        self.assertIn(side.capitalize(), endpoint)
        for _ in range(3):
            arena.close()
            self.assertTrue(process.is_alive())
            self._retained(arena)
        finish.set()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        arena.close()
        self._retired(arena)
        self._names_exist(arena)

    @parameterized.expand([("pool",), ("ring",)])
    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_unknown_remote_pid_conservatively_retains(self, kind: str) -> None:
        """An uncertain POSIX PID retains the remote endpoint and mapping roots."""
        arena = self._arena(kind)
        result, finish = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_ordinary_endpoint, (arena, "writer", result, finish))
        self.assertEqual(result.get(timeout=30)[0], process.pid)
        finish.set()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        with patch.object(
            pool_module.os, "kill", side_effect=PermissionError("unknown PID")
        ) as probe:
            arena.close()
            self._retained(arena)
            probe.assert_called_once_with(process.pid, 0)
        arena.close()
        self._retired(arena)

    def test_non_posix_process_check_never_uses_kill_zero(self) -> None:
        """Non-POSIX PID checks retain ownership without calling os.kill."""
        # Model a live Windows process handle so this runs on any host.
        winapi = SimpleNamespace(
            SYNCHRONIZE=0x00100000,
            WAIT_OBJECT_0=0,
            OpenProcess=lambda access, inherit, pid: 1,
            WaitForSingleObject=lambda handle, timeout: 0x102,  # WAIT_TIMEOUT
            CloseHandle=lambda handle: None,
        )
        with (
            patch.object(pool_module.os, "name", "nt"),
            patch.dict(sys.modules, {"_winapi": winapi}),
            patch.object(pool_module.os, "kill") as probe,
        ):
            self.assertTrue(pool_module._process_exists(os.getpid()))
            self.assertFalse(pool_module._process_exists(0))
            probe.assert_not_called()

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_posix_process_exists_before_and_after_actual_reap(self) -> None:
        """POSIX PID absence requires an actual worker exit and join."""
        arena = self._arena("pool")
        result, finish = self._queue(), mp.get_context("spawn").Event()
        process = self._spawn(_ordinary_endpoint, (arena, "writer", result, finish))
        pid = result.get(timeout=30)[0]
        self.assertEqual(pid, process.pid)
        self.assertTrue(pool_module._process_exists(pid))
        finish.set()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        self.assertFalse(pool_module._process_exists(pid))

    @parameterized.expand([("pool",), ("ring",)])
    def test_dead_condition_is_bounded_and_roots_are_preserved(self, kind: str) -> None:
        """A hard-dead condition fails promptly and retains uncertain mappings."""
        arena: _Arena = (
            SharedMemorySegmentPool(65536, 2, acquire_timeout=0)
            if kind == "pool"
            else SharedMemoryRingBuffer(65536, acquire_timeout=0)
        )
        ready = mp.get_context("spawn").Event()
        process = self._spawn(_hold_condition, (arena, ready))
        self.assertTrue(ready.wait(30))
        process.join(30)
        self.assertEqual(process.exitcode, 17)
        module = pool_module if kind == "pool" else ring_module
        start = time.monotonic()
        with patch.object(module, "_CONTROL_LOCK_TIMEOUT", 0.05):
            with self.assertRaisesRegex(RuntimeError, "retained"):
                arena.close()
        self.assertLess(time.monotonic() - start, 2)
        self.assertFalse(arena._closed)
        self.assertIsNotNone(arena._retirement_error)
        roots = (
            pool_module._REGISTRATION_QUARANTINE
            if kind == "pool"
            else ring_module._RETIREMENT_ROOTS
        )
        reference = weakref.ref(arena)
        self.assertIn(arena, roots)
        self._names_exist(arena)
        # The dead condition cannot be repaired, so physical retirement never
        # completes: the creator keeps its names and its retained root.
        with self.assertRaises(BufferError):
            arena.unlink()
        self._names_exist(arena)
        self.addCleanup(self._remove_os_names, self._os_names(arena))
        del arena
        gc.collect()
        self.assertIsNotNone(reference())

    @parameterized.expand([("pool",), ("ring",)])
    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_two_spawn_writer_sessions_preserve_first_output(self, kind: str) -> None:
        """New writer PIDs publish distinct values without overwriting old aliases."""
        arena = self._arena(kind)
        result = self._queue()
        outputs = []
        pids = []
        for value in (b"first session", b"second session"):
            finish = mp.get_context("spawn").Event()
            process = self._spawn(_ordinary_publish, (arena, value, result, finish))
            pid, offset, nbytes, span = result.get(timeout=30)
            self.assertEqual(pid, process.pid)
            pids.append(pid)
            reader = arena.open_reader()
            view = reader.read_binary(offset, nbytes)
            reader.end_unit(span, [])
            outputs.append(view)
            self.assertEqual(bytes(view), value)
            finish.set()
            process.join(30)
            self.assertEqual(process.exitcode, 0)
        self.assertNotEqual(*pids)
        self.assertEqual(
            [bytes(view) for view in outputs], [b"first session", b"second session"]
        )
        if isinstance(arena, SharedMemorySegmentPool):
            self.assertEqual(arena._published, 2)
            self.assertEqual(arena._reclaimed, 0)
            self.assertIs(arena.open_reader(), reader)
            outputs.pop().release()
            self.assertEqual(arena._reclaimed, 0)
            outputs.pop().release()
            self.assertEqual(arena._reclaimed, 2)
        else:
            self.assertEqual((arena._head, arena._tail), (27, 27))
            for view in outputs:
                view.release()

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_rebound_pool_reader_starts_at_delivered_not_reclaimed(self) -> None:
        """A new reader PID resumes at completed delivery instead of replaying."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        writer = pool.open_writer()
        result = self._queue()
        for index, value in enumerate((b"reader one", b"reader two"), start=1):
            writer.begin_unit()
            offset, nbytes = writer.write_binary(value)
            span = writer.commit_unit()
            process = self._spawn(
                _ordinary_consume, (pool, offset, nbytes, span, result)
            )
            pid, restored, delivered = result.get(timeout=30)
            self.assertEqual(pid, process.pid)
            self.assertEqual(restored, value)
            self.assertEqual(delivered, index)
            process.join(30)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(pool._read_cursor, 2)
        self.assertEqual(pool._reclaimed, 2)

    def test_pool_reader_new_handle_uses_shared_delivered_cursor(self) -> None:
        """New handles use delivered progress while old aliases gate FIFO reclaim."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        writer, reader = pool.open_writer(), pool.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"old alias")
        span = writer.commit_unit()
        old = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual((pool._read_cursor, pool._reclaimed), (1, 0))
        attached = SharedMemorySegmentPool.__new__(SharedMemorySegmentPool)
        attached.__setstate__(pool.__getstate__())
        self.addCleanup(attached.close)
        next_reader = attached.open_reader()
        self.assertEqual(next_reader._next, 1)
        self.assertEqual(bytes(old), b"old alias")
        old.release()
        attached.close()

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_direct_owner_cannot_rebind_after_actual_reap(self) -> None:
        """A reaped direct producer PID never admits a replacement producer."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        result, finish = self._queue(), mp.get_context("spawn").Event()
        first = self._spawn(_metadata_worker, (pool, result, finish))
        pid, ticket, metadata = result.get(timeout=30)
        self.assertEqual(pid, first.pid)
        with pool.recover(ticket) as batch:
            self.assertEqual(batch.nbytes, 0)
            self.assertEqual(metadata["pid"], pid)
        finish.set()
        first.join(30)
        self.assertEqual(first.exitcode, 0)
        self.assertFalse(pool_module._process_exists(pid))
        next_process = self._spawn(_probe_endpoint, (pool, "direct", result))
        next_pid, error = result.get(timeout=30)
        self.assertEqual(next_pid, next_process.pid)
        self.assertNotEqual(pid, next_pid)
        self.assertEqual(error, "RuntimeError")
        next_process.join(30)
        self.assertEqual(next_process.exitcode, 0)
        self.assertEqual((pool._published, pool._reclaimed), (1, 1))

    @parameterized.expand(
        [("pool", "writer"), ("pool", "reader"), ("ring", "writer"), ("ring", "reader")]
    )
    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_live_endpoint_role_cannot_be_rebound(self, kind: str, side: str) -> None:
        """A second process cannot replace an endpoint whose original PID is still alive."""
        arena = self._arena(kind)
        result, finish = self._queue(), mp.get_context("spawn").Event()
        original = self._spawn(_ordinary_endpoint, (arena, side, result, finish))
        pid, _ = result.get(timeout=30)
        self.assertEqual(pid, original.pid)
        competing = self._spawn(_probe_endpoint, (arena, side, result))
        next_pid, error = result.get(timeout=30)
        self.assertEqual(next_pid, competing.pid)
        self.assertNotEqual(pid, next_pid)
        self.assertEqual(error, "RuntimeError")
        competing.join(30)
        self.assertEqual(competing.exitcode, 0)
        self.assertTrue(original.is_alive())
        finish.set()
        original.join(30)
        self.assertEqual(original.exitcode, 0)

    def test_ring_reset_does_not_zero_unread_cursors(self) -> None:
        """Resets preserve unread spans and monotonic absolute ring cursors."""
        ring = cast(SharedMemoryRingBuffer, self._arena("ring"))
        writer, reader = ring.open_writer(), ring.open_reader()
        writer.begin_unit()
        offset, size = writer.write_binary(b"unread")
        span = writer.commit_unit()
        writer.reset()
        reader.reset()
        self.assertEqual((ring._head, ring._tail), (6, 0))
        first = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        writer.begin_unit()
        offset, size = writer.write_binary(b"next")
        span = writer.commit_unit()
        second = reader.read_binary(offset, size)
        reader.end_unit(span, [])
        self.assertEqual((ring._head, ring._tail), (10, 10))
        self.assertEqual((bytes(first), bytes(second)), (b"unread", b"next"))
        ring.close()
        self._retired(ring)
        self.assertEqual(bytes(first), b"unread")
        first.release()
        second.release()

    def _assert_loaded_core(self, pool: SharedMemorySegmentPool) -> None:
        pool_receipt = _loaded_code_receipt(pool_module, "SharedMemorySegmentPool")
        ring_receipt = _loaded_code_receipt(ring_module, "SharedMemoryRingBuffer")
        for receipt in (pool_receipt, ring_receipt):
            self.assertTrue(
                all(method["matches_source"] for method in receipt["methods"].values()),
                receipt,
            )
        self.assertEqual(pool_module._CTRL.size, 42)
        self.assertTrue(pool._owns_names)
        buffer_module = sys.modules["spdl.io.lib._buffer"]
        native_file = getattr(buffer_module, "__file__", None)
        native_path = Path(native_file or sys.executable)
        print(
            "SPDL_LIFETIME_RUNTIME="
            + json.dumps(
                {
                    "python": sys.version,
                    "executable": sys.executable,
                    "torch": torch.__version__,
                    "pool": pool_receipt,
                    "ring": ring_receipt,
                    "native_buffer_file": native_file,
                    "native_buffer_origin": str(
                        getattr(
                            getattr(buffer_module, "__spec__", None), "origin", None
                        )
                    ),
                    "hashed_runtime_image": str(native_path),
                    "runtime_image_sha256": hashlib.sha256(
                        native_path.read_bytes()
                    ).hexdigest(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _region_value(self, arena: _Arena, sequence: int) -> dict[str, Any]:
        pipeline = (
            PipelineBuilder()
            .add_source([sequence])
            .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
            .pipe(_public_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(1)
            .build(num_threads=3, max_failures=0)
        )
        with pipeline.auto_stop():
            epoch = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(len(epoch), 1)
            value = epoch[0]
            self._reported_pids.add(value["pid"])
        self.assertFalse(pool_module._process_exists(value["pid"]))
        return value

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_three_region_generations_release_out_of_order_without_gc(self) -> None:
        """Three public pipelines preserve old aliases and reclaim released later units."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        automatic_gc = gc.isenabled()
        first = self._region_value(pool, 71)
        first_pid = first["pid"]
        alias = first["value"].detach()
        del first
        reader = pool.open_reader()
        second = self._region_value(pool, 83)
        second_pid = second["pid"]
        self.assertNotEqual(first_pid, second_pid)
        self.assertEqual(second["id"], 83)
        self.assertEqual(second["value"].tolist(), [83] * 1024)
        self.assertEqual(alias.tolist(), [71] * 1024)
        del second, reader
        self.assertEqual((pool._published, pool._reclaimed), (2, 0))
        del alias
        self.assertEqual((pool._published, pool._reclaimed), (2, 2))
        third = self._region_value(pool, 97)
        self.assertNotIn(third["pid"], (first_pid, second_pid))
        self.assertEqual(third["id"], 97)
        self.assertEqual(third["value"].tolist(), [97] * 1024)
        self.assertFalse(third["cuda"])
        del third
        self.assertEqual((pool._published, pool._reclaimed), (3, 3))
        self.assertEqual(gc.isenabled(), automatic_gc)
        pool.close()
        self._retired(pool)

    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_three_region_generations_native_release_precedes_real_reuse(self) -> None:
        """Real third-generation reuse occurs only after the old native release callback."""
        pool = cast(SharedMemorySegmentPool, self._arena("pool"))
        automatic_gc = gc.isenabled()
        releases: list[tuple[int, int]] = []
        original = pool_module._PoolReader.unit_released

        def released(reader: Any, unit: int) -> None:
            published = reader.pool._published
            original(reader, unit)
            releases.append((unit, published))

        with patch.object(pool_module._PoolReader, "unit_released", released):
            first = self._region_value(pool, 71)
            self._assert_loaded_core(pool)
            first_pid = first["pid"]
            alias = first["value"].detach()
            del first
            second = self._region_value(pool, 83)
            second_pid = second["pid"]
            self.assertEqual(second["value"].tolist(), [83] * 1024)
            self.assertEqual(alias.tolist(), [71] * 1024)
            del second
            self.assertEqual((pool._published, pool._reclaimed), (2, 0))
            self.assertNotIn(0, [unit for unit, _ in releases])
            del alias
            before_next_work = (pool._published, pool._reclaimed)
            third = self._region_value(pool, 97)
            self.assertNotIn(third["pid"], (first_pid, second_pid))
            self.assertEqual(third["id"], 97)
            self.assertEqual(third["value"].tolist(), [97] * 1024)
            self.assertFalse(third["cuda"])
            self.assertIn((0, 2), releases, before_next_work)
            del third
            self.assertEqual((pool._published, pool._reclaimed), (3, 3), releases)
        self.assertEqual(gc.isenabled(), automatic_gc)
        pool.close()
        self._retired(pool)

    @parameterized.expand([("pool",), ("ring",)])
    @skipUnless(os.name == "posix", "requires POSIX PID lifetime proof")
    def test_public_two_region_pipelines_keep_borrowed_arena(self, kind: str) -> None:
        """Two normally stopped public region pipelines preserve the first held output."""
        arena = self._arena(kind)
        values = []
        for sequence in (71, 83):
            pipeline = (
                PipelineBuilder()
                .add_source([sequence])
                .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
                .pipe(_public_payload)
                .to(MAIN_PROCESS, arena=arena)
                .add_sink(1)
                .build(num_threads=3, max_failures=0)
            )
            with pipeline.auto_stop():
                epoch = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(len(epoch), 1)
                values.append(epoch[0])
                self._reported_pids.add(epoch[0]["pid"])
            self.assertFalse(pool_module._process_exists(epoch[0]["pid"]))
            self.assertFalse(arena._closed)
            self.assertFalse(arena._shutdown)
            self._names_exist(arena)
        self.assertNotEqual(values[0]["pid"], values[1]["pid"])
        self.assertEqual([value["id"] for value in values], [71, 83])
        self.assertTrue(all(not value["cuda"] for value in values))
        self.assertEqual(values[0]["value"].tolist(), [71] * 1024)
        self.assertEqual(values[1]["value"].tolist(), [83] * 1024)
        if isinstance(arena, SharedMemorySegmentPool):
            self.assertEqual((arena._published, arena._reclaimed), (2, 0))
        del values, epoch, pipeline
        gc.collect()
        arena.close()
        self._retired(arena)
