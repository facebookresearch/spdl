# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V2 explicit object transport, borrowing, cancellation and completion regressions."""

from __future__ import annotations

import asyncio
import copy
import gc
import json
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import unittest
import weakref
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
from parameterized import parameterized
from spdl.pipeline import (
    _pool_cleanup,
    _subprocess_pipeline_pool as pools,
    AsyncQueue,
    build_pipeline,
    iterate_in_subprocess,
    PipelineBuilder,
    PipelineFailure,
    run_pipeline_in_subprocess,
    SharedMemoryRingBuffer,
    SharedMemorySegmentPool,
    StageInfo,
)
from spdl.pipeline._arena._region import (
    _ArenaEnvelope,
    _ArenaInputQueue,
    _ArenaOutputQueue,
    _ArenaSendingQueue,
    _RegionChannel,
)
from spdl.pipeline._bg_task import set_default_background_tasks
from spdl.pipeline._build import set_default_build_callback
from spdl.pipeline._components import (
    _AsyncQueueWithSyncMirror,
    _DONE,
    _ITEM,
    _POOL_SHUTDOWN,
    _SESSION_END,
    set_default_hook_class,
    set_default_queue_class,
    StatsQueue,
    TaskStatsHook,
    ThreadOutputQueue,
)
from spdl.pipeline._components._subprocess_pipe import (
    _collect,
    _collect_continuous,
    _ERROR,
    _pop_buffered_serialization_error,
    _RESULT,
)
from spdl.pipeline._pipeline import _PipelineImpl
from spdl.pipeline._subprocess_pipeline_pool import (
    _pipeline_worker_loop,
    _ProcessBackend,
    _run_sessions,
    _SerializationFailureState,
    _SubprocessPipelineHandle,
)
from spdl.pipeline.defs import (
    InterpreterPoolExecutorConfig,
    MAIN_PROCESS,
    Pipe,
    PipelineConfig,
    PlacementConfig,
    ProcessPoolExecutorConfig,
    SinkConfig,
    SourceConfig,
)
from spdl.tests.pipeline._defaults import _DefaultsTestCase


def _disable_network_hooks() -> None:
    set_default_build_callback(None)
    set_default_background_tasks([])
    set_default_hook_class(TaskStatsHook)
    set_default_queue_class(StatsQueue)


def _identity(item: Any) -> Any:
    return item


def _payload(index: int) -> dict[str, Any]:
    return {
        "id": index,
        "data": {
            "array": np.full(16384, index % 251, dtype=np.uint8),
            "tensor": torch.full((8192,), index, dtype=torch.int64),
            "bytes": bytes([index % 251]) * 16384,
        },
    }


def _drop_and_expand(item: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if item["id"] % 2 == 0:
        yield item
        yield {**item, "id": item["id"] + 1000}


def _fail(item: Any) -> Any:
    raise ValueError("region deliberate stage failure")


def _fail_after_two(item: dict[str, Any]) -> dict[str, Any]:
    if item["id"] >= 2:
        raise ValueError("region deliberate late failure")
    return item


def _broken_restore() -> object:
    raise ValueError("deliberate arena reconstruction failure")


class _BrokenRestore:
    def __reduce__(self) -> Any:
        return _broken_restore, ()


def _direct_ring_items() -> list[dict[str, Any]]:
    return [{"id": index, "blob": bytes([index]) * 16384} for index in range(32)]


class _RegionTestCase(_DefaultsTestCase):
    def setUp(self) -> None:
        super().setUp()
        _disable_network_hooks()
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)

    def objects(
        self, kind: str, workers: int = 1, capacity: int = 1 << 20, slots: int = 16
    ) -> tuple[Any, ...]:
        result = tuple(
            SharedMemorySegmentPool(capacity, slots, acquire_timeout=5)
            if kind == "pool"
            else SharedMemoryRingBuffer(capacity * slots, acquire_timeout=5)
            for _ in range(workers)
        )
        self.addCleanup(self.close_objects, result)
        return result

    def close_objects(self, objects: Sequence[Any]) -> None:
        for arena in objects:
            arena.close()
            arena.unlink()


class BorrowedArenaConfigTest(_RegionTestCase):
    def test_single_object_and_sequence_snapshot(self) -> None:
        """Public markers hold real objects and snapshot mutable worker sequences."""
        arenas = self.objects("ring", 2)
        self.assertIs(PlacementConfig(MAIN_PROCESS, arena=arenas[0]).arena, arenas[0])
        original = list(arenas)
        marker = PlacementConfig(MAIN_PROCESS, arena=original)
        original.clear()
        self.assertEqual(marker.arena, arenas)
        self.assertEqual(len(cast(tuple[Any, ...], marker.arena)), 2)

    @parameterized.expand(
        [
            (case,)
            for case in (
                "missing",
                "extra",
                "duplicate",
                "alias",
                "both-directions",
                "adjacent",
                "main",
                "interpreter",
                "invalid",
            )
        ]
    )
    def test_invalid_objects_rejected_before_spawn_without_owner_cleanup(
        self, case: str
    ) -> None:
        """Invalid cardinality/aliases/boundaries never spawn or close caller objects."""
        arenas = self.objects("ring", 2)
        pool = ProcessPoolExecutorConfig(max_workers=2, mp_context="spawn")
        value: Any = arenas
        if case == "missing":
            value = arenas[0]
        elif case == "extra":
            value = (*arenas, SharedMemoryRingBuffer(65536))
            self.addCleanup(self.close_objects, value[2:])
        elif case == "duplicate":
            value = (arenas[0], arenas[0])
        elif case == "alias":
            value = (arenas[0], copy.copy(arenas[0]))
            self.addCleanup(value[1].close)
        elif case == "invalid":
            value = (arenas[0], object())
        with (
            patch.object(_ProcessBackend, "spawn") as spawn,
            patch.object(arenas[0], "close") as close,
            patch.object(arenas[0], "unlink") as unlink,
        ):
            with self.assertRaises((ValueError, TypeError)):
                builder = PipelineBuilder().add_source([])
                if case == "main":
                    builder.to(MAIN_PROCESS, arena=value)
                elif case == "interpreter":
                    builder.to(InterpreterPoolExecutorConfig(), arena=value).pipe(
                        _identity
                    ).to(MAIN_PROCESS)
                elif case == "adjacent":
                    builder.to(pool).pipe(_identity).to(pool, arena=value).pipe(
                        _identity
                    ).to(MAIN_PROCESS)
                elif case == "both-directions":
                    builder.to(pool, arena=value).pipe(_identity).to(
                        MAIN_PROCESS, arena=value
                    )
                else:
                    builder.to(pool).pipe(_identity).to(MAIN_PROCESS, arena=value)
                build_pipeline(builder.add_sink().get_config(), num_threads=3)
            spawn.assert_not_called()
            close.assert_not_called()
            unlink.assert_not_called()

    def test_custom_executor_seven_argument_contract(self) -> None:
        """Custom process executors receive explicit objects without an API wrapper."""
        calls = []

        class Custom(ProcessPoolExecutorConfig):
            def _build(
                self,
                stages: Sequence[object],
                backend: Any,
                report_stats_interval: float,
                fail_fast: bool,
                continuous: bool,
                input_buffer_size: int,
                output_buffer_size: int,
            ) -> Any:
                calls.append(
                    (
                        backend._input_arena,
                        backend._output_arena,
                        input_buffer_size,
                        output_buffer_size,
                    )
                )
                return super()._build(
                    stages,
                    backend,
                    report_stats_interval,
                    fail_fast,
                    continuous,
                    input_buffer_size,
                    output_buffer_size,
                )

        incoming, outgoing = self.objects("ring"), self.objects("pool")
        pipeline = (
            PipelineBuilder()
            .add_source(range(7))
            .to(
                Custom(max_workers=1, mp_context="spawn"),
                arena=incoming[0],
                buffer_size=3,
            )
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=outgoing[0], buffer_size=2)
            .add_sink(2)
            .build(num_threads=3)
        )
        with pipeline.auto_stop():
            self.assertCountEqual(
                [item["id"] for item in pipeline.get_iterator(timeout=30)], range(7)
            )
        self.assertEqual(calls, [(incoming[0], outgoing[0], 3, 2)])

    def test_merged_configs_cannot_borrow_the_same_allocation(self) -> None:
        """Validation catches shared SPSC allocations across merged child configs."""
        from spdl.pipeline.defs import Merge

        arena = self.objects("ring")[0]

        def config() -> Any:
            return (
                PipelineBuilder()
                .add_source([])
                .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
                .pipe(_identity)
                .to(MAIN_PROCESS, arena=arena)
                .add_sink()
                .get_config()
            )

        merged = PipelineConfig(Merge([config(), config()]), [], SinkConfig(1))
        with patch.object(_ProcessBackend, "spawn") as spawn:
            with self.assertRaisesRegex(ValueError, "independent arena"):
                build_pipeline(merged, num_threads=3)
            spawn.assert_not_called()

    def test_spawn_rollback_keeps_caller_arena_open(self) -> None:
        """Failed construction cleans only pipeline resources, not caller arenas."""
        arenas = self.objects("ring", 2)
        builder = (
            PipelineBuilder()
            .add_source([])
            .to(ProcessPoolExecutorConfig(max_workers=2, mp_context="spawn"))
            .pipe(_identity)
            .to(MAIN_PROCESS, arena=arenas)
            .add_sink()
        )
        with (
            patch.object(_ProcessBackend, "spawn", side_effect=OSError("spawn")),
            patch.object(arenas[0], "close") as close,
            patch.object(arenas[0], "unlink") as unlink,
        ):
            with self.assertRaisesRegex(OSError, "spawn"):
                builder.build(num_threads=3)
            close.assert_not_called()
            unlink.assert_not_called()
        for arena in arenas:
            mapping = shared_memory.SharedMemory(name=arena.name)
            mapping.close()


class BorrowedArenaTransportTest(_RegionTestCase):
    @parameterized.expand(
        [
            (kind, direction)
            for kind in ("pool", "ring")
            for direction in ("input", "output", "both")
        ]
    )
    def test_buffered_nested_payload_roundtrip(self, kind: str, direction: str) -> None:
        """Explicit independent worker objects preserve every value and final transfer."""
        incoming = self.objects(kind, 3) if direction != "output" else None
        outgoing = self.objects(kind, 3) if direction != "input" else None
        pipeline = (
            PipelineBuilder()
            .add_source(range(23))
            .pipe(_payload)
            .to(
                ProcessPoolExecutorConfig(max_workers=3, mp_context="spawn"),
                arena=incoming,
                buffer_size=3,
            )
            .pipe(_identity)
            .to(MAIN_PROCESS, arena=outgoing, buffer_size=2)
            .add_sink(2)
            .build(num_threads=3)
        )
        ids = []
        with pipeline.auto_stop():
            for item in pipeline.get_iterator(timeout=30):
                index = item["id"]
                self.assertTrue(
                    np.array_equal(
                        item["data"]["array"],
                        np.full(16384, index % 251, dtype=np.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        item["data"]["tensor"],
                        torch.full((8192,), index, dtype=torch.int64),
                    )
                )
                self.assertEqual(item["data"]["bytes"], bytes([index % 251]) * 16384)
                ids.append(index)
                del item
        self.assertCountEqual(ids, range(23))

    @parameterized.expand([("pool",), ("ring",)])
    def test_continuous_epochs_and_aggregation(self, kind: str) -> None:
        """Epoch reuse and cardinality-changing stages retain independent borrowed storage."""
        incoming, outgoing = self.objects(kind, 2), self.objects(kind, 2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(23), continuous=True)
            .pipe(_payload)
            .to(
                ProcessPoolExecutorConfig(max_workers=2, mp_context="spawn"),
                arena=incoming,
                buffer_size=3,
            )
            .aggregate(4)
            .disaggregate()
            .pipe(_drop_and_expand)
            .to(MAIN_PROCESS, arena=outgoing, buffer_size=3)
            .add_sink(2)
            .build(num_threads=3)
        )
        with pipeline.auto_stop():
            for _ in range(3):
                ids = []
                for item in pipeline.get_iterator(timeout=30):
                    ids.append(item["id"])
                    del item
                self.assertCountEqual(
                    ids,
                    [
                        i + offset
                        for i in range(23)
                        if i % 2 == 0
                        for offset in (0, 1000)
                    ],
                )

    @parameterized.expand([("pool",), ("ring",)])
    def test_same_objects_reused_after_successful_stop(self, kind: str) -> None:
        """Successful stop leaves caller objects usable by a later pipeline."""
        arenas = self.objects(kind, 2)
        for start in (0, 20):
            pipeline = (
                PipelineBuilder()
                .add_source(range(start, start + 7))
                .to(ProcessPoolExecutorConfig(max_workers=2, mp_context="spawn"))
                .pipe(_payload)
                .to(MAIN_PROCESS, arena=arenas)
                .add_sink(2)
                .build(num_threads=3)
            )
            with pipeline.auto_stop():
                seen = []
                for item in pipeline.get_iterator(timeout=30):
                    seen.append(item["id"])
                    self.assertEqual(int(item["data"]["tensor"][0]), item["id"])
                    del item
                self.assertCountEqual(seen, range(start, start + 7))

    @parameterized.expand([("pool",), ("ring",)])
    def test_wrapper_and_explicit_adjacent_regions(self, kind: str) -> None:
        """MAIN markers make adjacent ingress/egress allocations explicit and independent."""
        out_first, in_next, out_next = (
            self.objects(kind),
            self.objects(kind),
            self.objects(kind),
        )
        target = ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn")
        cfg = (
            PipelineBuilder()
            .add_source(range(7))
            .to(target)
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=out_first[0])
            .to(target, arena=in_next[0])
            .pipe(_identity)
            .to(MAIN_PROCESS, arena=out_next[0])
            .add_sink(2)
            .get_config()
        )
        wrapped = run_pipeline_in_subprocess(cfg, num_threads=3, mp_context="spawn")
        try:
            for _ in range(2):
                self.assertCountEqual([item["id"] for item in wrapped], range(7))
        finally:
            del wrapped
            gc.collect()

    def test_storage_alias_survives_stop_and_caller_close(self) -> None:
        """Borrowed output aliases stay valid after pipeline and caller teardown."""
        arena = self.objects("pool", slots=16)[0]
        pipeline = (
            PipelineBuilder()
            .add_source(range(7))
            .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(2)
            .build(num_threads=3)
        )
        retained = None
        with pipeline.auto_stop():
            for item in pipeline.get_iterator(timeout=30):
                if item["id"] == 3:
                    retained = item["data"]["tensor"].detach()
                del item
        self.assertIsNotNone(retained)
        arena.close()
        # The live alias still owns the mapping, so names stay until it is released.
        with self.assertRaises(BufferError):
            arena.unlink()
        self.assertTrue(
            torch.equal(retained, torch.full((8192,), 3, dtype=torch.int64))
        )
        del retained
        gc.collect()
        arena.close()
        arena.unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_fail_fast_egress_error_keeps_caller_storage_retirable(
        self, kind: str
    ) -> None:
        """A worker error abandons unread egress without closing caller storage."""
        outgoing = self.objects(kind, slots=4)
        pipeline = (
            PipelineBuilder()
            .add_source(range(6))
            .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
            .pipe(_payload)
            .pipe(_fail_after_two)
            .to(MAIN_PROCESS, arena=outgoing[0])
            .add_sink(2)
            .build(num_threads=3, max_failures=0)
        )
        received = []
        with (
            patch.object(outgoing[0], "close") as close,
            patch.object(outgoing[0], "unlink") as unlink,
        ):
            with pipeline.auto_stop(), self.assertRaises(PipelineFailure) as caught:
                for item in pipeline.get_iterator(timeout=30):
                    self.assertTrue(
                        torch.equal(
                            item["data"]["tensor"],
                            torch.full((8192,), item["id"], dtype=torch.int64),
                        )
                    )
                    received.append(item["id"])
                    del item
            close.assert_not_called()
            unlink.assert_not_called()
        self.assertIn(
            "region deliberate late failure", str(caught.exception.exceptions)
        )
        self.assertEqual(received, sorted(received))
        self.assertLessEqual(len(received), 2)
        del pipeline
        gc.collect()
        # Caller retirement still succeeds after the pipeline abandoned unread spans.
        outgoing[0].close()
        outgoing[0].unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_worker_error_wakes_full_ingress_without_pipeline_close(
        self, kind: str
    ) -> None:
        """Early worker error wakes a blocked borrowed ingress and preserves its cause."""
        incoming = self.objects(kind, slots=1)
        pipeline = (
            PipelineBuilder()
            .add_source(range(5))
            .pipe(_payload)
            .to(
                ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"),
                arena=incoming[0],
            )
            .pipe(_fail)
            .to(MAIN_PROCESS)
            .add_sink(2)
            .build(num_threads=3, max_failures=0)
        )
        with (
            patch.object(incoming[0], "close") as close,
            patch.object(incoming[0], "unlink") as unlink,
        ):
            with pipeline.auto_stop(), self.assertRaises(PipelineFailure):
                list(pipeline.get_iterator(timeout=30))
            close.assert_not_called()
            unlink.assert_not_called()


class _FakeWorker:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures

    def stop(self) -> None:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("failed reap")


def _registered_fake(failures: int = 0) -> Any:
    def spawn(target: Any, args: Any, register: Any) -> _FakeWorker:
        worker = _FakeWorker(failures)
        register(worker)
        return worker

    return spawn


def _retained(pool: object) -> bool:
    return (os.getpid(), id(pool)) in _pool_cleanup._PENDING


def _retire(pool: Any, attempts: int = 50) -> None:
    """Retry until bounded queue-feeder joins finish; the last failure still raises."""
    for _ in range(attempts - 1):
        try:
            pool.shutdown()
            return
        except ExceptionGroup:
            time.sleep(0.1)
    pool.shutdown()


class BorrowedArenaLifetimeTest(_RegionTestCase):
    def manager(self, arena: Any, failures: int = 0) -> Any:
        cfg = PipelineConfig(SourceConfig([]), [Pipe(_identity)], SinkConfig(1))
        with patch.object(
            _ProcessBackend, "spawn", side_effect=_registered_fake(failures)
        ):
            return pools._SubprocessPipelinePool(
                _ProcessBackend(mp.get_context("spawn")),
                1,
                cfg,
                {"num_threads": 2},
                None,
                (),
                output_arena=arena,
            )

    def test_failed_reap_and_owner_drop_keep_complete_borrowed_roots(self) -> None:
        """Clearing pipeline owners cannot lose borrowed context on a failed reap."""
        arena = self.objects("ring")[0]
        manager = self.manager(arena, 2)
        manager_ref = weakref.ref(manager)
        worker_ref = weakref.ref(manager._workers[0])
        queue_ref = weakref.ref(manager._in_qs[0])
        holder = SimpleNamespace(_pools=[manager], _pool_shutdown_lock=threading.Lock())
        with (
            patch.object(pools._LG, "warning"),
            patch.object(arena, "close") as close,
            patch.object(arena, "unlink") as unlink,
        ):
            _PipelineImpl._shutdown_pools(cast(Any, holder))
            del holder, manager
            gc.collect()
            self.assertIsNotNone(manager_ref())
            self.assertIsNotNone(worker_ref())
            self.assertIsNotNone(queue_ref())
            manager = manager_ref()
            assert manager is not None
            self.assertTrue(_retained(manager))
            self.assertEqual(manager._borrowed_arenas, (arena,))
            with self.assertRaises(ExceptionGroup):
                manager.shutdown()
            self.assertTrue(_retained(manager))
            _retire(manager)
            self.assertFalse(_retained(manager))
            self.assertEqual(manager._borrowed_arenas, ())
            close.assert_not_called()
            unlink.assert_not_called()
        del manager
        gc.collect()
        self.assertIsNone(manager_ref())

    def test_queue_cleanup_failure_retains_roots_and_never_closes_arena(self) -> None:
        """Queue cleanup retries retain caller arenas without taking ownership."""
        arena = self.objects("ring")[0]
        manager = self.manager(arena)
        with (
            patch.object(manager._backend, "close_queue", side_effect=OSError("queue")),
            self.assertRaises(ExceptionGroup),
        ):
            manager.shutdown()
        self.assertTrue(_retained(manager))
        self.assertEqual(manager._borrowed_arenas, (arena,))
        with (
            patch.object(arena, "close") as close,
            patch.object(arena, "unlink") as unlink,
        ):
            _retire(manager)
            self.assertFalse(_retained(manager))
            self.assertEqual(manager._borrowed_arenas, ())
            close.assert_not_called()
            unlink.assert_not_called()

    def test_constructor_cleanup_failure_keeps_primary_and_borrowed_root(
        self,
    ) -> None:
        """Failed construction cleanup preserves the primary error and retry context."""
        arena, other = self.objects("ring", 2)
        before = set(_pool_cleanup._PENDING)
        primary = OSError("primary spawn failure")
        started: list[_FakeWorker] = []

        def spawn(target: Any, args: Any, register: Any) -> _FakeWorker:
            if started:
                raise primary
            worker = _FakeWorker(1)
            register(worker)
            started.append(worker)
            return worker

        cfg = PipelineConfig(SourceConfig([]), [Pipe(_identity)], SinkConfig(1))
        owner: pools._SubprocessPipelinePool | None = None
        try:
            with (
                patch.object(_ProcessBackend, "spawn", side_effect=spawn),
                patch.object(pools._LG, "exception"),
            ):
                with self.assertRaises(OSError) as caught:
                    pools._SubprocessPipelinePool(
                        _ProcessBackend(mp.get_context("spawn")),
                        2,
                        cfg,
                        {"num_threads": 2},
                        None,
                        (),
                        output_arena=(arena, other),
                    )
            self.assertIs(caught.exception, primary)
            primary.__traceback__ = None
            gc.collect()
            added = [
                pool
                for key, pool in _pool_cleanup._PENDING.items()
                if key not in before
            ]
            self.assertEqual(len(added), 1)
            owner = cast(pools._SubprocessPipelinePool, added[0])
            self.assertEqual(owner._borrowed_arenas, (arena, other))
            with (
                patch.object(arena, "close") as close,
                patch.object(arena, "unlink") as unlink,
            ):
                _retire(owner)
                close.assert_not_called()
                unlink.assert_not_called()
            self.assertFalse(_retained(owner))
            self.assertEqual(owner._borrowed_arenas, ())
        finally:
            primary.__traceback__ = None
            if owner is not None and _retained(owner):
                owner.shutdown()


class RegionArenaProtocolTest(_RegionTestCase):
    def test_direct_ring_abandon_reiterate_accounts_old_spans(self) -> None:
        """Discarded direct-iterator spans cannot advance a newly reset ring generation."""
        ring = SharedMemoryRingBuffer(
            65536, acquire_timeout=30, _mp_context=mp.get_context("spawn")
        )
        src = iterate_in_subprocess(
            _direct_ring_items,
            arena=ring,
            buffer_size=2,
            mp_context="spawn",
            timeout=30,
        )
        try:
            iterator = iter(src)
            first = next(iterator)
            self.assertEqual(first["id"], 0)
            deadline = time.monotonic() + 10
            while ring._head == ring._tail and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertGreater(ring._head, ring._tail)
            del first, iterator
            gc.collect()
            for _ in range(3):
                got = []
                for item in src:
                    self.assertEqual(item["blob"], bytes([item["id"]]) * 16384)
                    got.append(item["id"])
                    del item
                self.assertEqual(got, list(range(32)))
        finally:
            del src
            gc.collect()

    def test_corrupt_input_balances_worker_done_once(self) -> None:
        """A terminal input reader still emits exactly one worker completion message."""
        arena = SharedMemoryRingBuffer(32768)
        raw: queue.Queue[Any] = queue.Queue()
        out: queue.Queue[Any] = queue.Queue()
        writer = _RegionChannel(0, arena)
        raw.put((0, writer.encode([_BrokenRestore()])))
        raw.put((0, writer.encode([bytes(16384)])))
        incoming = _ArenaInputQueue(raw, 0, _RegionChannel(0, arena))
        cfg = PipelineConfig(SourceConfig([]), [Pipe(_identity)], SinkConfig(2))
        try:
            with self.assertLogs(pools._LG, "WARNING") as logs:
                _run_sessions(
                    incoming, out, cfg, {"num_threads": 2, "max_failures": 0}, 1
                )
            messages = [out.get_nowait() for _ in range(out.qsize())]
            self.assertEqual([kind for kind, _ in messages], [_ERROR, _DONE])
            self.assertIn(
                "deliberate arena reconstruction failure", str(messages[0][1])
            )
            self.assertIn("corrupt region input channel", "\n".join(logs.output))
        finally:
            arena.close()
            arena.unlink()

    @parameterized.expand([("ok", None), ("stage", "stage"), ("init", "init")])
    def test_worker_loop_emits_one_done_per_session(
        self, _: str, failure: str | None
    ) -> None:
        """Each session ends with exactly one completion, including failed ones."""
        raw: queue.Queue[Any] = queue.Queue()
        out: queue.Queue[Any] = queue.Queue()
        for items in ([1, 2], [3]):
            raw.put((_ITEM, items))
            raw.put((_SESSION_END, None))
        raw.put((_POOL_SHUTDOWN, None))
        stage = _fail if failure == "stage" else _identity
        cfg = PipelineConfig(SourceConfig([]), [Pipe(stage)], SinkConfig(2))
        _pipeline_worker_loop(
            raw,
            out,
            0,
            cfg,
            {"num_threads": 2, "max_failures": 0},
            False,
            _broken_restore if failure == "init" else None,
            (),
            None,
            1,
        )
        messages = [out.get_nowait() for _ in range(out.qsize())]
        kinds = [kind for kind, _ in messages if kind != _RESULT]
        if failure == "init":
            self.assertEqual(kinds, [_ERROR, _DONE])
            return
        # Two input sessions plus the empty session ended by pool shutdown.
        if failure == "stage":
            self.assertEqual(kinds, [_ERROR, _DONE, _ERROR, _DONE, _DONE])
        else:
            self.assertEqual(kinds, [_DONE, _DONE, _DONE])
            values = [v for kind, chunk in messages if kind == _RESULT for v in chunk]
            self.assertEqual(values, [1, 2, 3])

    def test_error_relay_scan_survives_failed_discard(self) -> None:
        """A failed discard while awaiting a detailed error still returns that error."""
        detailed = RuntimeError("detailed worker error")

        class FailedChannelQueue:
            def __init__(self) -> None:
                self.messages = [(_RESULT, object()), (_ERROR, detailed)]

            def get(self, timeout: float) -> Any:
                if not self.messages:
                    raise queue.Empty
                return self.messages.pop(0)

            def decode_result(self, payload: Any, discard: bool) -> None:
                raise RuntimeError("Arena region channel is no longer usable")

        with self.assertLogs("spdl.pipeline._components._subprocess_pipe", "WARNING"):
            self.assertIs(
                _pop_buffered_serialization_error(FailedChannelQueue(), 5), detailed
            )

    def test_shutdown_signals_every_arena_despite_failure(self) -> None:
        """Optional channel interruptions never call a borrowed global shutdown."""

        class ChannelArena:
            def __init__(self, fails: bool = False) -> None:
                self.fails = fails
                self.calls = 0
                self.global_calls = 0

            def _interrupt_region(self) -> None:
                self.calls += 1
                if self.fails:
                    raise OSError("deliberate channel interruption failure")

            def shutdown_arena(self) -> None:
                self.global_calls += 1

        bad, good = ChannelArena(True), ChannelArena()
        handle = _SubprocessPipelineHandle([], None, 1, False, None, None)
        handle.input_arenas = cast(Any, [bad, good])
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ):
            handle.stop_arenas()
        self.assertEqual(bad.calls, 1)
        self.assertEqual(good.calls, 1)
        self.assertEqual(bad.global_calls, 0)
        self.assertEqual(good.global_calls, 0)

    @parameterized.expand([("session",), ("continuous",)])
    def test_serialization_scan_preserves_envelope_order(self, mode: str) -> None:
        """Failure scanning accounts for the already-dequeued unit before later units."""
        arena = SharedMemoryRingBuffer(65536)
        raw: queue.Queue[tuple[int, Any]] = queue.Queue()
        writer = _RegionChannel(0, arena)
        raw.put((_RESULT, writer.encode([bytes(16384)])))
        raw.put((_RESULT, writer.encode([bytes(16384)])))
        original = RuntimeError("deliberate detailed serialization error")
        raw.put((_ERROR, original))
        receiver = _ArenaOutputQueue(raw, [_RegionChannel(0, arena)])
        state = _SerializationFailureState(
            threading.Event(), threading.Event(), threading.Event()
        )
        state.failed.set()
        state.relay_failed.set()

        async def collect(executor: ThreadPoolExecutor) -> None:
            events = asyncio.Event(), asyncio.Event()
            if mode == "session":
                await _collect(
                    receiver,
                    1,
                    AsyncQueue(
                        StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
                    ),
                    executor,
                    *events,
                    (("output", state),),
                )
            else:
                await _collect_continuous(
                    receiver,
                    1,
                    AsyncQueue(
                        StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
                    ),
                    executor,
                    *events,
                    (("output", state),),
                )

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                with self.assertRaisesRegex(
                    RuntimeError, "deliberate detailed serialization error"
                ):
                    asyncio.run(collect(executor))
            self.assertEqual(arena._tail, arena._head)
        finally:
            arena.close()
            arena.unlink()

    def test_publication_failure_is_terminal(self) -> None:
        """A descriptor dropped after commit cannot silently retry on the same channel."""
        arena = SharedMemoryRingBuffer(32768)

        class BrokenQueue:
            def put(self, *args: Any, **kwargs: Any) -> None:
                raise OSError("deliberate pipe failure")

        sender = _ArenaSendingQueue(BrokenQueue(), _RESULT, _RegionChannel(0, arena))
        try:
            with self.assertRaisesRegex(OSError, "deliberate pipe failure"):
                sender.put((_RESULT, [bytes(16384)]))
            self.assertEqual(arena._head, 16384)
            self.assertEqual(arena._tail, 0)
            with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                sender.put((_RESULT, [bytes(16384)]))
        finally:
            arena.close()
            arena.unlink()

    @parameterized.expand([("pool",), ("ring",)])
    def test_discard_releases_every_backend(self, kind: str) -> None:
        """Unread committed transfers advance pool and ring accounting exactly once."""
        arena = (
            SharedMemorySegmentPool(32768, 1)
            if kind == "pool"
            else SharedMemoryRingBuffer(32768)
        )
        try:
            writer, reader = _RegionChannel(0, arena), _RegionChannel(0, arena)
            for i in range(5):
                envelope = writer.encode([bytes([i]) * 16384])
                reader.decode(envelope, discard=True)
            if isinstance(arena, SharedMemorySegmentPool):
                self.assertEqual(arena._published, arena._reclaimed)
            else:
                self.assertEqual(arena._head, arena._tail)
        finally:
            arena.close()
            arena.unlink()

    def test_sequence_violation_is_terminal(self) -> None:
        """Duplicate, gapped and unknown-worker envelopes cannot silently reclaim a unit."""
        arena = SharedMemoryRingBuffer(32768)
        try:
            writer = _RegionChannel(0, arena)
            env = writer.encode([bytes(16384)])
            for worker, sequence in ((0, 1), (1, 0), (0, -1)):
                reader = _RegionChannel(0, arena)
                with self.assertRaisesRegex(RuntimeError, "sequence mismatch"):
                    reader.decode(_ArenaEnvelope(worker, sequence, env.blob))
                self.assertEqual(arena._tail, 0)
                with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                    reader.decode(env)
        finally:
            arena.close()
            arena.unlink()

    def test_publication_retries_do_not_repickle(self) -> None:
        """A full descriptor queue retries the committed envelope without another offload."""
        arena = SharedMemoryRingBuffer(32768)
        q: queue.Queue[Any] = queue.Queue(1)
        q.put((1, None))
        sender = _ArenaSendingQueue(q, 0, _RegionChannel(0, arena))
        msg = (0, [bytes(16384)])
        try:
            for _ in range(3):
                with self.assertRaises(queue.Full):
                    sender.put_nowait(msg)
            self.assertEqual(arena._head, 16384)
            q.get_nowait()
            sender.put_nowait(msg)
            self.assertEqual(arena._head, 16384)
            envelope = q.get_nowait()[1]
            self.assertEqual(envelope.sequence, 0)
            _RegionChannel(0, arena).decode(envelope, True)
        finally:
            arena.close()
            arena.unlink()

    def test_unknown_worker_does_not_restore(self) -> None:
        """Demultiplexing rejects an unknown worker before accessing backing storage."""
        receiver = _ArenaOutputQueue(queue.Queue(), [])
        with self.assertRaisesRegex(RuntimeError, "unknown worker"):
            receiver.decode_result(_ArenaEnvelope(0, 0, b""))

    def test_no_bulk_bytes_in_descriptor(self) -> None:
        """The queue envelope contains markers, not a large binary leaf."""
        arena = SharedMemoryRingBuffer(32768)
        try:
            envelope = _RegionChannel(0, arena).encode([bytes(16384)])
            self.assertLess(len(pickle.dumps(envelope)), 1024)
            self.assertEqual(arena._head, 16384)
        finally:
            arena.close()
            arena.unlink()


_MIRROR_LAST_CHUNK_ENTERED = threading.Event()
_MIRROR_LAST_CHUNK_RELEASE = threading.Event()


def _mirror_last_chunk_value(item: int) -> int:
    _MIRROR_LAST_CHUNK_ENTERED.set()
    if not _MIRROR_LAST_CHUNK_RELEASE.wait(30):
        raise TimeoutError("Test did not release the final chunk producer")
    return item


def _mirror_last_chunk_failure(_: int) -> Iterator[int]:
    _MIRROR_LAST_CHUNK_ENTERED.set()
    if not _MIRROR_LAST_CHUNK_RELEASE.wait(30):
        raise TimeoutError("Test did not release the failing final chunk producer")
    yield from range(15, 18)
    raise ValueError("deliberate final producer failure")


class RegionMirrorCompletionTest(_RegionTestCase):
    @parameterized.expand(
        [
            (f"{sink}_{outcome}", sink, failure)
            for sink in ("mirrored", "thread")
            for outcome, failure in (("complete", False), ("failure", True))
        ]
    )
    def test_final_chunk_publication_racing_empty_poll_is_not_eof(
        self, _: str, sink: str, failure: bool
    ) -> None:
        """Worker result draining preserves final values published after an empty poll."""
        _MIRROR_LAST_CHUNK_ENTERED.clear()
        _MIRROR_LAST_CHUNK_RELEASE.clear()
        pipeline = (
            PipelineBuilder()
            .add_source([0] if failure else range(15, 18))
            .pipe(_mirror_last_chunk_failure if failure else _mirror_last_chunk_value)
            .add_sink(4)
            .build(
                num_threads=2,
                max_failures=0,
                use_thread_output_queue=sink == "thread",
            )
        )
        pipeline.start()
        impl = pipeline._impl
        sink_queue = cast(Any, impl._output_queue)
        if sink == "mirrored":
            self.assertIsInstance(sink_queue, _AsyncQueueWithSyncMirror)
            raw = sink_queue._sync_queue
        else:
            self.assertIsInstance(sink_queue, ThreadOutputQueue)
            raw = sink_queue._queue
        original = raw.get_nowait
        raced = False

        def poll() -> Any:
            nonlocal raced
            try:
                return original()
            except queue.Empty:
                if not raced:
                    raced = True
                    _MIRROR_LAST_CHUNK_RELEASE.set()
                    if not impl._event_loop._task_completed.wait(10):
                        raise TimeoutError("Final producer did not publish completion")
                raise

        try:
            self.assertTrue(_MIRROR_LAST_CHUNK_ENTERED.wait(10))
            self.assertEqual(raw.qsize(), 0)
            with self.assertRaises(queue.Empty):
                pipeline._get_item_nowait()
            with patch.object(raw, "get_nowait", poll):
                try:
                    first = pipeline._get_item_nowait()
                except queue.Empty:
                    # The thread sink rechecks emptiness after completion and reports
                    # "not yet" rather than EOF; the mirrored sink returns the value.
                    self.assertEqual(sink, "thread")
                    first = pipeline.get_item()
            self.assertTrue(raced)
            self.assertEqual(first, 15)
            self.assertEqual(pipeline.get_item(), 16)
            self.assertEqual(pipeline.get_item(), 17)
            if failure:
                with self.assertRaises(PipelineFailure) as error:
                    pipeline.get_item()
                cause = error.exception.exceptions[0].__cause__
                self.assertIsInstance(cause, ValueError)
                self.assertEqual(str(cause), "deliberate final producer failure")
            else:
                with self.assertRaises(EOFError):
                    pipeline.get_item()
        finally:
            _MIRROR_LAST_CHUNK_RELEASE.set()
            pipeline.stop()


_BORROWED_ENTERED: Any = None
_BORROWED_RELEASE: Any = None


def _blocked_borrowed_initializer(entered: Any, release: Any) -> None:
    entered.set()
    if not release.wait(30):
        raise TimeoutError("Borrowed test initializer was not released")


class _StructuralRing:
    """A custom duck-typed transport with only public endpoint and cleanup methods."""

    def __init__(self, capacity: int = 65536) -> None:
        self.ring = SharedMemoryRingBuffer(capacity, acquire_timeout=30)

    def open_writer(self) -> Any:
        return self.ring.open_writer()

    def open_reader(self) -> Any:
        return self.ring.open_reader()

    def shutdown_arena(self) -> None:
        self.ring.shutdown_arena()

    def close(self) -> None:
        self.ring.close()

    def unlink(self) -> None:
        self.ring.unlink()


def _blocked_pipe_initializer(entered: Any, receiver: Any) -> None:
    entered.set()
    if not receiver.poll(30):
        raise TimeoutError("Caller-close test pipe was not released")
    receiver.recv_bytes()


class BorrowedCustomAndProcessTest(_RegionTestCase):
    def test_structural_backend_needs_no_protocol_or_factory_import(self) -> None:
        """An ordinary custom object supplies endpoint methods without public type imports."""
        arena = _StructuralRing(1 << 20)
        self.addCleanup(arena.unlink)
        self.addCleanup(arena.close)
        pipeline = (
            PipelineBuilder()
            .add_source(range(7))
            .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(2)
            .build(num_threads=3)
        )
        with pipeline.auto_stop():
            self.assertCountEqual(
                [item["id"] for item in pipeline.get_iterator(timeout=30)], range(7)
            )
        # Successful borrowed stop did not call the wrapper's public close.
        pipeline = (
            PipelineBuilder()
            .add_source(range(20, 27))
            .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(2)
            .build(num_threads=3)
        )
        with pipeline.auto_stop():
            self.assertCountEqual(
                [item["id"] for item in pipeline.get_iterator(timeout=30)],
                range(20, 27),
            )

    def test_real_failed_reap_retains_borrowed_owner_until_retry(self) -> None:
        """A real blocked worker keeps its complete borrowing context after public stop."""
        ctx = mp.get_context("spawn")
        entered, release = ctx.Event(), ctx.Event()
        arena = self.objects("ring")[0]
        pipeline = None
        owner_ref = None
        try:
            pipeline = (
                PipelineBuilder()
                .add_source([1])
                .to(
                    ProcessPoolExecutorConfig(
                        max_workers=1,
                        mp_context="spawn",
                        initializer=_blocked_borrowed_initializer,
                        initargs=(entered, release),
                    )
                )
                .pipe(_identity)
                .to(MAIN_PROCESS, arena=arena)
                .add_sink(1)
                .build(num_threads=3)
            )
            owner = pipeline._impl._pools[0]
            owner_ref = weakref.ref(owner)
            worker_ref = weakref.ref(owner._workers[0])
            queue_ref = weakref.ref(owner._in_qs[0])
            pid = owner._workers[0]._proc.pid
            pipeline.start()
            self.assertTrue(entered.wait(20))
            failure = RuntimeError("failed real reap")
            with (
                patch.object(pools._ProcessWorker, "stop", side_effect=failure),
                patch.object(pools._LG, "warning"),
                patch.object(arena, "close") as close,
                patch.object(arena, "unlink") as unlink,
            ):
                pipeline.stop()
                close.assert_not_called()
                unlink.assert_not_called()
            failure.__traceback__ = None
            self.assertEqual(pipeline._impl._pools, [owner])
            self.assertTrue(_retained(owner))
            # Dropping the pipeline retries its pending pool; keep that retry failing
            # so only the cleanup registry roots the borrowing owner afterwards.
            with (
                patch.object(pools._ProcessWorker, "stop", side_effect=failure),
                patch.object(pools._LG, "warning"),
            ):
                del owner, pipeline
                pipeline = None
                gc.collect()
            failure.__traceback__ = None
            self.assertIsNotNone(owner_ref())
            self.assertIsNotNone(worker_ref())
            self.assertIsNotNone(queue_ref())
            owner = owner_ref()
            assert owner is not None
            self.assertEqual(owner._borrowed_arenas, (arena,))
            os.kill(pid, 0)
            release.set()
            _retire(owner)
            self.assertFalse(_retained(owner))
            self.assertEqual(owner._borrowed_arenas, ())
            self.assertEqual(owner._workers, [])
            # Pipeline teardown did not close caller storage, which remains reopenable.
            mapping = shared_memory.SharedMemory(name=arena.name)
            mapping.close()
            del owner
            gc.collect()
            self.assertIsNone(owner_ref())
        finally:
            release.set()
            if pipeline is not None:
                pipeline.stop()
            if owner_ref is not None:
                owner = owner_ref()
                if owner is not None:
                    owner.shutdown()

    def test_custom_public_close_wakes_cancelled_parent_writer(self) -> None:
        """Caller close wakes an actual non-ACK storage wait before pipeline join."""
        arena = _StructuralRing(131072)
        self.addCleanup(arena.unlink)
        self.addCleanup(arena.close)
        ctx = mp.get_context("spawn")
        entered = ctx.Event()
        receiver, sender = ctx.Pipe(duplex=False)
        blocked = threading.Event()
        original_wait = arena.ring._wait_space

        def wait(deadline: Any, control_deadline: Any) -> Any:
            # Actual _wait_space is called only after a false capacity predicate.
            blocked.set()
            return original_wait(deadline, control_deadline)

        pipeline = None
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                try:
                    sender.send_bytes(b"release")
                except OSError:
                    pass

        try:
            pipeline = (
                PipelineBuilder()
                .add_source(range(40))
                .pipe(_payload)
                .to(
                    ProcessPoolExecutorConfig(
                        max_workers=1,
                        mp_context="spawn",
                        initializer=_blocked_pipe_initializer,
                        initargs=(entered, receiver),
                    ),
                    arena=arena,
                )
                .pipe(_identity)
                .to(MAIN_PROCESS)
                .add_sink(1)
                .build(num_threads=3)
            )
            with patch.object(arena.ring, "_wait_space", wait):
                pipeline.start()
                self.assertTrue(entered.wait(20))
                self.assertTrue(blocked.wait(20))
                arena.close()
                release()
                try:
                    pipeline.stop(timeout=10)
                except PipelineFailure as error:
                    self.assertIn("ring is closing", str(error.exceptions))
            self.assertTrue(blocked.is_set())
            self.assertFalse(pipeline._impl._event_loop.needs_join())
            self.assertEqual(pipeline._impl._pools, [])
        finally:
            arena.close()
            release()
            if pipeline is not None:
                try:
                    pipeline.stop(timeout=10)
                except PipelineFailure:
                    pass
            receiver.close()
            sender.close()


class _LegacyGlobalShutdownArena(_StructuralRing):
    def __init__(self) -> None:
        super().__init__(1 << 20)
        self.stopped = mp.get_context("spawn").Value("i", 0)

    def open_writer(self) -> Any:
        if self.stopped.value:
            raise RuntimeError("Caller arena was globally stopped")
        return super().open_writer()

    def open_reader(self) -> Any:
        if self.stopped.value:
            raise RuntimeError("Caller arena was globally stopped")
        return super().open_reader()

    def shutdown_arena(self) -> None:
        with self.stopped.get_lock():
            self.stopped.value += 1


class BorrowedContinuousCancellationTest(_RegionTestCase):
    def test_full_epoch_stop_does_not_globally_stop_caller_arena(self) -> None:
        """Continuous stop never invokes global shutdown or reuses unread spans unsafely."""
        arena = _LegacyGlobalShutdownArena()
        self.addCleanup(arena.unlink)
        self.addCleanup(arena.close)
        target = ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn")
        pipeline = (
            PipelineBuilder()
            .add_source(range(7), continuous=True)
            .to(target)
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(2)
            .build(num_threads=3)
        )
        with pipeline.auto_stop(timeout=10):
            self.assertCountEqual(
                [item["id"] for item in pipeline.get_iterator(timeout=20)], range(7)
            )
        self.assertEqual(arena.stopped.value, 0)
        self.assertFalse(arena.ring._shutdown)
        cursors = (arena.ring._head, arena.ring._tail)
        pipeline = (
            PipelineBuilder()
            .add_source(range(20, 27))
            .to(target)
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(2)
            .build(num_threads=3)
        )
        if cursors[0] != cursors[1]:
            # Cancelled prefetch is unread, not a completed reusable channel.
            # Refusal preserves that occupied span; no counter/reset salvage.
            with (
                pipeline.auto_stop(timeout=10),
                self.assertRaises(PipelineFailure) as error,
            ):
                list(pipeline.get_iterator(timeout=20))
            self.assertIn("unread span", str(error.exception.exceptions))
            self.assertEqual((arena.ring._head, arena.ring._tail), cursors)
        else:
            with pipeline.auto_stop(timeout=10):
                self.assertCountEqual(
                    [item["id"] for item in pipeline.get_iterator(timeout=20)],
                    range(20, 27),
                )
        self.assertEqual(arena.stopped.value, 0)
        self.assertFalse(arena.ring._shutdown)
        receipt = {
            "variant": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "gil_enabled": getattr(os.sys, "_is_gil_enabled", lambda: True)(),
            "scenario": "continuous-full-epoch-stop",
            "completed_epoch_ids": list(range(7)),
            "before_fresh_writer_head_tail": list(cursors),
            "branch": "unread-refusal"
            if cursors[0] != cursors[1]
            else "completed-reuse",
            "after_fresh_writer_head_tail": [arena.ring._head, arena.ring._tail],
            "legacy_global_shutdown_calls": arena.stopped.value,
            "creator_local_closed": arena.ring._closed,
            "creator_shared_shutdown": arena.ring._shutdown,
        }
        directory = Path("/tmp/spdl-B-V2/nonACK-validation/continuous-branch-receipts")
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "gil" if receipt["gil_enabled"] else "free-threaded"
        (directory / f"{receipt['variant']}-{suffix}.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        )
        print(
            "CONTINUOUS_CANCEL_BRANCH " + json.dumps(receipt, sort_keys=True),
            flush=True,
        )


class _TupleArena(tuple):
    """Tuple metadata remains part of one structural transport object."""

    def __init__(self, contents: Any) -> None:
        self.close_calls = 0
        self.unlink_calls = 0

    def open_writer(self) -> Any:
        return self[1].open_writer()

    def open_reader(self) -> Any:
        return self[1].open_reader()

    def shutdown_arena(self) -> None:
        self[1].shutdown_arena()

    def close(self) -> None:
        self.close_calls += 1
        self[1].close()

    def unlink(self) -> None:
        self.unlink_calls += 1
        self[1].unlink()


class BorrowedTupleNormalizationTest(_RegionTestCase):
    def test_structural_tuple_arena_remains_one_borrowed_object(self) -> None:
        """A tuple-subclass backend is transported as one object, not its metadata."""
        ring = SharedMemoryRingBuffer(1 << 20, acquire_timeout=5)
        arena = _TupleArena(("metadata", ring))
        self.addCleanup(ring.unlink)
        self.addCleanup(ring.close)
        target = ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn")
        marker = PlacementConfig(MAIN_PROCESS, arena=arena)
        self.assertIs(marker.arena, arena)
        pipeline = (
            PipelineBuilder()
            .add_source(range(7))
            .to(target)
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena, buffer_size=2)
            .add_sink(2)
            .build(num_threads=3)
        )
        seen = []
        with pipeline.auto_stop(timeout=10):
            for item in pipeline.get_iterator(timeout=20):
                index = item["id"]
                self.assertTrue(
                    np.array_equal(
                        item["data"]["array"],
                        np.full(16384, index % 251, dtype=np.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        item["data"]["tensor"],
                        torch.full((8192,), index, dtype=torch.int64),
                    )
                )
                self.assertEqual(item["data"]["bytes"], bytes([index % 251]) * 16384)
                seen.append(index)
                del item
        self.assertCountEqual(seen, range(7))
        self.assertEqual(arena.close_calls, 0)
        self.assertEqual(arena.unlink_calls, 0)
        arena.close()
        arena.unlink()
        self.assertEqual(arena.close_calls, 1)
        self.assertEqual(arena.unlink_calls, 1)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=ring.name)

    @parameterized.expand([("tuple",), ("list",)])
    def test_plain_wrong_count_sequence_still_fails_before_spawn(
        self, kind: str
    ) -> None:
        """Normalized plain tuples and lists retain strict per-worker cardinality."""
        arena = self.objects("ring")[0]
        supplied = (arena,) if kind == "tuple" else [arena]
        marker = PlacementConfig(MAIN_PROCESS, arena=supplied)
        self.assertIs(type(marker.arena), tuple)
        config = (
            PipelineBuilder()
            .add_source([])
            .to(ProcessPoolExecutorConfig(max_workers=2, mp_context="spawn"))
            .pipe(_identity)
            .to(MAIN_PROCESS, arena=supplied)
            .add_sink()
            .get_config()
        )
        with (
            patch.object(_ProcessBackend, "spawn") as spawn,
            patch.object(arena, "close") as close,
            patch.object(arena, "unlink") as unlink,
        ):
            with self.assertRaisesRegex(
                ValueError, "needs 2 independent objects; got 1"
            ):
                build_pipeline(config, num_threads=3)
            spawn.assert_not_called()
            close.assert_not_called()
            unlink.assert_not_called()


class BorrowedPendingRootTest(_RegionTestCase):
    def test_retired_queuefull_descriptor_survives_owner_drop_until_caller_close(
        self,
    ) -> None:
        """A retired committed envelope stays discoverable without caller cleanup takeover."""
        arena = SharedMemoryRingBuffer(32768, acquire_timeout=0)
        raw: queue.Queue[Any] = queue.Queue(1)
        raw.put((1, None))
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, arena))
        message = (_RESULT, [b"a" * 32768])
        cleanup_sender = weakref.ref(sender)
        try:
            with self.assertRaises(queue.Full):
                sender.put_nowait(message)
            assert sender._pending is not None
            committed = sender._pending[1][1]
            sender._discard_pending(message)
            self.assertIs(sender._pending[1][1], committed)
            self.assertTrue(sender._pending_retained)
            with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                sender.put_nowait((_RESULT, [b"b" * 32768]))
            sender_ref, queue_ref = weakref.ref(sender), weakref.ref(raw)
            del sender, raw, message
            self.assertIsNotNone(sender_ref())
            self.assertIsNotNone(queue_ref())
            owner = sender_ref()
            assert owner is not None
            self.assertIsNotNone(owner._pending)
            self.assertEqual((arena._head, arena._tail), (32768, 0))
            self.assertFalse(arena._closed)
            owner._release_retired_after_close()
            self.assertIsNotNone(owner._pending)
            arena.close()
            arena.unlink()
            owner._release_retired_after_close()
            self.assertIsNone(owner._pending)
            self.assertFalse(owner._pending_retained)
            del owner
            self.assertIsNone(sender_ref())
            self.assertIsNone(queue_ref())
        finally:
            arena.close()
            arena.unlink()
            pending_owner = cleanup_sender()
            if pending_owner is not None:
                pending_owner._release_retired_after_close()

    def test_pending_registration_failure_occurs_before_native_commit(self) -> None:
        """Retirement root allocation failure cannot orphan a new native span."""
        from spdl.pipeline._arena import _region as region_module

        arena = SharedMemoryRingBuffer(32768, acquire_timeout=0)
        sender = _ArenaSendingQueue(queue.Queue(1), _RESULT, _RegionChannel(0, arena))
        try:
            with patch.object(
                region_module.atexit, "register", side_effect=MemoryError("retain")
            ):
                with self.assertRaisesRegex(MemoryError, "retain"):
                    sender.put_nowait((_RESULT, [bytes(32768)]))
            self.assertEqual((arena._head, arena._tail), (0, 0))
            self.assertIsNone(sender._pending)
            self.assertFalse(sender._pending_retained)
        finally:
            arena.close()
            arena.unlink()

    def test_accepted_retry_releases_pending_root_without_reencoding(self) -> None:
        """Normal QueueFull retry commits once and releases its temporary root on acceptance."""
        arena = SharedMemoryRingBuffer(32768, acquire_timeout=0)
        raw: queue.Queue[Any] = queue.Queue(1)
        raw.put((1, None))
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, arena))
        message = (_RESULT, [bytes(32768)])
        try:
            for _ in range(3):
                with self.assertRaises(queue.Full):
                    sender.put_nowait(message)
            self.assertEqual(arena._head, 32768)
            self.assertTrue(sender._pending_retained)
            raw.get_nowait()
            sender.put_nowait(message)
            self.assertIsNone(sender._pending)
            self.assertFalse(sender._pending_retained)
            envelope = raw.get_nowait()[1]
            self.assertEqual(envelope.sequence, 0)
            self.assertEqual(_RegionChannel(0, arena).decode(envelope), [bytes(32768)])
            self.assertEqual(arena._head, arena._tail)
        finally:
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()

    def test_accepted_descriptor_unregister_failure_never_republishes(self) -> None:
        """Post-acceptance unregister failure retains retry state without duplicate delivery."""
        from spdl.pipeline._arena import _region as region_module

        arena = SharedMemoryRingBuffer(32768, acquire_timeout=0)
        raw: queue.Queue[Any] = queue.Queue(1)
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, arena))
        failure = MemoryError("deliberate unregister failure")
        try:
            with patch.object(
                region_module.atexit, "unregister", side_effect=failure
            ) as unregister:
                sender.put_nowait((_RESULT, [b"a" * 32768]))
                unregister.assert_called_once()
            self.assertIsNone(sender._pending)
            self.assertTrue(sender._pending_retained)
            self.assertIs(sender._pending_cleanup_error, failure)
            self.assertFalse(sender._channel.failed)
            self.assertEqual(raw.qsize(), 1)
            self.assertEqual(arena._head, 32768)
            envelope = raw.get_nowait()[1]
            self.assertEqual(envelope.sequence, 0)
            self.assertEqual(_RegionChannel(0, arena).decode(envelope), [b"a" * 32768])
            self.assertTrue(raw.empty())
            self.assertEqual(arena._head, arena._tail)
            self.assertTrue(sender._accepted_cleanup_failed)
            self.assertIsNotNone(sender._accepted_cleanup_message)
            accepted = sender._accepted_cleanup_message
            with self.assertRaisesRegex(RuntimeError, "already accepted"):
                sender.put_nowait(accepted)
            self.assertTrue(raw.empty())
            self.assertEqual(sender._channel._sequence, 1)
            self.assertEqual(arena._head, 32768)
            sender._release_retired_after_close()
            self.assertTrue(sender._pending_retained)
            self.assertIs(sender._pending_cleanup_error, failure)
            self.assertFalse(arena._closed)
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()
            self.assertFalse(sender._pending_retained)
            self.assertIsNone(sender._pending_cleanup_error)
        finally:
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()

    def test_local_holder_close_does_not_release_retired_publication(self) -> None:
        """Local holder closed state is not terminal owning-caller cleanup proof."""
        owner = SharedMemoryRingBuffer(32768, acquire_timeout=0)
        holder = copy.copy(owner)
        raw: queue.Queue[Any] = queue.Queue(1)
        raw.put((1, None))
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, holder))
        message = (_RESULT, [bytes(32768)])
        try:
            with self.assertRaises(queue.Full):
                sender.put_nowait(message)
            sender._discard_pending(message)
            holder.close()
            self.assertTrue(holder._closed)
            self.assertFalse(holder._owns_names)
            self.assertFalse(owner._shutdown)
            sender._release_retired_after_close()
            self.assertIsNotNone(sender._pending)
            self.assertTrue(sender._pending_retained)
        finally:
            owner.close()
            owner.unlink()
            # This isolated unit has no recipient or aliases, and caller
            # terminal cleanup is proven; release test bookkeeping only.
            sender._release_pending_root()


class AcceptedCleanupCancellationTest(_RegionTestCase):
    @parameterized.expand(
        [("ordinary", RuntimeError), ("cancelled", asyncio.CancelledError)]
    )
    def test_unregister_cancellation_preserves_one_accepted_transaction(
        self, _: str, error_type: Any
    ) -> None:
        """Accepted delivery cannot be re-encoded by a retry after cleanup cancellation."""
        from spdl.pipeline._arena import _region as region_module

        arena = SharedMemoryRingBuffer(65536, acquire_timeout=0)
        raw: queue.Queue[Any] = queue.Queue(4)
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, arena))
        first = (_RESULT, [b"a" * 32768])
        cancellation = error_type("post-acceptance cleanup failed")
        try:
            with patch.object(
                region_module.atexit, "unregister", side_effect=cancellation
            ):
                if isinstance(cancellation, Exception):
                    sender.put_nowait(first)
                else:
                    with self.assertRaises(error_type) as error:
                        sender.put_nowait(first)
                    self.assertIs(error.exception, cancellation)
            self.assertEqual(raw.qsize(), 1)
            self.assertEqual(sender._channel._sequence, 1)
            self.assertEqual((arena._head, arena._tail), (32768, 0))
            self.assertIsNone(sender._pending)
            self.assertTrue(sender._pending_retained)
            self.assertFalse(sender._channel.failed)
            self.assertIs(sender._accepted_cleanup_message, first)
            self.assertIs(sender._pending_cleanup_error, cancellation)
            self.assertTrue(sender._accepted_cleanup_failed)
            retry_error = None
            try:
                sender.put_nowait(first)
            except RuntimeError as error:
                retry_error = error
            print(
                "ACCEPTED_CANCEL_RETRY_STATE "
                + str(
                    {
                        "raw_accepted_descriptors": raw.qsize(),
                        "producer_sequence": sender._channel._sequence,
                        "_head": arena._head,
                        "_tail": arena._tail,
                        "retry_error": str(retry_error),
                    }
                ),
                flush=True,
            )
            self.assertIsNotNone(retry_error)
            self.assertIn("already accepted", str(retry_error))
            self.assertEqual(raw.qsize(), 1)
            self.assertEqual(sender._channel._sequence, 1)
            self.assertEqual(arena._head, 32768)
            receiver = _RegionChannel(0, arena)
            envelope = raw.get_nowait()[1]
            self.assertEqual(receiver.decode(envelope), [b"a" * 32768])
            # An equal-value fresh transaction is not the same object. A
            # cancelled sender is retired rather than silently resuming it.
            fresh = (_RESULT, [b"a" * 32768])
            self.assertIsNot(fresh, first)
            with self.assertRaisesRegex(RuntimeError, "already accepted"):
                sender.put_nowait(fresh)
            self.assertTrue(raw.empty())
            self.assertEqual(arena._head, arena._tail)
            sender._release_retired_after_close()
            self.assertTrue(sender._pending_retained)
            self.assertIs(sender._accepted_cleanup_message, first)
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()
            self.assertFalse(sender._pending_retained)
            self.assertIsNone(sender._accepted_cleanup_message)
            self.assertIsNone(sender._pending_cleanup_error)
        finally:
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()

    def test_normal_accepted_same_value_transactions_are_distinct(self) -> None:
        """Normal successful delivery permits distinct repeated equal-value transactions."""
        arena = SharedMemoryRingBuffer(65536, acquire_timeout=0)
        raw: queue.Queue[Any] = queue.Queue(4)
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, arena))
        receiver = _RegionChannel(0, arena)
        try:
            first = (_RESULT, [b"a" * 32768])
            second = (_RESULT, [b"a" * 32768])
            self.assertIsNot(first, second)
            sender.put_nowait(first)
            self.assertEqual(receiver.decode(raw.get_nowait()[1]), [b"a" * 32768])
            sender.put_nowait(second)
            self.assertEqual(receiver.decode(raw.get_nowait()[1]), [b"a" * 32768])
            self.assertEqual(sender._channel._sequence, 2)
            self.assertEqual(arena._head, arena._tail)
            self.assertFalse(sender._pending_retained)
        finally:
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()

    def test_normal_successful_same_message_object_is_published_twice(self) -> None:
        """Without cleanup failure, two puts of the same object remain two publications."""
        arena = SharedMemoryRingBuffer(65536, acquire_timeout=0)
        raw: queue.Queue[Any] = queue.Queue(4)
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, arena))
        receiver = _RegionChannel(0, arena)
        message = (_RESULT, [b"a" * 32768])
        try:
            sequences = []
            for _ in range(2):
                sender.put_nowait(message)
                self.assertEqual(raw.qsize(), 1)
                envelope = raw.get_nowait()[1]
                sequences.append(envelope.sequence)
                self.assertEqual(receiver.decode(envelope), [b"a" * 32768])
                self.assertIsNone(sender._pending)
                self.assertIsNone(sender._accepted_cleanup_message)
                self.assertIsNone(sender._pending_cleanup_error)
                self.assertFalse(sender._accepted_cleanup_failed)
                self.assertFalse(sender._pending_retained)
                self.assertFalse(sender._channel.failed)
            self.assertEqual(sequences, [0, 1])
            self.assertEqual(sender._channel._sequence, 2)
            self.assertEqual((arena._head, arena._tail), (65536, 65536))
            self.assertTrue(raw.empty())
        finally:
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()


_POOL_WAIT_WRITER_OBSERVED = False


def _observe_pool_writer_initializer(arena: Any, sender: Any) -> None:
    global _POOL_WAIT_WRITER_OBSERVED
    _POOL_WAIT_WRITER_OBSERVED = False
    original = arena._wait_space

    def wait(deadline: Any, control_deadline: Any) -> Any:
        global _POOL_WAIT_WRITER_OBSERVED
        if not _POOL_WAIT_WRITER_OBSERVED:
            _POOL_WAIT_WRITER_OBSERVED = True
            sender.send_bytes(b"false-native-alias-capacity-wait")
        return original(deadline, control_deadline)

    arena._wait_space = wait
    torch.set_num_threads(1)


def _bounded_native_alias_scenario(report: Any) -> None:
    import faulthandler
    import hashlib
    import signal
    from pathlib import Path

    from spdl.pipeline._arena import _pool as pool_module

    os.setsid()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    _disable_network_hooks()
    torch.set_num_threads(1)
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    arena = SharedMemorySegmentPool(131072, 1, acquire_timeout=90)
    pipeline = None
    held = None

    def state() -> Any:
        reader = arena._reader
        return {
            "_published": arena._published,
            "read_cursor": arena._read_cursor,
            "_reclaimed": arena._reclaimed,
            "mapping_exports": dict(arena._mapping_exports),
            "released": sorted(reader._released) if reader is not None else None,
            "creator_closed": arena._closed,
            "creator_shutdown": arena._shutdown,
        }

    report.send(
        {
            "stage": "allocated",
            "scenario_pid": os.getpid(),
            "names": [arena._ctrl_name, *arena._seg_names],
            "pool_source_sha256": hashlib.sha256(
                Path(pool_module.__file__).read_bytes()
            ).hexdigest(),
        }
    )
    try:
        pipeline = (
            PipelineBuilder()
            .add_source([3], continuous=True)
            .to(
                ProcessPoolExecutorConfig(
                    max_workers=1,
                    mp_context="spawn",
                    initializer=_observe_pool_writer_initializer,
                    initargs=(arena, sender),
                )
            )
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(2)
            .build(num_threads=3)
        )
        manager = pipeline._impl._pools[0]
        process = manager._workers[0]._proc
        report.send({"stage": "worker_started", "worker_pid": process.pid})
        pipeline.start()
        count = 0
        for item in pipeline.get_iterator(timeout=20):
            self_check = item["id"] == 3 and torch.equal(
                item["data"]["tensor"], torch.full((8192,), 3, dtype=torch.int64)
            )
            if not self_check:
                raise AssertionError("First epoch whole tensor check failed")
            held = item["data"]["tensor"].detach()
            count += 1
            del item
        if count != 1 or not receiver.poll(20):
            raise AssertionError(
                "Full epoch/native-alias capacity waiter not established"
            )
        receiver.recv_bytes()
        report.send({"stage": "before_stop_held", "state": state()})
        pipeline.stop(timeout=10)
        report.send(
            {
                "stage": "after_stop_held",
                "state": state(),
                "worker_exitcode": process.exitcode,
                "worker_alive": process.is_alive(),
            }
        )
        assert held is not None
        if not torch.equal(held, torch.full((8192,), 3, dtype=torch.int64)):
            raise AssertionError("Native alias became invalid after stop")
        report.send({"stage": "before_alias_release", "state": state()})
        held = None
        report.send({"stage": "after_alias_release", "state": state()})
        if arena._reclaimed != arena._published or any(arena._mapping_exports.values()):
            raise AssertionError("Released alias still occupies storage")
        pipeline = (
            PipelineBuilder()
            .add_source(range(20, 24))
            .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
            .pipe(_payload)
            .to(MAIN_PROCESS, arena=arena)
            .add_sink(2)
            .build(num_threads=3)
        )
        with pipeline.auto_stop(timeout=10):
            seen = []
            for item in pipeline.get_iterator(timeout=10):
                index = item["id"]
                if (
                    not torch.equal(
                        item["data"]["tensor"],
                        torch.full((8192,), index, dtype=torch.int64),
                    )
                    or not np.array_equal(
                        item["data"]["array"],
                        np.full(16384, index % 251, dtype=np.uint8),
                    )
                    or item["data"]["bytes"] != bytes([index % 251]) * 16384
                ):
                    raise AssertionError("Same pool fresh values failed")
                seen.append(index)
                del item
            if sorted(seen) != list(range(20, 24)):
                raise AssertionError("Same pool fresh IDs failed")
        report.send(
            {"stage": "same_pool_reuse_verified", "state": state(), "ids": seen}
        )
    except BaseException as error:
        report.send(
            {"stage": "error", "type": type(error).__name__, "detail": str(error)}
        )
        raise
    finally:
        held = None
        report.send({"stage": "before_terminal_recovery"})
        arena.close()
        if pipeline is not None:
            try:
                pipeline.stop(timeout=15)
            except PipelineFailure:
                pass
        arena.unlink()
        receiver.close()
        sender.close()
        report.send({"stage": "done"})
        report.close()


class BorrowedNativeAliasBoundedDiagnostic(_RegionTestCase):
    @unittest.skipUnless(
        os.name == "posix", "diagnostic probe uses POSIX signals and py-spy"
    )
    def test_real_native_alias_stop_release_reuse_is_bounded(self) -> None:
        """A child-owned probe records alias/cursor stages before any terminal recovery."""
        import signal

        variant = f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
        directory = Path(
            self.enterContext(
                TemporaryDirectory(
                    prefix=f"spdl-native-receipt-{variant}-{os.getpid()}-"
                )
            )
        )
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_bounded_native_alias_scenario, args=(sender,))
        process.start()
        observations = []
        names = []
        worker_pid = None
        complete = False
        try:
            while receiver.poll(20):
                event = receiver.recv()
                observations.append(event)
                if event["stage"] == "allocated":
                    names = event["names"]
                if event["stage"] == "worker_started":
                    worker_pid = event["worker_pid"]
                if event["stage"] in (
                    "before_alias_release",
                    "before_terminal_recovery",
                ):
                    # A short event wait, not a sleep or guessed CPU threshold.
                    if not receiver.poll(5):
                        if process.is_alive():
                            import subprocess

                            stack = subprocess.run(
                                [
                                    "/usr/local/bin/py-spy",
                                    "dump",
                                    "--pid",
                                    str(process.pid),
                                ],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                            observations.append(
                                {
                                    "stage": "blocked_stack",
                                    "returncode": stack.returncode,
                                    "stdout": stack.stdout,
                                    "stderr": stack.stderr,
                                }
                            )
                        break
                if event["stage"] == "done":
                    complete = True
                    break
            self.assertTrue(complete, str(observations))
            after_stop = next(
                event for event in observations if event["stage"] == "after_stop_held"
            )
            self.assertIsNotNone(after_stop["worker_exitcode"])
            self.assertFalse(after_stop["worker_alive"])
            self.assertEqual(after_stop["state"]["read_cursor"], 1)
            self.assertEqual(after_stop["state"]["_reclaimed"], 0)
            self.assertTrue(any(after_stop["state"]["mapping_exports"].values()))
            after_release = next(
                event
                for event in observations
                if event["stage"] == "after_alias_release"
            )
            self.assertEqual(
                after_release["state"]["_published"],
                after_release["state"]["_reclaimed"],
            )
            self.assertFalse(any(after_release["state"]["mapping_exports"].values()))
            self.assertFalse(after_release["state"]["creator_closed"])
            self.assertFalse(after_release["state"]["creator_shutdown"])
            self.assertTrue(
                any(
                    event["stage"] == "same_pool_reuse_verified"
                    for event in observations
                ),
                str(observations),
            )
            receipt = directory / "native-positive-observations.json"
            receipt.write_text(json.dumps(observations, indent=2) + "\n")
            print("NATIVE_ALIAS_RECEIPT " + json.dumps(observations), flush=True)
        finally:
            if process.is_alive():
                assert process.pid is not None
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            # Only known names from this isolated, now-reaped owned process group.
            safe = process.exitcode is not None
            if worker_pid is not None:
                try:
                    os.kill(worker_pid, 0)
                    safe = False
                except ProcessLookupError:
                    pass
            if safe:
                for name in names:
                    try:
                        memory = shared_memory.SharedMemory(name=name)
                    except FileNotFoundError:
                        continue
                    memory.unlink()
                    memory.close()
            receiver.close()
            sender.close()


class CorrectedUnreadAdmissionTest(_RegionTestCase):
    @parameterized.expand([("pool",), ("ring",)])
    def test_cancelled_committed_unit_refuses_fresh_writer_without_reclaim(
        self, kind: str
    ) -> None:
        """Fresh writers refuse unread storage without freeing or reinterpreting it."""
        arena: Any = (
            SharedMemorySegmentPool(32768, 2, acquire_timeout=0)
            if kind == "pool"
            else SharedMemoryRingBuffer(65536, acquire_timeout=0)
        )
        raw: queue.Queue[Any] = queue.Queue(1)
        raw.put((1, None))
        sender = _ArenaSendingQueue(raw, _RESULT, _RegionChannel(0, arena))
        message = (_RESULT, [b"a" * 32768])
        try:
            with self.assertRaises(queue.Full):
                sender.put_nowait(message)
            self.assertIsNotNone(sender._pending)
            assert sender._pending is not None
            committed = sender._pending[1][1]
            sender._discard_pending(message)
            self.assertIs(sender._pending[1][1], committed)
            self.assertTrue(sender._channel.failed)
            raw.get_nowait()
            self.assertTrue(raw.empty())
            if kind == "pool":
                before = (arena._published, arena._read_cursor, arena._reclaimed)
                self.assertEqual(before, (1, 0, 0))
                self.assertFalse(any(arena._mapping_exports.values()))
            else:
                before = (arena._head, arena._tail)
                self.assertEqual(before, (32768, 0))
            with self.assertRaisesRegex(RuntimeError, "unread"):
                _RegionChannel(0, arena).encode([b"b" * 32768])
            if kind == "pool":
                self.assertEqual(
                    (arena._published, arena._read_cursor, arena._reclaimed), before
                )
            else:
                self.assertEqual((arena._head, arena._tail), before)
            self.assertFalse(arena._closed)
            self.assertFalse(arena._shutdown)
            self.assertIsNotNone(sender._pending)
        finally:
            arena.close()
            arena.unlink()
            sender._release_retired_after_close()

    def test_delivered_native_alias_allows_new_writer_and_preserves_old_value(
        self,
    ) -> None:
        """Delivered-but-unreclaimed native storage is not mistaken for unread metadata."""
        arena = SharedMemorySegmentPool(32768, 2, acquire_timeout=0)
        held = None
        try:
            first = _RegionChannel(0, arena).encode(
                [torch.full((4096,), 3, dtype=torch.int64)]
            )
            first_receiver = _RegionChannel(0, arena)
            restored = first_receiver.decode(first)
            assert restored is not None
            held = restored[0].detach()
            del restored
            self.assertEqual(
                (arena._published, arena._read_cursor, arena._reclaimed), (1, 1, 0)
            )
            self.assertTrue(any(arena._mapping_exports.values()))
            second = _RegionChannel(0, arena).encode(
                [torch.full((4096,), 7, dtype=torch.int64)]
            )
            restored = _RegionChannel(0, arena).decode(second)
            assert restored is not None
            self.assertTrue(
                torch.equal(restored[0], torch.full((4096,), 7, dtype=torch.int64))
            )
            self.assertTrue(
                torch.equal(held, torch.full((4096,), 3, dtype=torch.int64))
            )
            del restored
            self.assertEqual(arena._read_cursor, 2)
            held = None
            self.assertFalse(any(arena._mapping_exports.values()))
            self.assertEqual(arena._published, arena._reclaimed)
        finally:
            held = None
            arena.close()
            arena.unlink()
