# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import itertools
import json
import os
import queue as _queue
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from spdl.pipeline import (
    AsyncQueue,
    build_pipeline,
    Pipeline,
    PipelineBuilder,
    PipelineFailure,
    run_pipeline_in_subprocess,
    TaskHook,
)
from spdl.pipeline._components import _subprocess_pipe
from spdl.pipeline._components._common import StageInfo
from spdl.pipeline._fuse import _find_fusable_runs
from spdl.pipeline.config import set_default_hook_class, set_default_queue_class
from spdl.pipeline.defs import Pipe


def add_one(x: int) -> int:
    return x + 1


def _raise_initializer() -> None:
    raise RuntimeError("initializer boom")


def times_two(x: int) -> int:
    return x * 2


def boom(x: int) -> int:
    raise ValueError("boom")


def route_by_parity(x: int) -> int:
    """Router: even items to path 0, odd items to path 1."""
    return x % 2


async def aident(x: int) -> int:
    """An async (no-executor) branch stage; runs on the worker's loop once fused."""
    return x


class _PidStamp:
    """A mutable item each stage stamps its ``os.getpid()`` into, to prove process-locality.

    Picklable, so it survives the main -> worker -> main round-trip; the stamps written inside a
    worker travel back on the returned copy.
    """

    def __init__(self, value: int) -> None:
        self.value = value
        self.pids: dict[str, int] = {}


def _stamp_router(item: _PidStamp) -> int:
    """Path-variants router: records its pid, then routes by parity."""
    item.pids["router"] = os.getpid()
    return item.value % 2


def _stamp_branch(item: _PidStamp) -> _PidStamp:
    """A branch stage (used on both paths): records its pid."""
    item.pids["branch"] = os.getpid()
    return item


def _stamp_merge(item: _PidStamp) -> _PidStamp:
    """Post-merge (fan-in) stage: records its pid after the branches merge back."""
    item.pids["merge"] = os.getpid()
    return item


class _Unpicklable:
    """An object that refuses to be pickled, to prove it never crosses a process boundary."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __reduce__(self) -> Any:
        raise TypeError("_Unpicklable must not be pickled")


def wrap(x: int) -> _Unpicklable:
    return _Unpicklable(x + 1)


def unwrap(o: _Unpicklable) -> int:
    return o.value * 2


def dup(x: int) -> Iterator[int]:
    """A sync-generator op: yields two values per input (1->2 fan-out)."""
    yield x * 2
    yield x * 2 + 1


async def adup(x: int) -> AsyncIterator[int]:
    """An async-generator op: yields two values per input (1->2 fan-out)."""
    yield x
    yield x + 1


def _run(pipeline: Pipeline[Any], timeout: float = 60.0) -> list[Any]:
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=timeout))


def stall_on_first(x: int) -> int:
    """Stall only on the earliest item, so one worker holds it while its peer races ahead."""
    if x == 0:
        time.sleep(3.0)
    return x + 1


class _Buffer1Queue(AsyncQueue):
    """An ``AsyncQueue`` pinned to ``buffer_size=1`` (no prefetch slack)."""

    def __init__(
        self, info: Any, *, buffer_size: int = 1, interval: float = -1
    ) -> None:
        super().__init__(info, buffer_size=1)


def _install_small_buffers() -> None:
    """Executor initializer: pin the worker sub-pipeline's queues to ``buffer_size=1``.

    A single-slot buffer stops the worker holding the earliest item from prefetching past it, so
    that worker stays parked on the slow item while its peer drains the rest — the timing that
    surfaces the startup race below (a cheap, deterministic stand-in for the recording overhead
    that first exposed it under stress).
    """
    set_default_queue_class(_Buffer1Queue)


class SubprocessPipelineFuseTest(unittest.TestCase):
    def test_two_pool_stages_fused_match_unfused(self) -> None:
        """A fused two-stage process-pool pipeline matches the unfused result."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))

        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=3)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_unpicklable_intermediate_passes_through_fused(self) -> None:
        """A fused run keeps the op->op handoff in-process, so an unpicklable value works."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))

        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(wrap, executor=ex, concurrency=2)
            .pipe(unwrap, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_generator_op_fused(self) -> None:
        """A generator op is fusable, fanning out 1->N inside the worker sub-pipeline."""
        n = 8
        # dup yields {2x, 2x+1}; add_one shifts each by one, covering 1..2n exactly.
        ref = list(range(1, 2 * n + 1))
        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(dup, executor=ex, concurrency=2)
            .pipe(add_one, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_async_generator_op_composes_with_fused(self) -> None:
        """An async-generator op (main-process) fans out downstream of a fused run."""
        n = 8
        # add_one+times_two fuse to (x+1)*2; adup then yields {v, v+1}, covering 2..2n+1.
        ref = list(range(2, 2 * n + 2))
        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .pipe(adup)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_aggregate_between_pool_stages_not_absorbed(self) -> None:
        """An aggregate between two pool stages stays in the main process (not absorbed).

        Its main-process batching semantics are unchanged: each batch holds exactly
        ``aggregate``'s size.
        """
        n = 12
        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .aggregate(3)
            .pipe(len, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        # Each aggregated batch has exactly 3 items, so every produced value is 3.
        out = _run(fused)
        self.assertTrue(all(v == 3 for v in out))
        self.assertEqual(sum(out), n)

    def test_initializer_failure_surfaces_not_hangs(self) -> None:
        """A worker initializer that raises surfaces as a failure instead of hanging.

        A real failure surfaces as :py:class:`PipelineFailure`; a hung pipeline would instead
        raise :py:class:`TimeoutError` from the finite ``get_iterator`` timeout, so asserting on
        ``PipelineFailure`` proves the failed initializer is reported rather than wedging the
        collector forever.
        """
        ex = ProcessPoolExecutor(max_workers=2, initializer=_raise_initializer)
        fused = (
            PipelineBuilder()
            .add_source(range(2))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(4)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with self.assertRaises(PipelineFailure):
            _run(fused, timeout=30.0)

    def test_flag_off_is_unaffected(self) -> None:
        """Without the flag, the same pipeline runs the stages normally and matches."""
        n = 10
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=False)
        )
        self.assertEqual(sorted(_run(pipeline)), ref)


class StartupRaceFuseTest(unittest.TestCase):
    """A slow worker holding the earliest items must not have them silently dropped.

    Regression test for the silent earliest-item drop in the non-continuous fused
    pipeline. With a single shared work-stealing input queue, a worker that drains its
    stream and reaches ``_SESSION_END`` early could loop back and consume a second end
    marker meant for a slower peer still holding (un-flushed) items; that peer then
    never ended, the collector reached its ``_DONE`` count from the wrong workers and
    finished, and the peer's earliest items were dropped with no error. Per-worker input
    queues (one ``_SESSION_END`` each) make that impossible. ``stall_on_first`` +
    ``buffer_size=1`` deterministically pin the worker that grabs item ``0`` while its
    peer races through the rest — exactly the interleaving the race needs.
    """

    def test_slow_worker_does_not_drop_earliest_items(self) -> None:
        """A worker stalled on the earliest item still delivers it instead of dropping it."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2, initializer=_install_small_buffers)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(stall_on_first, executor=ex, concurrency=1)
            .pipe(times_two, executor=ex, concurrency=1)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)


class ContinuousFuseTest(unittest.TestCase):
    """Fusion with a continuous (multi-epoch) source."""

    def test_multi_epoch_correct(self) -> None:
        """A continuous fused pipeline yields the correct set each epoch."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=3)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(3):  # three epochs from the same warm worker pool
                epoch = sorted(pipeline.get_iterator(timeout=60))
                self.assertEqual(epoch, ref)

    def test_unpicklable_intermediate_multi_epoch(self) -> None:
        """The unpicklable op->op handoff keeps working across epochs."""
        n = 10
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(wrap, executor=ex, concurrency=2)
            .pipe(unwrap, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_aggregate_not_absorbed_multi_epoch(self) -> None:
        """A non-absorbed aggregate keeps single-flush-per-epoch semantics across epochs.

        Because the aggregate runs in the main process (not per worker), each epoch produces
        full-size batches plus one combined partial, and every item is accounted for.
        """
        n = 12
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .aggregate(3)
            .pipe(len, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):  # each epoch's batches cover exactly n items
                out = list(pipeline.get_iterator(timeout=60))
                self.assertEqual(sum(out), n)

    def test_fewer_items_than_workers(self) -> None:
        """An epoch with fewer items than workers still completes (some workers run empty)."""
        n = 2
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=4)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_generator_op_multi_epoch(self) -> None:
        """A fused generator op keeps its 1->N fan-out correct across epochs."""
        n = 8
        ref = list(range(1, 2 * n + 1))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(dup, executor=ex, concurrency=2)
            .pipe(add_one, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(3):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_async_generator_op_multi_epoch(self) -> None:
        """A main-process async-generator op fans out downstream of a fused run each epoch."""
        n = 8
        ref = list(range(2, 2 * n + 2))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .pipe(adup)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(3):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_continuous_op_failure_does_not_deadlock(self) -> None:
        """Op failures (dropped per SPDL default) still let each epoch's barrier complete.

        ``boom`` raises on every item, so the fused workers produce no results; the test checks
        the continuous epoch barrier still completes each epoch (workers report the boundary
        with zero results) instead of deadlocking, matching unfused drop-on-failure behavior.
        """
        n = 8
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(boom, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(list(pipeline.get_iterator(timeout=60)), [])

    def test_continuous_in_subprocess(self) -> None:
        """Continuous fusion composes with run_pipeline_in_subprocess across epochs."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        config = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(
            config, num_threads=4, fuse_subprocess_stages=True
        )
        for _ in range(3):  # one epoch per iteration
            self.assertEqual(sorted(src), ref)

    def test_teardown_mid_stream_does_not_hang(self) -> None:
        """Tearing a continuous fused subprocess pipeline down mid-stream must not hang.

        A teardown before the stream is drained leaves the per-worker input queue full; the
        pool shutdown must cancel the queue feeder-thread join instead of blocking forever
        waiting to flush buffered items into a pipe the (terminated) workers no longer drain.
        Uses the same path data loaders do — ``run_pipeline_in_subprocess`` + fusion — and
        triggers teardown via the iterable's documented ``_finalizer`` handle.
        """
        n = 100_000  # far more than any buffer, so the stream is still full at teardown
        ex = ProcessPoolExecutor(max_workers=2)
        self.addCleanup(ex.shutdown)
        config = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(2)
            .get_config()
        )
        src = run_pipeline_in_subprocess(
            config, num_threads=4, fuse_subprocess_stages=True
        )
        it = iter(src)
        next(it)  # consume a couple of items so the queues stay backed up
        next(it)
        torn_down = threading.Event()

        def _teardown() -> None:
            src._finalizer()  # pyre-ignore[16]: documented teardown handle
            torn_down.set()

        t = threading.Thread(target=_teardown, daemon=True)
        t.start()
        self.assertTrue(
            torn_down.wait(timeout=60),
            "fused-pool teardown hung on a full input queue after a mid-stream stop",
        )
        t.join(timeout=10)


class PathVariantsFuseTest(unittest.TestCase):
    """Fusion of a path-variants stage whose branches share one pool executor."""

    def test_lone_path_variants_pool_branches_fused_match_unfused(self) -> None:
        """A single path-variants stage with same-pool branches fuses and matches unfused.

        Even items take the ``add_one`` branch, odd items the ``times_two`` branch. The whole
        routing construct moves into one worker, so the result must match the unfused run.
        """
        n = 16
        ref = sorted(x + 1 if x % 2 == 0 else x * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        builder = (
            PipelineBuilder()
            .add_source(range(n))
            .path_variants(
                route_by_parity,
                [[Pipe(add_one, executor=ex)], [Pipe(times_two, executor=ex)]],
            )
            .add_sink(n)
        )
        config = builder.get_config()
        # A lone same-pool path-variants is a fusable run on its own.
        self.assertEqual(len(_find_fusable_runs(config.pipes)), 1)
        pipeline = build_pipeline(config, num_threads=4, fuse_subprocess_stages=True)
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_path_variants_async_branch_stage_fuses(self) -> None:
        """A branch mixing an async (no-executor) stage with a pool stage still fuses.

        Mirrors a cache miss-path shape (async I/O then pool CPU): the async stage runs on the
        worker's loop and the pool stage on its threads, all inside one worker.
        """
        n = 16
        ref = sorted(x + 1 if x % 2 == 0 else x * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        builder = (
            PipelineBuilder()
            .add_source(range(n))
            .path_variants(
                route_by_parity,
                [
                    [Pipe(add_one, executor=ex)],
                    [Pipe(aident), Pipe(times_two, executor=ex)],
                ],
            )
            .add_sink(n)
        )
        config = builder.get_config()
        self.assertEqual(len(_find_fusable_runs(config.pipes)), 1)
        pipeline = build_pipeline(config, num_threads=4, fuse_subprocess_stages=True)
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_path_variants_mixed_executors_not_fused(self) -> None:
        """Branches on two different pool executors are not fused, and still run correctly."""
        n = 16
        ref = sorted(x + 1 if x % 2 == 0 else x * 2 for x in range(n))
        ex1 = ProcessPoolExecutor(max_workers=2)
        ex2 = ProcessPoolExecutor(max_workers=2)
        builder = (
            PipelineBuilder()
            .add_source(range(n))
            .path_variants(
                route_by_parity,
                [[Pipe(add_one, executor=ex1)], [Pipe(times_two, executor=ex2)]],
            )
            .add_sink(n)
        )
        config = builder.get_config()
        self.assertEqual(_find_fusable_runs(config.pipes), [])
        pipeline = build_pipeline(config, num_threads=4, fuse_subprocess_stages=True)
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_path_variants_fused_with_adjacent_pool_pipe(self) -> None:
        """A pool-pipe adjacent to a same-pool path-variants stage fuses into one run."""
        n = 16
        # add_one shifts each item to v=x+1; then even v -> add_one (v+1), odd v -> times_two (v*2).
        ref = sorted(
            (v + 1) if v % 2 == 0 else (v * 2) for v in (x + 1 for x in range(n))
        )
        ex = ProcessPoolExecutor(max_workers=2)
        builder = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex)
            .path_variants(
                route_by_parity,
                [[Pipe(add_one, executor=ex)], [Pipe(times_two, executor=ex)]],
            )
            .add_sink(n)
        )
        config = builder.get_config()
        runs = _find_fusable_runs(config.pipes)
        self.assertEqual(len(runs), 1)
        self.assertEqual(
            (runs[0].start, runs[0].stop), (0, 2)
        )  # both stages in one run
        pipeline = build_pipeline(config, num_threads=4, fuse_subprocess_stages=True)
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_path_variants_fused_multi_epoch(self) -> None:
        """A fused path-variants stage stays correct across epochs of a continuous source."""
        n = 12
        ref = sorted(x + 1 if x % 2 == 0 else x * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .path_variants(
                route_by_parity,
                [[Pipe(add_one, executor=ex)], [Pipe(times_two, executor=ex)]],
            )
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(3):  # three epochs from the same warm worker pool
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def _assert_one_worker_process(self, item: _PidStamp, main_pid: int) -> None:
        """The router, branch, and post-merge stages all ran in one worker process (not main)."""
        self.assertEqual(set(item.pids), {"router", "branch", "merge"})
        # router -> branch -> merge stayed in the SAME process for this item ...
        self.assertEqual(item.pids["router"], item.pids["branch"])
        self.assertEqual(item.pids["branch"], item.pids["merge"])
        # ... and that process is a worker, not the main process.
        self.assertNotEqual(item.pids["router"], main_pid)

    def test_path_variants_router_branches_merge_share_one_process(self) -> None:
        """Router, branches, and the fan-in (post-merge) stage all run in one worker process.

        Each stage stamps ``os.getpid()`` into its item, so the assertion proves the whole
        router -> branch -> merge -> post-merge chain runs inside a single worker process per
        item (distinct from the main process): fusion leaves no per-stage process hop. A
        regression that stopped fusing path-variants would run them in the main process and
        fail the ``assertNotEqual(main_pid)`` check.
        """
        n = 16
        ex = ProcessPoolExecutor(max_workers=2)
        builder = (
            PipelineBuilder()
            .add_source([_PidStamp(x) for x in range(n)])
            .path_variants(
                _stamp_router,
                [
                    [Pipe(_stamp_branch, executor=ex)],
                    [Pipe(_stamp_branch, executor=ex)],
                ],
            )
            .pipe(_stamp_merge, executor=ex)  # fan-in / post-merge, same pool
            .add_sink(n)
        )
        config = builder.get_config()
        # path-variants + the post-merge pool-pipe fuse into a single run.
        self.assertEqual(len(_find_fusable_runs(config.pipes)), 1)
        pipeline = build_pipeline(config, num_threads=4, fuse_subprocess_stages=True)
        results = _run(pipeline)

        self.assertEqual(len(results), n)
        main_pid = os.getpid()
        worker_pids = set()
        for item in results:
            self._assert_one_worker_process(item, main_pid)
            worker_pids.add(item.pids["router"])
        # work really ran in worker process(es), never the main process.
        self.assertTrue(worker_pids)
        self.assertNotIn(main_pid, worker_pids)

    def test_path_variants_process_locality_multi_epoch(self) -> None:
        """Process-locality holds for a continuous (multi-epoch) fused path-variants too.

        This is the shape ``op_mode=mp`` uses. Pickling isolates each epoch's stamps (the
        source objects are copied into the worker), so every epoch's output items carry a fresh
        same-process router/branch/merge stamp.
        """
        n = 12
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source([_PidStamp(x) for x in range(n)], continuous=True)
            .path_variants(
                _stamp_router,
                [
                    [Pipe(_stamp_branch, executor=ex)],
                    [Pipe(_stamp_branch, executor=ex)],
                ],
            )
            .pipe(_stamp_merge, executor=ex)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        main_pid = os.getpid()
        with pipeline.auto_stop():
            for _ in range(2):  # two epochs from the same warm worker pool
                items = list(pipeline.get_iterator(timeout=60))
                self.assertEqual(len(items), n)
                for item in items:
                    self._assert_one_worker_process(item, main_pid)


class FuseInSubprocessTest(unittest.TestCase):
    """Fusion composed with whole-pipeline subprocess execution."""

    def test_fused_run_in_subprocess_matches(self) -> None:
        """A fused config run via run_pipeline_in_subprocess yields the same result."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        config = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=3)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(
            config, num_threads=4, fuse_subprocess_stages=True
        )
        self.assertEqual(sorted(src), ref)

    def test_fused_unpicklable_intermediate_in_subprocess(self) -> None:
        """The fused handle survives the trip into the pipeline subprocess and the
        unpicklable op->op handoff stays inside a worker."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        config = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(wrap, executor=ex, concurrency=2)
            .pipe(unwrap, executor=ex, concurrency=2)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(
            config, num_threads=4, fuse_subprocess_stages=True
        )
        self.assertEqual(sorted(src), ref)


class FeedAbortTest(unittest.TestCase):
    """The bridge feeder must wind down promptly when the collector signals abort."""

    def test_feed_ends_session_when_aborted_while_idle(self) -> None:
        """A feeder parked on an empty input queue still emits the per-worker _SESSION_END.

        On a worker error the collector sets ``abort`` while the feeder is typically
        blocked waiting on a slow/idle upstream. The feeder must wake and send exactly one
        ``_SESSION_END`` onto each worker's own queue so the collector can drain every
        ``_DONE`` instead of hanging until its stall timeout. Driving ``_feed`` directly
        keeps the abort-while-idle race deterministic.
        """
        num_workers = 3

        async def _scenario() -> list[list[Any]]:
            in_qs: list[_queue.Queue[Any]] = [
                _queue.Queue() for _ in range(num_workers)
            ]
            input_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="input")
            )  # stays empty -> get() blocks
            abort = asyncio.Event()
            feeder_idle = asyncio.Event()
            with ThreadPoolExecutor(max_workers=num_workers + 1) as ex:
                task = asyncio.ensure_future(
                    _subprocess_pipe._feed(input_queue, in_qs, ex, abort, feeder_idle)
                )
                await asyncio.sleep(0.1)  # let the feeder park on input_queue.get()
                self.assertFalse(task.done(), "feeder should be parked on empty queue")
                abort.set()
                await asyncio.wait_for(task, timeout=5.0)
            return [[q.get_nowait() for _ in range(q.qsize())] for q in in_qs]

        msgs = asyncio.run(_scenario())
        # Every worker's own queue receives exactly one _SESSION_END.
        self.assertEqual(msgs, [[(_subprocess_pipe._SESSION_END, None)]] * num_workers)


class StallGuardTest(unittest.TestCase):
    """The collector's stall guard against an abruptly-dead worker."""

    def test_check_stall_raises_past_timeout(self) -> None:
        """``_check_stall`` raises once no message has arrived for longer than the bound."""
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 0.0
        try:
            with self.assertRaises(TimeoutError):
                _subprocess_pipe._check_stall(time.monotonic() - 1.0)
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig

    def test_check_stall_quiet_within_timeout(self) -> None:
        """``_check_stall`` does not raise while progress is within the bound."""
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 60.0
        try:
            _subprocess_pipe._check_stall(time.monotonic())  # should not raise
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig

    def test_collect_suppresses_stall_while_feeder_idle(self) -> None:
        """An idle feeder suppresses the collector's stall guard during input starvation.

        With the timeout pinned to zero, any stall check on an empty queue would trip instantly;
        the collector must instead keep draining while ``feeder_idle`` is set (nothing dispatched,
        no worker message due) and still finish once the worker reports ``_DONE``.
        """
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 0.0

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )
            abort = asyncio.Event()
            feeder_idle = asyncio.Event()
            feeder_idle.set()  # feeder parked on an idle upstream -> no message expected
            with ThreadPoolExecutor(max_workers=2) as ex:
                task = asyncio.ensure_future(
                    _subprocess_pipe._collect(
                        out_q, 1, output_queue, ex, abort, feeder_idle
                    )
                )
                await asyncio.sleep(
                    0.6
                )  # several empty poll cycles; must not trip the guard
                self.assertFalse(
                    task.done(), "idle feeder must suppress the stall guard"
                )
                out_q.put((_subprocess_pipe._DONE, None))
                await asyncio.wait_for(task, timeout=5.0)

        try:
            asyncio.run(_scenario())
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig


# Set inside each fused worker process by ``_install_recording_hooks`` (the executor
# initializer). The worker is a separate process, so the recording hooks below cannot share
# in-memory counters with the test; each instead writes a small JSON file into this directory,
# which the test reads back to prove the hooks fired inside the worker.
_EVIDENCE_DIR: str | None = None

# A per-process monotonic counter for evidence filenames. ``id(self)`` is only unique among
# simultaneously-live objects -- CPython recycles addresses, so two recorders created and
# destroyed in sequence could collide on one path and clobber each other's count. This counter
# (combined with ``pid``) is guaranteed distinct for every write within a process.
_EVIDENCE_SEQ = itertools.count()
_EVIDENCE_SEQ_LOCK = threading.Lock()


def _write_evidence(record: dict[str, Any]) -> None:
    if (d := _EVIDENCE_DIR) is None:
        return
    with _EVIDENCE_SEQ_LOCK:
        seq = next(_EVIDENCE_SEQ)
    # ``pid`` proves the write came from a worker process (never the main/test process); ``seq``
    # keeps each writer's file distinct within that process, so no two writers ever race on one
    # path. Write to a temp file then atomically rename, so a reader polling mid-write never sees
    # a truncated/partial file.
    path = os.path.join(d, f"{record['kind']}-{record['pid']}-{seq}.json")
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(record))
        os.replace(tmp, path)
    except OSError:
        # This runs inside the worker's stage_hook ``finally`` (stage teardown). It must never
        # raise into the stage: an I/O failure here -- e.g. the evidence ``TemporaryDirectory``
        # already torn down in a teardown race -- would otherwise crash the stage and drop items
        # from the data path the test is validating. A lost evidence file at most makes an
        # assertion under-count, which surfaces as a clear assertion failure (not a baffling
        # dropped-item one).
        pass


class _RecordingTaskHook(TaskHook):
    """A ``TaskHook`` that records how many tasks ran in its (subprocess) stage."""

    def __init__(self, info: Any, interval: float = -1) -> None:
        self.info = info
        self.n_tasks = 0

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        try:
            yield
        finally:
            _write_evidence(
                {
                    "kind": "task",
                    "name": str(self.info),
                    "pid": os.getpid(),
                    "iid": id(self),
                    "n_tasks": self.n_tasks,
                }
            )

    @asynccontextmanager
    async def task_hook(self, input_item: Any = None) -> AsyncIterator[None]:
        self.n_tasks += 1
        yield


class _RecordingQueue(AsyncQueue):
    """An ``AsyncQueue`` whose ``stage_hook`` records that it ran in the (subprocess) stage."""

    def __init__(
        self, info: Any, *, buffer_size: int = 1, interval: float = -1
    ) -> None:
        super().__init__(info, buffer_size=buffer_size)
        self.n_get = 0

    async def get(self) -> object:
        item = await super().get()
        self.n_get += 1
        return item

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        try:
            yield
        finally:
            _write_evidence(
                {
                    "kind": "queue",
                    "name": str(self.info),
                    "pid": os.getpid(),
                    "iid": id(self),
                    "n_get": self.n_get,
                }
            )


def _install_recording_hooks(evidence_dir: str) -> None:
    """Executor initializer: runs inside each fused worker before its sub-pipeline is built.

    Fusion reads this off the ``ProcessPoolExecutor`` (``_pool_params``) and runs it in every
    worker process, so the worker's nested ``build_pipeline`` — which is given no explicit
    ``task_hook_factory``/``queue_class`` — picks up these recording classes as its defaults.
    """
    global _EVIDENCE_DIR
    _EVIDENCE_DIR = evidence_dir
    set_default_hook_class(_RecordingTaskHook)
    set_default_queue_class(_RecordingQueue)


class FuseHookTest(unittest.TestCase):
    """``TaskHook`` and the queue ``stage_hook`` fire inside the fused worker subprocess.

    The fused run executes as a nested pipeline inside main-process-owned worker processes; the
    per-stage hooks/stats fire there, not in the bridge stage. These tests install recording
    hook/queue classes as the worker defaults (via the pool's ``initializer``) and assert, from
    the evidence files those recorders leave behind, that both fired in a non-main process.
    """

    def _read_evidence(self, evidence_dir: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for name in os.listdir(evidence_dir):
            if not name.endswith(".json"):
                continue  # skip ``.json.tmp`` files still being written
            with open(os.path.join(evidence_dir, name)) as f:
                records.append(json.loads(f.read()))
        return records

    def _await_evidence(
        self, evidence_dir: str, n: int, timeout: float = 60.0
    ) -> list[dict[str, Any]]:
        """Re-read evidence until the workers' task total lands, then return the records.

        Each fused worker writes its evidence files to disk during nested-pipeline teardown, a
        side channel with no happens-before relationship to the main iterator completing. Polling
        until the recorded task total reaches the expected ``2 * n`` closes that read-too-early
        window without masking regressions: on a genuine failure (hooks never fire) the poll times
        out and the caller's assertions report the shortfall.
        """
        deadline = time.monotonic() + timeout
        records = self._read_evidence(evidence_dir)
        while sum(r["n_tasks"] for r in records if r["kind"] == "task") < 2 * n:
            if time.monotonic() > deadline:
                break  # fall through; the assertions below report the shortfall
            time.sleep(0.02)
            records = self._read_evidence(evidence_dir)
        return records

    def _assert_hooks_fired(self, records: list[dict[str, Any]], n: int) -> None:
        task_records = [r for r in records if r["kind"] == "task"]
        queue_records = [r for r in records if r["kind"] == "queue"]

        self.assertTrue(task_records, "TaskHook never fired in any fused worker")
        self.assertTrue(
            queue_records, "queue stage_hook never fired in any fused worker"
        )

        # Every record came from a worker process, never the main/test process — this is what
        # proves the hooks fired "in the subprocess".
        main_pid = os.getpid()
        for r in records:
            self.assertNotEqual(r["pid"], main_pid)

        # The two fused pipe stages each see every item once: ``add_one`` over n items, then
        # ``times_two`` over n items => 2n ``task_hook`` invocations, summed across workers.
        self.assertEqual(sum(r["n_tasks"] for r in task_records), 2 * n)

        # The fused sub-pipeline's queues carried the data inside the worker(s).
        self.assertGreaterEqual(sum(r["n_get"] for r in queue_records), n)

    def test_hooks_fire_in_fused_workers(self) -> None:
        """Hooks fire in the fused workers spawned by ``PipelineBuilder.build``."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        with tempfile.TemporaryDirectory() as evidence_dir:
            ex = ProcessPoolExecutor(
                max_workers=2,
                initializer=_install_recording_hooks,
                initargs=(evidence_dir,),
            )
            fused = (
                PipelineBuilder()
                .add_source(range(n))
                .pipe(add_one, executor=ex, concurrency=2)
                .pipe(times_two, executor=ex, concurrency=2)
                .add_sink(n)
                .build(num_threads=4, fuse_subprocess_stages=True)
            )
            self.assertEqual(sorted(_run(fused)), ref)
            self._assert_hooks_fired(self._await_evidence(evidence_dir, n), n)

    def test_hooks_fire_in_fused_workers_via_subprocess(self) -> None:
        """Hooks fire in the fused workers when the run is driven from a pipeline subprocess."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        with tempfile.TemporaryDirectory() as evidence_dir:
            ex = ProcessPoolExecutor(
                max_workers=2,
                initializer=_install_recording_hooks,
                initargs=(evidence_dir,),
            )
            config = (
                PipelineBuilder()
                .add_source(range(n))
                .pipe(add_one, executor=ex, concurrency=2)
                .pipe(times_two, executor=ex, concurrency=2)
                .add_sink(n)
                .get_config()
            )
            src = run_pipeline_in_subprocess(
                config, num_threads=4, fuse_subprocess_stages=True
            )
            # Each worker writes its evidence during nested-pipeline teardown, on a side channel
            # (the filesystem) with no happens-before relationship to iteration ending — so
            # ``_await_evidence`` polls for the files rather than assuming they are already there.
            self.assertEqual(sorted(src), ref)
            self._assert_hooks_fired(self._await_evidence(evidence_dir, n), n)


def _has_interpreter_pool_executor() -> bool:
    if sys.version_info < (3, 14):
        return False
    try:
        from concurrent.futures.interpreter import InterpreterPoolExecutor  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_INTERPRETER: bool = _has_interpreter_pool_executor()


if _HAS_INTERPRETER:

    class SubprocessPipelineInterpreterPoolFuseTest(unittest.TestCase):
        def test_interpreter_pool_stages_fused(self) -> None:
            """Stages sharing an InterpreterPoolExecutor are recognized and fused."""
            from concurrent.futures import InterpreterPoolExecutor  # pyre-ignore[21]

            n = 12
            ref = sorted((x + 1) * 2 for x in range(n))
            ex = InterpreterPoolExecutor(max_workers=2)
            fused = (
                PipelineBuilder()
                .add_source(range(n))
                .pipe(add_one, executor=ex, concurrency=2)
                .pipe(times_two, executor=ex, concurrency=2)
                .add_sink(n)
                .build(num_threads=4, fuse_subprocess_stages=True)
            )
            self.assertEqual(sorted(_run(fused)), ref)
