# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import multiprocessing as mp
import os
import sys
import threading
import time
import unittest
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from spdl.pipeline import (
    AsyncQueue,
    build_pipeline,
    Pipeline,
    PipelineBuilder,
    PipelineFailure,
    run_pipeline_in_subprocess,
)
from spdl.pipeline._fuse import _fuse_marked_regions
from spdl.pipeline.config import set_default_queue_class
from spdl.pipeline.defs import (
    Aggregate,
    InterpreterPoolExecutorConfig,
    MAIN_PROCESS,
    Pipe,
    PipelineConfig,
    PlacementConfig,
    ProcessPoolExecutorConfig,
    SinkConfig,
    SourceConfig,
)
from spdl.pipeline.defs._defs import _SubprocessPipelineConfig


def add_one(x: int) -> int:
    return x + 1


def times_two(x: int) -> int:
    return x * 2


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


class _PidStamp:
    """A picklable item stages stamp their ``os.getpid()`` into, to prove process-locality."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.pids: dict[str, int] = {}


def stamp_region(item: _PidStamp) -> _PidStamp:
    """A region stage: records the pid it runs in."""
    item.pids["region"] = os.getpid()
    return item


def stamp_main(item: _PidStamp) -> _PidStamp:
    """A main-process stage (after the region closes): records the pid it runs in."""
    item.pids["main"] = os.getpid()
    return item


class _ProcStamp:
    """Item recording ``(pid, ppid, daemon)`` of the stages that touch it, so a test can check
    which process a stage ran in and whether that process is a daemon."""

    def __init__(self, value: int, main_pid: int) -> None:
        self.value = value
        self.main_pid = main_pid
        self.intermediate: tuple[int, int, bool] | None = None
        self.region: tuple[int, int, bool] | None = None


def stamp_intermediate(item: _ProcStamp) -> _ProcStamp:
    """A stage OUTSIDE any region: under ``run_pipeline_in_subprocess`` it runs in the
    intermediate pipeline subprocess."""
    item.intermediate = (os.getpid(), os.getppid(), mp.current_process().daemon)
    return item


def stamp_region_proc(item: _ProcStamp) -> _ProcStamp:
    """A stage INSIDE a subprocess region: it runs in a region worker process."""
    item.region = (os.getpid(), os.getppid(), mp.current_process().daemon)
    return item


def _cfg(src: Any, pipes: Sequence[Any], buffer: int = 16) -> PipelineConfig[Any]:
    return PipelineConfig(
        src=SourceConfig(src), pipes=list(pipes), sink=SinkConfig(buffer)
    )


def _run(pipeline: Pipeline[Any], timeout: float = 60.0) -> list[Any]:
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=timeout))


class MarkedRegionFuseTest(unittest.TestCase):
    def test_region_of_two_pipes(self) -> None:
        """Two pipes inside a subprocess region produce the same result as running inline."""
        n = 16
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=2)),
                Pipe(add_one),
                Pipe(times_two),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=4)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    def test_aggregate_inside_region(self) -> None:
        """An aggregate stage inside a region is absorbed and runs in the worker."""
        config = _cfg(
            range(10),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),
                Aggregate(3),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        result = _run(pipeline)
        # add_one over 0..9, then batched by 3 (one worker preserves order).
        self.assertEqual(result, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]])

    def test_unpicklable_intermediate_stays_in_region(self) -> None:
        """An unpicklable value handed between two region stages never crosses a boundary."""
        n = 5
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(wrap),
                Pipe(unwrap),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    def test_region_runs_in_worker_main_runs_in_main(self) -> None:
        """A region stage runs in a worker process; a stage after MAIN_PROCESS runs in main."""
        config = _cfg(
            [_PidStamp(i) for i in range(4)],
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(stamp_region),
                PlacementConfig(target=MAIN_PROCESS),
                Pipe(stamp_main),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        results = _run(pipeline)
        self.assertEqual(len(results), 4)
        for item in results:
            self.assertNotEqual(item.pids["region"], os.getpid())
            self.assertEqual(item.pids["main"], os.getpid())

    def test_no_markers_is_unchanged(self) -> None:
        """A config with no markers builds and runs entirely in the main process."""
        n = 8
        config = _cfg(range(n), [Pipe(add_one), Pipe(times_two)])
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    def test_multiple_regions(self) -> None:
        """Two separate subprocess regions around a main-process stage both fuse correctly.

        Exercises the main -> subprocess -> main -> subprocess -> main transition cycle, so a
        bug carrying stale ``target``/``region`` state across ``_flush()`` calls would surface.
        """
        n = 8
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),  # region 1: x + 1
                PlacementConfig(target=MAIN_PROCESS),
                Pipe(times_two),  # main: (x + 1) * 2
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),  # region 2: (x + 1) * 2 + 1
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(
            sorted(_run(pipeline)), sorted((x + 1) * 2 + 1 for x in range(n))
        )

    def test_region_open_at_end_of_pipes(self) -> None:
        """A region left open at the end of the pipes is flushed (no closing MAIN_PROCESS).

        Covers the ``_flush()`` after the segmentation loop, which closes a region that runs to
        the end of ``pipes``; the sink still executes in the main process.
        """
        n = 8
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),
                Pipe(times_two),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))


# ---------------------------------------------------------------------------
# Behavioral coverage ported from the removed identity-fusion tests
# (subprocess_pipeline_fuse_test.py), now expressed via the public ``.to()`` API.
# These assert the guarantees the old ``fuse_subprocess_stages`` feature provided.
# ---------------------------------------------------------------------------


def boom(x: int) -> int:
    raise ValueError("boom")


def _raise_initializer() -> None:
    raise RuntimeError("initializer boom")


def dup(x: int) -> Iterator[int]:
    """A sync-generator op: yields two values per input (1->2 fan-out)."""
    yield x * 2
    yield x * 2 + 1


async def adup(x: int) -> AsyncIterator[int]:
    """An async-generator op: yields two values per input (1->2 fan-out)."""
    yield x
    yield x + 1


async def _astamp(item: _PidStamp) -> _PidStamp:
    """An async op that records its pid; runs on the worker's event loop inside the region."""
    item.pids["async"] = os.getpid()
    return item


def _stamp_branch(item: _PidStamp) -> _PidStamp:
    """Records its pid; used as a plain region stage and as a path-variants branch."""
    item.pids["branch"] = os.getpid()
    return item


def _stamp_merge(item: _PidStamp) -> _PidStamp:
    """A post-merge / fan-in region stage that records its pid."""
    item.pids["merge"] = os.getpid()
    return item


def _stamp_router(item: _PidStamp) -> int:
    """Path-variants router: records its pid, then routes by parity."""
    item.pids["router"] = os.getpid()
    return item.value % 2


def route_by_parity(x: int) -> int:
    """Router: even items to path 0, odd items to path 1."""
    return x % 2


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
    """Region-worker initializer: pin the worker sub-pipeline's queues to ``buffer_size=1``.

    A single-slot buffer stops the worker holding the earliest item from prefetching past it, so
    that worker stays parked on the slow item while its peer drains the rest -- surfacing the
    startup race where an early-finishing worker could otherwise steal a slow peer's end marker.
    """
    set_default_queue_class(_Buffer1Queue)


class RegionBehaviorTest(unittest.TestCase):
    """End-to-end guarantees the fusing feature provided, now via ``.to()`` regions."""

    def test_generator_op_in_region(self) -> None:
        """A sync-generator op inside a region fans out 1->N in the worker sub-pipeline."""
        n = 8
        # dup yields {2x, 2x+1}; add_one shifts each by one, covering 1..2n exactly.
        ref = list(range(1, 2 * n + 1))
        pipeline = (
            PipelineBuilder()
            .add_source(range(n))
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(dup, concurrency=2)
            .pipe(add_one, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_async_op_in_region_runs_in_worker(self) -> None:
        """An async op inside a region runs on the worker's loop, in the worker process.

        The sync stages on either side and the async op in the middle all stamp ``os.getpid()``;
        the assertion proves the async stage shares the worker process with its neighbours rather
        than hopping back to the main process.
        """
        n = 16
        pipeline = (
            PipelineBuilder()
            .add_source([_PidStamp(x) for x in range(n)])
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(_stamp_branch, concurrency=2)
            .pipe(_astamp, concurrency=2)
            .pipe(_stamp_merge, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        main_pid = os.getpid()
        results = _run(pipeline)
        self.assertEqual(len(results), n)
        for item in results:
            self.assertEqual(set(item.pids), {"branch", "async", "merge"})
            self.assertEqual(item.pids["branch"], item.pids["async"])
            self.assertEqual(item.pids["async"], item.pids["merge"])
            self.assertNotEqual(item.pids["async"], main_pid)

    def test_async_generator_op_in_region(self) -> None:
        """An async-generator op inside a region fans out inside the worker."""
        n = 8
        # add_one -> v=x+1; adup(v) yields {v, v+1}.
        ref = sorted(v for x in range(n) for v in (x + 1, x + 2))
        pipeline = (
            PipelineBuilder()
            .add_source(range(n))
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one, concurrency=2)
            .pipe(adup, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_initializer_failure_surfaces_not_hangs(self) -> None:
        """A region-worker initializer that raises surfaces as a failure instead of hanging.

        A real failure surfaces as :py:class:`PipelineFailure`; a hung pipeline would instead
        raise :py:class:`TimeoutError` from the finite ``get_iterator`` timeout, so asserting on
        ``PipelineFailure`` proves the failed initializer is reported rather than wedging forever.
        """
        pipeline = (
            PipelineBuilder()
            .add_source(range(2))
            .to(
                ProcessPoolExecutorConfig(max_workers=2, initializer=_raise_initializer)
            )
            .pipe(add_one, concurrency=2)
            .pipe(times_two, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(4)
            .build(num_threads=4)
        )
        with self.assertRaises(PipelineFailure):
            _run(pipeline, timeout=30.0)

    def test_slow_worker_does_not_drop_earliest_items(self) -> None:
        """A worker stalled on the earliest item still delivers it instead of dropping it."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        pipeline = (
            PipelineBuilder()
            .add_source(range(n))
            .to(
                ProcessPoolExecutorConfig(
                    max_workers=2, initializer=_install_small_buffers
                )
            )
            .pipe(stall_on_first, concurrency=1)
            .pipe(times_two, concurrency=1)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_lone_path_variants_in_region(self) -> None:
        """A path-variants stage inside a region routes and runs entirely in the worker.

        Even items take the ``add_one`` branch, odd items the ``times_two`` branch; the whole
        routing construct moves into the worker, so the result must match an inline run.
        """
        n = 16
        ref = sorted(x + 1 if x % 2 == 0 else x * 2 for x in range(n))
        pipeline = (
            PipelineBuilder()
            .add_source(range(n))
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .path_variants(route_by_parity, [[Pipe(add_one)], [Pipe(times_two)]])
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), ref)

    def test_path_variants_process_locality_in_region(self) -> None:
        """Router, branches, and the post-merge stage all run in one worker process.

        Each stage stamps ``os.getpid()``; the assertion proves the whole
        router -> branch -> merge chain runs inside a single worker process per item (distinct
        from the main process), i.e. fusion leaves no per-stage process hop.
        """
        n = 16
        pipeline = (
            PipelineBuilder()
            .add_source([_PidStamp(x) for x in range(n)])
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .path_variants(
                _stamp_router, [[Pipe(_stamp_branch)], [Pipe(_stamp_branch)]]
            )
            .pipe(_stamp_merge)  # fan-in / post-merge, same region
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        results = _run(pipeline)
        self.assertEqual(len(results), n)
        main_pid = os.getpid()
        worker_pids = set()
        for item in results:
            self.assertEqual(set(item.pids), {"router", "branch", "merge"})
            self.assertEqual(item.pids["router"], item.pids["branch"])
            self.assertEqual(item.pids["branch"], item.pids["merge"])
            self.assertNotEqual(item.pids["router"], main_pid)
            worker_pids.add(item.pids["router"])
        self.assertNotIn(main_pid, worker_pids)

    def test_run_pipeline_in_subprocess_with_region(self) -> None:
        """A ``.to()`` region composes with ``run_pipeline_in_subprocess``."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        config = (
            PipelineBuilder()
            .add_source(range(n))
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one, concurrency=2)
            .pipe(times_two, concurrency=3)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(config, num_threads=4)
        self.assertEqual(sorted(src), ref)

    def test_run_pipeline_in_subprocess_unpicklable_intermediate(self) -> None:
        """The region handle survives into the pipeline subprocess and the unpicklable
        op->op handoff stays inside a worker."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        config = (
            PipelineBuilder()
            .add_source(range(n))
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(wrap, concurrency=2)
            .pipe(unwrap, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(config, num_threads=4)
        self.assertEqual(sorted(src), ref)


class ContinuousRegionTest(unittest.TestCase):
    """A ``.to()`` region under a continuous source stays warm and correct across epochs."""

    def test_multi_epoch_correct(self) -> None:
        """A continuous region yields the correct set each epoch from the same warm pool."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one, concurrency=2)
            .pipe(times_two, concurrency=3)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        with pipeline.auto_stop():
            for _ in range(3):  # three epochs from the same warm worker pool
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_fewer_items_than_workers(self) -> None:
        """An epoch with fewer items than workers still completes (some workers run empty)."""
        n = 2
        ref = sorted((x + 1) * 2 for x in range(n))
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .to(ProcessPoolExecutorConfig(max_workers=4))
            .pipe(add_one, concurrency=2)
            .pipe(times_two, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_op_failure_does_not_deadlock(self) -> None:
        """Op failures (dropped per SPDL default) still let each epoch's barrier complete.

        ``boom`` raises on every item, so the region produces no results; the continuous epoch
        barrier must still complete each epoch (workers report the boundary with zero results)
        instead of deadlocking.
        """
        n = 8
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one, concurrency=2)
            .pipe(boom, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(list(pipeline.get_iterator(timeout=60)), [])

    def test_teardown_mid_stream_does_not_hang(self) -> None:
        """Tearing a continuous region subprocess pipeline down mid-stream must not hang.

        A teardown before the stream is drained leaves the per-worker input queue full; pool
        shutdown must cancel the queue feeder-thread join instead of blocking forever flushing
        buffered items into a pipe the terminated workers no longer drain.
        """
        n = 100_000  # far more than any buffer, so the stream is still full at teardown
        config = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one, concurrency=2)
            .pipe(times_two, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(2)
            .get_config()
        )
        src = run_pipeline_in_subprocess(config, num_threads=4)
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
            "region-pool teardown hung on a full input queue after a mid-stream stop",
        )
        t.join(timeout=10)

    def test_continuous_in_subprocess(self) -> None:
        """A continuous region composes with ``run_pipeline_in_subprocess`` across epochs."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        config = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one, concurrency=2)
            .pipe(times_two, concurrency=2)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(config, num_threads=4)
        for _ in range(3):  # one epoch per iteration
            self.assertEqual(sorted(src), ref)


class MarkedRegionSegmentationTest(unittest.TestCase):
    """Unit tests for the ``_fuse_marked_regions`` config rewrite.

    Replaces the pure-detection coverage of the removed ``_find_fusable_runs`` tests. Because the
    rewrite eagerly spawns worker pools, each fused case reaps its pools in a ``finally``.
    """

    def test_no_markers_returns_config_unchanged(self) -> None:
        """With no region markers the rewrite is a no-op: same config, no pools."""
        config = _cfg(range(4), [Pipe(add_one), Pipe(times_two)])
        new_config, pools = _fuse_marked_regions(config)
        self.assertEqual(pools, [])
        self.assertIs(new_config, config)

    def test_region_becomes_single_fused_stage_absorbing_aggregate(self) -> None:
        """A region collapses to one fused stage (absorbing aggregate); main stages are kept."""
        config = _cfg(
            range(4),
            [
                Pipe(add_one),  # main process, before the region
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(times_two),
                Aggregate(2),
                PlacementConfig(target=MAIN_PROCESS),
                Pipe(add_one),  # main process, after the region
            ],
        )
        new_config, pools = _fuse_marked_regions(config)
        try:
            kinds = [type(p).__name__ for p in new_config.pipes]
            self.assertEqual(
                kinds, ["PipeConfig", "_SubprocessPipelineConfig", "PipeConfig"]
            )
            fused = [
                p for p in new_config.pipes if isinstance(p, _SubprocessPipelineConfig)
            ]
            self.assertEqual(len(fused), 1)
        finally:
            for pool in pools:
                pool.shutdown()

    def test_two_regions_produce_two_fused_stages(self) -> None:
        """Two separate regions produce two independent fused stages."""
        config = _cfg(
            range(4),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),
                PlacementConfig(target=MAIN_PROCESS),
                Pipe(times_two),
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        new_config, pools = _fuse_marked_regions(config)
        try:
            fused = [
                p for p in new_config.pipes if isinstance(p, _SubprocessPipelineConfig)
            ]
            self.assertEqual(len(fused), 2)
        finally:
            for pool in pools:
                pool.shutdown()


if sys.version_info >= (3, 14):

    class SubinterpreterRegionFuseTest(unittest.TestCase):
        """Region fusion targeting subinterpreter workers (Python 3.14+)."""

        def test_region_of_two_pipes(self) -> None:
            """Two pipes in a subinterpreter region produce the same result as inline."""
            n = 16
            config = _cfg(
                range(n),
                [
                    PlacementConfig(
                        target=InterpreterPoolExecutorConfig(max_workers=2)
                    ),
                    Pipe(add_one),
                    Pipe(times_two),
                    PlacementConfig(target=MAIN_PROCESS),
                ],
            )
            pipeline = build_pipeline(config, num_threads=4)
            self.assertEqual(
                sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n))
            )

        def test_aggregate_inside_region(self) -> None:
            """An aggregate stage inside a subinterpreter region is absorbed and runs there."""
            config = _cfg(
                range(10),
                [
                    PlacementConfig(
                        target=InterpreterPoolExecutorConfig(max_workers=1)
                    ),
                    Pipe(add_one),
                    Aggregate(3),
                    PlacementConfig(target=MAIN_PROCESS),
                ],
            )
            pipeline = build_pipeline(config, num_threads=2)
            self.assertEqual(_run(pipeline), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]])

        def test_unpicklable_intermediate_stays_in_region(self) -> None:
            """An unpicklable value between two subinterpreter stages never crosses out."""
            n = 5
            config = _cfg(
                range(n),
                [
                    PlacementConfig(
                        target=InterpreterPoolExecutorConfig(max_workers=1)
                    ),
                    Pipe(wrap),
                    Pipe(unwrap),
                    PlacementConfig(target=MAIN_PROCESS),
                ],
            )
            pipeline = build_pipeline(config, num_threads=2)
            self.assertEqual(
                sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n))
            )

else:

    class SubinterpreterRegionRequiresPy314Test(unittest.TestCase):
        """On Python < 3.14, a subinterpreter region is rejected with a clear error."""

        def test_raises_runtime_error(self) -> None:
            """Building a subinterpreter region on an older interpreter raises RuntimeError."""
            config = _cfg(
                range(4),
                [
                    PlacementConfig(
                        target=InterpreterPoolExecutorConfig(max_workers=1)
                    ),
                    Pipe(add_one),
                    PlacementConfig(target=MAIN_PROCESS),
                ],
            )
            with self.assertRaises(RuntimeError) as cm:
                build_pipeline(config, num_threads=2)
            self.assertIn("3.14", str(cm.exception))


class RegionUnderSubprocessTest(unittest.TestCase):
    """A ``.to()`` region driven by :py:func:`run_pipeline_in_subprocess`.

    ``run_pipeline_in_subprocess`` runs the source/sink and any non-region stages in an
    *intermediate* subprocess, while each ``.to(ProcessPoolExecutorConfig(...))`` region is fused in
    the **main** process (via ``_fuse_marked_regions``) so its worker pool is main-owned --
    spawned by main, not by the intermediate subprocess (which, as a daemon, cannot have
    children). These tests pin that placement and the daemon flags that let teardown reap
    everything at exit.
    """

    def test_region_pool_is_main_owned_and_daemon(self) -> None:
        """The region worker pool is spawned from main (hoisted), and both the intermediate
        subprocess and the region workers are daemons.

        A stage outside the region records the intermediate subprocess's
        ``(pid, ppid, daemon)``; a stage inside the region records the region worker's. The
        region worker's parent must be the main process (not the intermediate subprocess),
        and every spawned process must be a daemon so it is terminated at interpreter exit
        rather than hang-joined.
        """
        main_pid = os.getpid()
        n = 4
        config = _cfg(
            [_ProcStamp(i, main_pid) for i in range(n)],
            [
                Pipe(
                    stamp_intermediate
                ),  # outside the region -> intermediate subprocess
                PlacementConfig(
                    target=ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn")
                ),
                Pipe(stamp_region_proc),  # inside the region -> region worker
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        src = run_pipeline_in_subprocess(
            config,
            num_threads=2,
            use_thread_output_queue=True,
            buffer_size=8,
            timeout=60.0,
            mp_context="spawn",
            daemon=True,
        )
        results = list(src)

        self.assertEqual(len(results), n)
        for item in results:
            self.assertIsNotNone(item.intermediate)
            self.assertIsNotNone(item.region)
            inter_pid, inter_ppid, inter_daemon = item.intermediate
            region_pid, region_ppid, region_daemon = item.region
            # The intermediate subprocess is a daemon child of main.
            self.assertEqual(inter_ppid, main_pid)
            self.assertTrue(inter_daemon)
            # The region worker pool is spawned from MAIN (hoisted), not the intermediate
            # subprocess -- its parent is main, and distinct from the intermediate.
            self.assertEqual(region_ppid, main_pid)
            self.assertNotEqual(region_ppid, inter_pid)
            # The region workers are daemons.
            self.assertTrue(region_daemon)
