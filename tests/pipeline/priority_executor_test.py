# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import gc
import itertools
import pickle
import sys
import threading
import time
import unittest
import warnings
import weakref
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from queue import Empty
from typing import Type, TypeVar

from spdl.pipeline import (
    PipelineBuilder,
    PipelineFailure,
    PriorityExecutorEntrypoint,
    PriorityProcessPoolExecutor,
    PriorityThreadPoolExecutor,
)
from spdl.pipeline._priority_executor import _OWNER_REGISTRY, _PriorityQueueAdapter

_F = TypeVar("_F", bound=Callable[..., object])
_C = TypeVar("_C", bound=Type[object])


def _ignore_fork_warning(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"This process \(pid=\d+\) is multi-threaded, use of "
                    r"fork\(\) may lead to deadlocks in the child"
                ),
                category=DeprecationWarning,
            )
            return fn(*args, **kwargs)

    # pyre-ignore[7]
    return wrapper


def _ignore_fork_warning_in_class(cls: _C) -> _C:
    for name, member in list(vars(cls).items()):
        if name.startswith("test_") and callable(member):
            setattr(cls, name, _ignore_fork_warning(member))
    return cls


def _raise_value_error() -> None:
    raise ValueError("process boom")


# ─── _PriorityQueueAdapter unit tests ───


class TestPriorityQueueAdapter(unittest.TestCase):
    def test_priority_ordering(self) -> None:
        q = _PriorityQueueAdapter()
        q.put("low")
        q.put("high")
        q.put("mid")
        # All inserted without priority context → same default → FIFO
        self.assertEqual(q.get(), "low")
        self.assertEqual(q.get(), "high")
        self.assertEqual(q.get(), "mid")

    def test_priority_ordering_with_explicit_priority(self) -> None:
        q = _PriorityQueueAdapter()

        q.put("low", priority=(10, 0))
        q.put("high", priority=(0, 0))
        q.put("mid", priority=(5, 0))

        self.assertEqual(q.get(), "high")
        self.assertEqual(q.get(), "mid")
        self.assertEqual(q.get(), "low")

    def test_fifo_within_same_priority(self) -> None:
        q = _PriorityQueueAdapter()
        for i in range(5):
            q.put(f"item_{i}", priority=(1, i))

        for i in range(5):
            self.assertEqual(q.get(), f"item_{i}")

    def test_sentinel_none_processed_first(self) -> None:
        q = _PriorityQueueAdapter()
        q.put("work", priority=(0, 0))
        q.put(None)  # shutdown sentinel, no priority

        self.assertIsNone(q.get())
        self.assertEqual(q.get(), "work")

    def test_get_nowait_empty_raises(self) -> None:
        q = _PriorityQueueAdapter()
        with self.assertRaises(Empty):
            q.get_nowait()

    def test_empty_and_qsize(self) -> None:
        q = _PriorityQueueAdapter()
        self.assertTrue(q.empty())
        self.assertEqual(q.qsize(), 0)
        q.put("x")
        self.assertFalse(q.empty())
        self.assertEqual(q.qsize(), 1)


# ─── PriorityExecutorEntrypoint unit tests ───


class TestPriorityExecutorEntrypoint(unittest.TestCase):
    def test_submit_returns_future(self) -> None:
        executor = PriorityThreadPoolExecutor(max_workers=1)
        stage = executor.get_executor()
        fut = stage.submit(lambda: 42)
        self.assertIsInstance(fut, Future)
        self.assertEqual(fut.result(timeout=5), 42)
        executor.shutdown()

    def test_is_executor_compatible(self) -> None:
        executor = PriorityThreadPoolExecutor(max_workers=1)
        stage = executor.get_executor()
        self.assertIsInstance(stage, Executor)
        executor.shutdown()

    def test_shutdown_is_noop(self) -> None:
        executor = PriorityThreadPoolExecutor(max_workers=1)
        stage = executor.get_executor()
        stage.shutdown()  # should not affect the pool
        fut = stage.submit(lambda: 1)
        self.assertEqual(fut.result(timeout=5), 1)
        executor.shutdown()


# ─── PriorityThreadPoolExecutor ordering tests ───


class TestPriorityThreadPoolExecutorOrdering(unittest.TestCase):
    def test_downstream_stage_runs_first(self) -> None:
        """With 1 worker, tasks are executed in priority order."""
        barrier = threading.Barrier(2)
        results: list[str] = []

        executor = PriorityThreadPoolExecutor(max_workers=1)
        upstream = executor.get_executor(priority=0)
        downstream = executor.get_executor(priority=2)

        # Block the single worker so we can enqueue both tasks
        executor.get_executor().submit(lambda: barrier.wait(timeout=5))

        # Enqueue upstream first, then downstream
        upstream.submit(lambda: results.append("upstream"))
        downstream.submit(lambda: results.append("downstream"))

        # Release the worker
        barrier.wait(timeout=5)
        executor.shutdown(wait=True)

        self.assertEqual(results, ["downstream", "upstream"])

    def test_fifo_within_stage(self) -> None:
        barrier = threading.Barrier(2)
        results: list[int] = []

        executor = PriorityThreadPoolExecutor(max_workers=1)
        stage = executor.get_executor()

        executor.get_executor().submit(lambda: barrier.wait(timeout=5))

        for i in range(5):
            stage.submit(lambda i=i: results.append(i))

        barrier.wait(timeout=5)
        executor.shutdown(wait=True)

        self.assertEqual(results, [0, 1, 2, 3, 4])

    def test_multiple_stages_interleaved(self) -> None:
        """3 stages, items interleaved — should sort by priority then FIFO."""
        barrier = threading.Barrier(2)
        results: list[tuple[int, int]] = []

        executor = PriorityThreadPoolExecutor(max_workers=1)
        stages = [executor.get_executor(priority=p) for p in [0, 1, 2]]

        executor.get_executor().submit(lambda: barrier.wait(timeout=5))

        # Submit: stage0, stage1, stage2, stage0, stage1, stage2
        for round_idx in range(2):
            for idx, stage in enumerate(stages):
                stage.submit(lambda i=idx, ri=round_idx: results.append((i, ri)))

        barrier.wait(timeout=5)
        executor.shutdown(wait=True)

        # priority 2 (highest) first, then priority 1, then priority 0
        # Within same priority, FIFO by round
        self.assertEqual(
            results,
            [(2, 0), (2, 1), (1, 0), (1, 1), (0, 0), (0, 1)],
        )

    def test_basic_execution(self) -> None:
        executor = PriorityThreadPoolExecutor(max_workers=2)
        stage = executor.get_executor()
        futs = [stage.submit(lambda x=x: x * 2, x) for x in range(10)]
        results = {f.result(timeout=5) for f in futs}
        self.assertEqual(results, {x * 2 for x in range(10)})
        executor.shutdown()

    def test_exception_propagation(self) -> None:
        executor = PriorityThreadPoolExecutor(max_workers=1)
        stage = executor.get_executor()

        def fail() -> None:
            raise ValueError("boom")

        fut = stage.submit(fail)
        with self.assertRaises(ValueError):
            fut.result(timeout=5)
        executor.shutdown()

    def test_submit_after_shutdown_raises(self) -> None:
        executor = PriorityThreadPoolExecutor(max_workers=1)
        stage = executor.get_executor()
        executor.shutdown()
        with self.assertRaises(RuntimeError):
            executor._submit_with_priority((0, 0), lambda: None, (), {})

    def test_multiple_workers_all_complete(self) -> None:
        """With multiple workers, all tasks must complete."""
        executor = PriorityThreadPoolExecutor(max_workers=4)
        stages = [executor.get_executor() for _ in range(3)]

        counter: itertools.count[int] = itertools.count()
        results: list[int] = []
        lock: threading.Lock = threading.Lock()

        def work() -> None:
            val = next(counter)
            with lock:
                results.append(val)

        futs = []
        for stage in stages:
            for _ in range(10):
                futs.append(stage.submit(work))

        for f in futs:
            f.result(timeout=10)

        executor.shutdown()
        self.assertEqual(len(results), 30)


# ─── PriorityProcessPoolExecutor tests ───


@_ignore_fork_warning_in_class
class TestPriorityProcessPoolExecutor(unittest.TestCase):
    def test_basic_execution(self) -> None:
        executor = PriorityProcessPoolExecutor(max_workers=2)
        stage = executor.get_executor()
        futs = [stage.submit(pow, 2, x) for x in range(10)]
        results = {f.result(timeout=10) for f in futs}
        self.assertEqual(results, {2**x for x in range(10)})
        executor.shutdown()

    def test_exception_propagation(self) -> None:
        executor = PriorityProcessPoolExecutor(max_workers=1)
        stage = executor.get_executor()

        fut = stage.submit(_raise_value_error)
        with self.assertRaises(ValueError):
            fut.result(timeout=10)
        executor.shutdown()


# ─── Pipeline integration: correctness ───


class TestPriorityExecutorPipelineCorrectness(unittest.TestCase):
    def test_two_stage_pipeline(self) -> None:
        """All items flow through correctly with a shared priority executor."""
        pool = PriorityThreadPoolExecutor(max_workers=4)
        s1 = pool.get_executor()
        s2 = pool.get_executor()

        pipeline = (
            PipelineBuilder()
            .add_source(range(20))
            .pipe(lambda x: x * 2, executor=s1, concurrency=4)
            .pipe(lambda x: x + 1, executor=s2, concurrency=4)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, sorted(x * 2 + 1 for x in range(20)))
        pool.shutdown()

    def test_three_stage_pipeline(self) -> None:
        pool = PriorityThreadPoolExecutor(max_workers=4)
        s1 = pool.get_executor()
        s2 = pool.get_executor()
        s3 = pool.get_executor()

        pipeline = (
            PipelineBuilder()
            .add_source(range(15))
            .pipe(lambda x: x + 1, executor=s1, concurrency=2)
            .pipe(lambda x: x * 3, executor=s2, concurrency=2)
            .pipe(lambda x: x - 1, executor=s3, concurrency=2)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, sorted((x + 1) * 3 - 1 for x in range(15)))
        pool.shutdown()

    def test_mixed_priority_and_default_executor(self) -> None:
        """Some stages use priority executor, others use default."""
        pool = PriorityThreadPoolExecutor(max_workers=2)
        s1 = pool.get_executor()

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(lambda x: x * 2, executor=s1, concurrency=2)
            .pipe(lambda x: x + 1, concurrency=2)
            .add_sink(3)
            .build(num_threads=2)
        )

        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, sorted(x * 2 + 1 for x in range(10)))
        pool.shutdown()

    def test_exception_propagates_through_pipeline(self) -> None:
        pool = PriorityThreadPoolExecutor(max_workers=2)
        s1 = pool.get_executor()
        s2 = pool.get_executor()

        def fail_on_five(x: int) -> int:
            if x == 5:
                raise ValueError("boom on 5")
            return x

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(fail_on_five, executor=s1, concurrency=1, max_failures=0)
            .pipe(lambda x: x, executor=s2, concurrency=1)
            .add_sink(3)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        pool.shutdown()


# ─── Pipeline integration: priority ordering ───


class TestPriorityExecutorPipelineOrdering(unittest.TestCase):
    def test_downstream_not_starved(self) -> None:
        """With 1 shared worker, downstream should interleave with upstream,
        not wait until all upstream completes."""
        execution_log: list[tuple[str, int]] = []
        lock: threading.Lock = threading.Lock()

        pool = PriorityThreadPoolExecutor(max_workers=1)
        upstream_exec = pool.get_executor()
        downstream_exec = pool.get_executor()

        def upstream_op(item: int) -> int:
            with lock:
                execution_log.append(("up", item))
            # Sleep so the event loop has time to submit the downstream task
            # before the worker picks the next item.
            time.sleep(0.03)
            return item

        def downstream_op(item: int) -> int:
            with lock:
                execution_log.append(("down", item))
            return item

        pipeline = (
            PipelineBuilder()
            .add_source(range(8))
            .pipe(upstream_op, executor=upstream_exec, concurrency=1)
            .pipe(downstream_op, executor=downstream_exec, concurrency=1)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(len(results), 8)
        pool.shutdown()

        # With priority: up/down alternate → max consecutive upstream ≤ 2.
        # Without priority (FIFO): upstream dominates → all upstream first.
        max_consec_up = 0
        consec = 0
        for stage, _ in execution_log:
            if stage == "up":
                consec += 1
                max_consec_up = max(max_consec_up, consec)
            else:
                consec = 0

        self.assertLessEqual(
            max_consec_up,
            2,
            f"Upstream ran {max_consec_up} consecutive times — "
            f"downstream was starved. Full log: {execution_log}",
        )

    def test_three_stage_priority_order(self) -> None:
        """With 3 stages sharing 1 worker, the most downstream stage
        with pending work should run first."""
        execution_log: list[tuple[str, int]] = []
        lock: threading.Lock = threading.Lock()

        pool = PriorityThreadPoolExecutor(max_workers=1)
        s1_exec = pool.get_executor()
        s2_exec = pool.get_executor()
        s3_exec = pool.get_executor()

        def make_op(name: str):  # pyre-ignore[3]
            def op(item: int) -> int:
                with lock:
                    execution_log.append((name, item))
                time.sleep(0.03)
                return item

            return op

        pipeline = (
            PipelineBuilder()
            .add_source(range(6))
            .pipe(make_op("s1"), executor=s1_exec, concurrency=1)
            .pipe(make_op("s2"), executor=s2_exec, concurrency=1)
            .pipe(make_op("s3"), executor=s3_exec, concurrency=1)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(len(results), 6)
        pool.shutdown()

        # After downstream stages have pending work, upstream should not
        # run consecutively.
        stages = [stage for stage, _ in execution_log]
        for i in range(len(stages) - 1):
            if stages[i] == "s1" and stages[i + 1] == "s1":
                prior = stages[:i]
                self.assertNotIn(
                    "s2",
                    prior,
                    f"Consecutive s1 at positions {i},{i + 1} after s2 "
                    f"already had work. Log: {execution_log}",
                )

    def test_priority_vs_fifo_comparison(self) -> None:
        """Priority executor interleaves downstream at least as well as FIFO."""

        def run_pipeline(
            upstream_exec: Executor,
            downstream_exec: Executor,
            num_threads: int,
        ) -> list[str]:
            log: list[str] = []
            lock: threading.Lock = threading.Lock()

            def up_op(item: int) -> int:
                with lock:
                    log.append("up")
                time.sleep(0.03)
                return item

            def down_op(item: int) -> int:
                with lock:
                    log.append("down")
                return item

            pipeline = (
                PipelineBuilder()
                .add_source(range(8))
                .pipe(up_op, executor=upstream_exec, concurrency=1)
                .pipe(down_op, executor=downstream_exec, concurrency=1)
                .add_sink(3)
                .build(num_threads=num_threads)
            )

            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))
            return log

        # Priority executor
        pool = PriorityThreadPoolExecutor(max_workers=1)
        priority_log = run_pipeline(
            pool.get_executor(),
            pool.get_executor(),
            num_threads=1,
        )
        pool.shutdown()

        # Plain FIFO executor
        plain = ThreadPoolExecutor(max_workers=1)
        fifo_log = run_pipeline(plain, plain, num_threads=1)
        plain.shutdown()

        # With priority, downstream entries should appear at least as early.
        halfway = len(priority_log) // 2
        priority_down_first_half = priority_log[:halfway].count("down")
        fifo_down_first_half = fifo_log[:halfway].count("down")

        self.assertGreaterEqual(
            priority_down_first_half,
            fifo_down_first_half,
            f"Priority executor should interleave downstream tasks at least "
            f"as well as FIFO.\n"
            f"Priority log: {priority_log}\n"
            f"FIFO log: {fifo_log}",
        )


# ─── Drop-in compatibility ───


class TestDropInCompatibility(unittest.TestCase):
    def test_stage_executor_with_as_completed(self) -> None:
        from concurrent.futures import as_completed

        pool = PriorityThreadPoolExecutor(max_workers=2)
        stage = pool.get_executor()
        futs = [stage.submit(pow, 2, i) for i in range(5)]
        results = set()
        for f in as_completed(futs, timeout=5):
            results.add(f.result())
        self.assertEqual(results, {1, 2, 4, 8, 16})
        pool.shutdown()

    def test_context_manager(self) -> None:
        with PriorityThreadPoolExecutor(max_workers=2) as executor:
            stage = executor.get_executor()
            self.assertEqual(stage.submit(lambda: 99).result(timeout=5), 99)


# ─── Mixed priority + regular ThreadPoolExecutor pipelines ───


class TestMixedExecutorPipeline(unittest.TestCase):
    def test_priority_upstream_regular_downstream(self) -> None:
        """Priority executor on upstream stages, regular ThreadPoolExecutor
        on downstream stage."""
        pool = PriorityThreadPoolExecutor(max_workers=2)
        regular = ThreadPoolExecutor(max_workers=2)

        pipeline = (
            PipelineBuilder()
            .add_source(range(20))
            .pipe(lambda x: x * 2, executor=pool.get_executor(), concurrency=2)
            .pipe(lambda x: x + 1, executor=regular, concurrency=2)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, sorted(x * 2 + 1 for x in range(20)))
        pool.shutdown()
        regular.shutdown()

    def test_regular_upstream_priority_downstream(self) -> None:
        """Regular ThreadPoolExecutor on upstream, priority executor on
        downstream."""
        pool = PriorityThreadPoolExecutor(max_workers=2)
        regular = ThreadPoolExecutor(max_workers=2)

        pipeline = (
            PipelineBuilder()
            .add_source(range(20))
            .pipe(lambda x: x + 10, executor=regular, concurrency=2)
            .pipe(lambda x: x * 3, executor=pool.get_executor(), concurrency=2)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, sorted((x + 10) * 3 for x in range(20)))
        pool.shutdown()
        regular.shutdown()

    def test_three_stage_alternating_executors(self) -> None:
        """Three stages: priority, regular, priority — verifying they
        compose correctly."""
        pool = PriorityThreadPoolExecutor(max_workers=3)
        regular = ThreadPoolExecutor(max_workers=2)

        pipeline = (
            PipelineBuilder()
            .add_source(range(15))
            .pipe(lambda x: x + 1, executor=pool.get_executor(), concurrency=2)
            .pipe(lambda x: x * 2, executor=regular, concurrency=2)
            .pipe(lambda x: x - 1, executor=pool.get_executor(), concurrency=2)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, sorted((x + 1) * 2 - 1 for x in range(15)))
        pool.shutdown()
        regular.shutdown()

    def test_two_priority_pools_and_regular(self) -> None:
        """Two independent priority pools plus a regular pool, all in one
        pipeline."""
        pool_a = PriorityThreadPoolExecutor(max_workers=2)
        pool_b = PriorityThreadPoolExecutor(max_workers=2)
        regular = ThreadPoolExecutor(max_workers=2)

        pipeline = (
            PipelineBuilder()
            .add_source(range(12))
            .pipe(lambda x: x + 1, executor=pool_a.get_executor(), concurrency=2)
            .pipe(lambda x: x * 2, executor=regular, concurrency=2)
            .pipe(lambda x: x - 1, executor=pool_b.get_executor(), concurrency=2)
            .add_sink(3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, sorted((x + 1) * 2 - 1 for x in range(12)))
        pool_a.shutdown()
        pool_b.shutdown()
        regular.shutdown()

    def test_exception_in_regular_stage(self) -> None:
        """Exception in the regular executor stage propagates correctly."""
        pool = PriorityThreadPoolExecutor(max_workers=2)
        regular = ThreadPoolExecutor(max_workers=2)

        def fail_on_five(x: int) -> int:
            if x == 5:
                raise ValueError("mixed boom")
            return x

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(lambda x: x, executor=pool.get_executor(), concurrency=2)
            .pipe(fail_on_five, executor=regular, concurrency=1, max_failures=0)
            .add_sink(3)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        pool.shutdown()
        regular.shutdown()


# ─── Pickle support ───


class TestPriorityExecutorPickle(unittest.TestCase):
    def test_thread_executor_round_trip(self) -> None:
        pool = PriorityThreadPoolExecutor(max_workers=4)
        data = pickle.dumps(pool)
        pool.shutdown()

        restored = pickle.loads(data)
        stage = restored.get_executor()
        fut = stage.submit(pow, 2, 10)
        self.assertEqual(fut.result(timeout=5), 1024)
        restored.shutdown()

    def test_entrypoint_round_trip(self) -> None:
        pool = PriorityThreadPoolExecutor(max_workers=2)
        ep = pool.get_executor(priority=5)
        data = pickle.dumps(ep)
        pool.shutdown()

        restored_ep = pickle.loads(data)
        fut = restored_ep.submit(pow, 2, 3)
        self.assertEqual(fut.result(timeout=5), 8)
        restored_ep._owner.shutdown()

    def test_multiple_entrypoints_share_master(self) -> None:
        pool = PriorityThreadPoolExecutor(max_workers=2)
        ep1 = pool.get_executor(priority=1)
        ep2 = pool.get_executor(priority=2)

        data1 = pickle.dumps(ep1)
        data2 = pickle.dumps(ep2)

        # Remove original pool from registry to simulate new process
        pool_id = pool._id
        pool.shutdown()
        _OWNER_REGISTRY.pop(pool_id, None)

        restored1 = pickle.loads(data1)
        restored2 = pickle.loads(data2)

        self.assertIs(restored1._owner, restored2._owner)

        fut1 = restored1.submit(pow, 2, 3)
        fut2 = restored2.submit(pow, 3, 2)
        self.assertEqual(fut1.result(timeout=5), 8)
        self.assertEqual(fut2.result(timeout=5), 9)
        restored1._owner.shutdown()

    @_ignore_fork_warning
    def test_process_executor_round_trip(self) -> None:
        pool = PriorityProcessPoolExecutor(max_workers=2)
        data = pickle.dumps(pool)
        pool.shutdown()

        restored = pickle.loads(data)
        stage = restored.get_executor()
        fut = stage.submit(pow, 2, 10)
        self.assertEqual(fut.result(timeout=10), 1024)
        restored.shutdown()

    def test_entrypoint_is_executor_subclass_after_unpickle(self) -> None:
        pool = PriorityThreadPoolExecutor(max_workers=1)
        ep = pool.get_executor()
        data = pickle.dumps(ep)
        pool.shutdown()

        restored = pickle.loads(data)
        self.assertIsInstance(restored, Executor)
        self.assertIsInstance(restored, PriorityExecutorEntrypoint)
        restored._owner.shutdown()

    def test_entrypoint_priority_preserved(self) -> None:
        """Priority ordering is preserved across pickle round-trip."""
        pool = PriorityThreadPoolExecutor(max_workers=1)
        upstream = pool.get_executor(priority=0)
        downstream = pool.get_executor(priority=2)

        data_up = pickle.dumps(upstream)
        data_down = pickle.dumps(downstream)
        pool.shutdown()
        _OWNER_REGISTRY.pop(pool._id, None)

        restored_up = pickle.loads(data_down)
        restored_down = pickle.loads(data_up)

        # Submit and wait for results
        fut_up = restored_up.submit(pow, 2, 3)
        fut_down = restored_down.submit(pow, 3, 2)
        self.assertEqual(fut_up.result(timeout=5), 8)
        self.assertEqual(fut_down.result(timeout=5), 9)

        # Verify both share the same owner
        self.assertIs(restored_up._owner, restored_down._owner)
        # Verify priority values were preserved
        self.assertEqual(restored_up._priority, -2)
        self.assertEqual(restored_down._priority, 0)
        restored_up._owner.shutdown()


# ─── Garbage collection ───


class TestPriorityExecutorGarbageCollection(unittest.TestCase):
    def test_owner_gc_after_all_references_dropped(self) -> None:
        """Owner is garbage-collected once all entrypoints and user refs are gone."""
        pool = PriorityThreadPoolExecutor(max_workers=1)
        owner_id = pool._id
        weak = weakref.ref(pool)
        stage = pool.get_executor()

        self.assertIn(owner_id, _OWNER_REGISTRY)
        self.assertIsNotNone(weak())

        # Drop the user reference — entrypoint still holds a strong ref
        del pool
        gc.collect()
        self.assertIsNotNone(weak())

        # Drop the entrypoint — last strong ref gone
        del stage
        gc.collect()
        self.assertIsNone(weak())
        self.assertNotIn(owner_id, _OWNER_REGISTRY)

    def test_owner_gc_multiple_entrypoints(self) -> None:
        """Owner survives until ALL entrypoints are dropped."""
        pool = PriorityThreadPoolExecutor(max_workers=1)
        weak = weakref.ref(pool)
        s1 = pool.get_executor()
        s2 = pool.get_executor()
        del pool
        gc.collect()

        self.assertIsNotNone(weak())
        del s1
        gc.collect()
        self.assertIsNotNone(weak())
        del s2
        gc.collect()
        self.assertIsNone(weak())

    def test_owner_gc_after_shutdown(self) -> None:
        """Shutdown + dropping all refs allows GC."""
        pool = PriorityThreadPoolExecutor(max_workers=1)
        weak = weakref.ref(pool)
        stage = pool.get_executor()
        fut = stage.submit(lambda: 42)
        self.assertEqual(fut.result(timeout=5), 42)

        pool.shutdown()
        del pool
        gc.collect()
        # Entrypoint still holds a strong ref
        self.assertIsNotNone(weak())

        del stage
        gc.collect()
        self.assertIsNone(weak())


# ─── PriorityInterpreterPoolExecutor tests (Python 3.14+ only) ───


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

    class TestPriorityInterpreterPoolExecutor(unittest.TestCase):
        def test_basic_execution(self) -> None:
            from spdl.pipeline import PriorityInterpreterPoolExecutor

            executor = PriorityInterpreterPoolExecutor(max_workers=2)
            stage = executor.get_executor()
            futs = [stage.submit(pow, 2, x) for x in range(10)]
            results = {f.result(timeout=10) for f in futs}
            self.assertEqual(results, {2**x for x in range(10)})
            executor.shutdown()

        def test_exception_propagation(self) -> None:
            from spdl.pipeline import PriorityInterpreterPoolExecutor

            executor = PriorityInterpreterPoolExecutor(max_workers=1)
            stage = executor.get_executor()

            def fail() -> None:
                raise ValueError("interpreter boom")

            fut = stage.submit(fail)
            with self.assertRaises(ValueError):
                fut.result(timeout=10)
            executor.shutdown()

        def test_submit_after_shutdown_raises(self) -> None:
            from spdl.pipeline import PriorityInterpreterPoolExecutor

            executor = PriorityInterpreterPoolExecutor(max_workers=1)
            executor.get_executor()
            executor.shutdown()
            with self.assertRaises(RuntimeError):
                executor._submit_with_priority(  # pyre-ignore[16]
                    (0, 0), lambda: None, (), {}
                )

        def test_is_executor_compatible(self) -> None:
            from spdl.pipeline import PriorityInterpreterPoolExecutor

            executor = PriorityInterpreterPoolExecutor(max_workers=1)
            stage = executor.get_executor()
            self.assertIsInstance(stage, Executor)
            executor.shutdown()

        def test_pipeline_two_stage(self) -> None:
            from spdl.pipeline import PriorityInterpreterPoolExecutor

            pool = PriorityInterpreterPoolExecutor(max_workers=4)
            s1 = pool.get_executor()
            s2 = pool.get_executor()

            pipeline = (
                PipelineBuilder()
                .add_source(range(20))
                .pipe(lambda x: x * 2, executor=s1, concurrency=4)
                .pipe(lambda x: x + 1, executor=s2, concurrency=4)
                .add_sink(3)
                .build(num_threads=1)
            )

            with pipeline.auto_stop():
                results = sorted(pipeline.get_iterator(timeout=30))

            self.assertEqual(results, sorted(x * 2 + 1 for x in range(20)))
            pool.shutdown()
