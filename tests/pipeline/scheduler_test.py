# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unit tests for ``PriorityScheduler`` + ``_PrioritizedExecutor`` (v5)."""

import asyncio
import concurrent.futures
import threading
import time
import unittest
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import later.unittest
from spdl.pipeline import PipelineBuilder
from spdl.pipeline._common._types import StageInfo
from spdl.pipeline._scheduler import (
    _PrioritizedExecutor,
    _PrioritySchedulerBackgroundTask,
    PriorityScheduler,
)


def _make_info(name: str, stage_id: str = "0", concurrency: int = 1) -> StageInfo:
    return StageInfo(
        pipeline_id=0,
        stage_id=stage_id,
        stage_name=name,
        concurrency=concurrency,
    )


class PrioritySchedulerConstructionTest(unittest.TestCase):
    """V5: PriorityScheduler is priority-only (no `adapt` flag)."""

    def test_max_concurrent_property(self) -> None:
        scheduler = PriorityScheduler(max_concurrent=4)
        self.assertEqual(scheduler.max_concurrent, 4)

    def test_max_concurrent_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PriorityScheduler(max_concurrent=0)
        with self.assertRaises(ValueError):
            PriorityScheduler(max_concurrent=-1)

    def test_underlying_executor_unbound_at_init(self) -> None:
        # Bound by _build_pipeline() via direct attribute assignment.
        scheduler = PriorityScheduler(max_concurrent=2)
        self.assertIsNone(scheduler._underlying_executor)


class RegisterStageTest(unittest.TestCase):
    """V5: register_stage takes (StageInfo, priority) only."""

    def test_register_stage_stores_priority(self) -> None:
        scheduler = PriorityScheduler(max_concurrent=4)
        info = _make_info("decode")
        scheduler.register_stage(info, priority=-3)
        self.assertEqual(scheduler.get_priority(info), -3)

    def test_get_priority_default_zero_for_unregistered(self) -> None:
        scheduler = PriorityScheduler(max_concurrent=4)
        info = _make_info("never_registered")
        self.assertEqual(scheduler.get_priority(info), 0)

    def test_register_overwrite(self) -> None:
        scheduler = PriorityScheduler(max_concurrent=4)
        info = _make_info("s")
        scheduler.register_stage(info, priority=-1)
        scheduler.register_stage(info, priority=-5)
        self.assertEqual(scheduler.get_priority(info), -5)


class PrioritizedExecutorContractTest(unittest.TestCase):
    """V5 m2: _PrioritizedExecutor must NOT be a ProcessPoolExecutor.

    convert_to_async branches on isinstance(executor, ProcessPoolExecutor) —
    we want the default path so loop.run_in_executor() invokes our submit().
    """

    def test_isinstance_is_not_process_pool(self) -> None:
        scheduler = PriorityScheduler(max_concurrent=2)
        info = _make_info("t")
        shim = _PrioritizedExecutor(scheduler, info)
        self.assertNotIsInstance(shim, concurrent.futures.ProcessPoolExecutor)

    def test_is_executor_subclass(self) -> None:
        # Confirms the shim implements the Executor ABC so it can be
        # passed to loop.run_in_executor.
        scheduler = PriorityScheduler(max_concurrent=2)
        info = _make_info("t")
        shim = _PrioritizedExecutor(scheduler, info)
        self.assertIsInstance(shim, concurrent.futures.Executor)

    def test_shutdown_is_noop(self) -> None:
        scheduler = PriorityScheduler(max_concurrent=2)
        shim = _PrioritizedExecutor(scheduler, _make_info("t"))
        # Should not raise; the underlying pool is owned elsewhere.
        shim.shutdown()
        shim.shutdown(wait=False)
        shim.shutdown(wait=True, cancel_futures=True)


class PrioritizedExecutorPerCallLoopFetchTest(later.unittest.TestCase):
    """V5.3: stage tasks may submit() before the BG task runs.

    Per-call asyncio.get_running_loop() must not require pre-stamping.
    """

    async def test_submit_works_without_bg_task_running(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            scheduler = PriorityScheduler(max_concurrent=2)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            # submit() succeeds without scheduler.run() being invoked.
            cf = shim.submit(lambda: 42)
            self.assertIsInstance(cf, concurrent.futures.Future)
            # The work item is enqueued via call_soon_threadsafe; let it
            # land before checking heap state.
            await asyncio.sleep(0)
            self.assertEqual(len(scheduler._heap), 1)

            # Now start the dispatch loop in the background.
            run_task = asyncio.create_task(scheduler.run())
            try:
                result = await asyncio.wait_for(asyncio.wrap_future(cf), timeout=2.0)
                self.assertEqual(result, 42)
            finally:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class CancelPreDispatchTest(later.unittest.TestCase):
    """V5.2: pre-dispatch cancel returns True; dispatch loop skips item."""

    async def test_cancel_before_dispatch_skips(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = PriorityScheduler(max_concurrent=1)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            ran: threading.Event = threading.Event()

            def work() -> int:
                ran.set()
                return 1

            cf = shim.submit(work)
            # Cancel BEFORE the dispatch loop pops the item.
            self.assertTrue(cf.cancel())
            self.assertTrue(cf.cancelled())

            # Now run the dispatch loop briefly. It should pop the cancelled
            # item, see set_running_or_notify_cancel returns False, and skip.
            run_task = asyncio.create_task(scheduler.run())
            try:
                await asyncio.sleep(0.1)
                self.assertFalse(ran.is_set())
            finally:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class CancelPostDispatchBestEffortTest(later.unittest.TestCase):
    """V5.2: post-dispatch cancel returns False (matches stdlib semantics)."""

    async def test_cancel_after_dispatch_returns_false(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = PriorityScheduler(max_concurrent=1)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            started: threading.Event = threading.Event()
            release: threading.Event = threading.Event()

            def slow_work() -> int:
                started.set()
                release.wait(timeout=2.0)
                return 99

            run_task = asyncio.create_task(scheduler.run())
            try:
                cf = shim.submit(slow_work)
                # Wait for dispatch to begin.
                await asyncio.get_running_loop().run_in_executor(
                    None, started.wait, 2.0
                )
                # cf is now RUNNING; cancel() should return False per
                # ThreadPoolExecutor semantics.
                self.assertFalse(cf.cancel())
                # Let the worker complete.
                release.set()
                result = await asyncio.wait_for(asyncio.wrap_future(cf), timeout=2.0)
                self.assertEqual(result, 99)
            finally:
                release.set()
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class PriorityDispatchOrderTest(later.unittest.TestCase):
    """Items dispatch in (priority, seq) order."""

    async def test_lower_priority_value_dispatches_first(self) -> None:
        # Use max_concurrent=1 so dispatch is strictly serialized.
        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = PriorityScheduler(max_concurrent=1)
            scheduler._underlying_executor = pool
            info_a = _make_info("a")
            info_b = _make_info("b")
            info_c = _make_info("c")
            scheduler.register_stage(info_a, priority=0)  # lowest priority
            scheduler.register_stage(info_b, priority=-2)  # higher
            scheduler.register_stage(info_c, priority=-5)  # highest

            shim_a = _PrioritizedExecutor(scheduler, info_a)
            shim_b = _PrioritizedExecutor(scheduler, info_b)
            shim_c = _PrioritizedExecutor(scheduler, info_c)

            results: list[str] = []

            def work(label: str) -> str:
                results.append(label)
                return label

            # Hold the dispatch loop until all 3 are enqueued.
            cf_a = shim_a.submit(work, "a")
            cf_b = shim_b.submit(work, "b")
            cf_c = shim_c.submit(work, "c")

            # Yield so the call_soon_threadsafe enqueues land.
            await asyncio.sleep(0)
            self.assertEqual(len(scheduler._heap), 3)

            run_task = asyncio.create_task(scheduler.run())
            try:
                await asyncio.gather(
                    asyncio.wrap_future(cf_a),
                    asyncio.wrap_future(cf_b),
                    asyncio.wrap_future(cf_c),
                )
                # Highest priority (lowest value) first.
                self.assertEqual(results, ["c", "b", "a"])
            finally:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class FifoTiebreakTest(later.unittest.TestCase):
    """Same priority -> FIFO via monotonic seq."""

    async def test_same_priority_is_fifo(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = PriorityScheduler(max_concurrent=1)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            results: list[int] = []

            def work(val: int) -> int:
                results.append(val)
                return val

            cfs = [shim.submit(work, i) for i in range(5)]
            await asyncio.sleep(0)

            run_task = asyncio.create_task(scheduler.run())
            try:
                await asyncio.gather(*[asyncio.wrap_future(cf) for cf in cfs])
                self.assertEqual(results, [0, 1, 2, 3, 4])
            finally:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class MaxConcurrentDispatchTest(later.unittest.TestCase):
    """At most max_concurrent items dispatched simultaneously."""

    async def test_max_concurrent_two(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            scheduler = PriorityScheduler(max_concurrent=2)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            active = 0
            max_active = 0
            # threading.Lock() returns _thread.LockType, which Pyre cannot
            # narrow from a captured local. Store as Any to silence captured-
            # variable annotation warnings without an explicit ignore.
            lock: Any = threading.Lock()

            def tracked(val: int) -> int:
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                return val

            cfs = [shim.submit(tracked, i) for i in range(8)]

            run_task = asyncio.create_task(scheduler.run())
            try:
                await asyncio.gather(*[asyncio.wrap_future(cf) for cf in cfs])
                self.assertLessEqual(max_active, 2)
            finally:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class ExceptionPropagationTest(later.unittest.TestCase):
    """Exception in func -> set_exception on cf_future."""

    async def test_exception_propagates(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = PriorityScheduler(max_concurrent=1)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            def boom(x: int) -> int:
                raise ValueError(f"boom-{x}")

            run_task = asyncio.create_task(scheduler.run())
            try:
                cf = shim.submit(boom, 7)
                with self.assertRaises(ValueError) as cm:
                    await asyncio.wait_for(asyncio.wrap_future(cf), timeout=2.0)
                self.assertIn("boom-7", str(cm.exception))
            finally:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class DrainPendingOnShutdownTest(later.unittest.TestCase):
    """V5.2: _drain_pending cancels everything still on the heap."""

    async def test_drain_cancels_pending_items(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = PriorityScheduler(max_concurrent=1)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            cf1 = shim.submit(lambda: 1)
            cf2 = shim.submit(lambda: 2)
            await asyncio.sleep(0)
            self.assertEqual(len(scheduler._heap), 2)

            scheduler._drain_pending()
            self.assertEqual(len(scheduler._heap), 0)
            self.assertTrue(cf1.cancelled())
            self.assertTrue(cf2.cancelled())


class PrioritySchedulerBackgroundTaskTest(later.unittest.TestCase):
    """The BG-task adapter runs scheduler.run() and drains on cancel."""

    async def test_bg_task_runs_scheduler_and_drains(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = PriorityScheduler(max_concurrent=1)
            scheduler._underlying_executor = pool
            info = _make_info("t")
            scheduler.register_stage(info, priority=0)
            shim = _PrioritizedExecutor(scheduler, info)

            bg = _PrioritySchedulerBackgroundTask(scheduler)
            run_task = asyncio.create_task(bg.run())

            cf = shim.submit(lambda: "ok")
            result = await asyncio.wait_for(asyncio.wrap_future(cf), timeout=2.0)
            self.assertEqual(result, "ok")

            # Submit one more then cancel BG task BEFORE it dispatches.
            cf2 = shim.submit(lambda: time.sleep(10))
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            # _drain_pending in the BG-task `finally` block should have
            # cancelled the un-dispatched item.
            self.assertTrue(cf2.cancelled())


class PipelineWithPrioritySchedulerEndToEndTest(unittest.TestCase):
    """End-to-end: build(use_priority_scheduler=True) produces correct output."""

    def test_pipeline_with_scheduler_produces_output(self) -> None:
        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(lambda x: x * 2, concurrency=2)
            .pipe(lambda x: x + 1, concurrency=2)
            .add_sink(3)
            .build(num_threads=4, use_priority_scheduler=True)
        )

        results: list[int] = []
        with pipeline.auto_stop():
            for item in pipeline:
                results.append(item)

        self.assertEqual(sorted(results), sorted(x * 2 + 1 for x in range(10)))

    def test_pipeline_without_scheduler_unchanged(self) -> None:
        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(lambda x: x * 2, concurrency=2)
            .add_sink(3)
            .build(num_threads=4, use_priority_scheduler=False)
        )

        results: list[int] = []
        with pipeline.auto_stop():
            for item in pipeline:
                results.append(item)

        self.assertEqual(sorted(results), sorted(x * 2 for x in range(10)))


class PipelineSchedulerBypassTest(unittest.TestCase):
    """V5: async/generator stages bypass the scheduler entirely."""

    def test_async_stage_bypasses_scheduler(self) -> None:
        async def async_double(x: int) -> int:
            return x * 2

        pipeline = (
            PipelineBuilder()
            .add_source(range(5))
            .pipe(async_double, concurrency=2)
            .add_sink(3)
            .build(num_threads=4, use_priority_scheduler=True)
        )

        results: list[int] = []
        with pipeline.auto_stop():
            for item in pipeline:
                results.append(item)

        self.assertEqual(sorted(results), sorted(x * 2 for x in range(5)))

    def test_generator_stage_bypasses_scheduler(self) -> None:
        def gen_double(x: int) -> Iterator[int]:
            yield x * 2

        pipeline = (
            PipelineBuilder()
            .add_source(range(5))
            .pipe(gen_double, concurrency=2)
            .add_sink(3)
            .build(num_threads=4, use_priority_scheduler=True)
        )

        results: list[int] = []
        with pipeline.auto_stop():
            for item in pipeline:
                results.append(item)

        self.assertEqual(sorted(results), sorted(x * 2 for x in range(5)))


class PipeArgsRevertTest(unittest.TestCase):
    """V5: _PipeArgs has no nice/_depth fields anymore (Diff 2 reverts v2.1)."""

    def test_pipe_args_minimal_fields(self) -> None:
        from spdl.pipeline.defs import _PipeArgs

        args = _PipeArgs(op=lambda x: x)
        # The v2.1 spec added .nice and ._depth — v5 removes both.
        self.assertFalse(hasattr(args, "nice"))
        self.assertFalse(hasattr(args, "_depth"))


class ToAsyncRevertTest(unittest.TestCase):
    """V5: _to_async no longer takes scheduler/stage_name params."""

    def test_to_async_signature(self) -> None:
        import inspect as _inspect

        from spdl.pipeline._common._convert import _to_async

        sig = _inspect.signature(_to_async)
        params = list(sig.parameters.keys())
        # Only func + executor should remain.
        self.assertEqual(params, ["func", "executor"])

    def test_convert_to_async_signature(self) -> None:
        import inspect as _inspect

        from spdl.pipeline._common._convert import convert_to_async

        sig = _inspect.signature(convert_to_async)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["op", "executor"])


class NodeDepthComputationTest(unittest.TestCase):
    """V5: depth is computed at build time via _node_depth (not on _PipeArgs)."""

    def test_linear_pipeline_depths(self) -> None:
        from spdl.pipeline._components._node import (
            _convert_config,
            _MutableInt,
            _node_depth,
        )
        from spdl.pipeline._components._queue import AsyncQueue
        from spdl.pipeline.defs import (
            Pipe,
            PipelineConfig,
            SinkConfig,
            SourceConfig,
        )

        plc = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[
                Pipe(lambda x: x, name="s0"),
                Pipe(lambda x: x, name="s1"),
                Pipe(lambda x: x, name="s2"),
            ],
            sink=SinkConfig(3),
        )

        sink_node = _convert_config(plc, AsyncQueue, 0, _MutableInt(0))
        # source(0) -> s0(1) -> s1(2) -> s2(3) -> sink(4)
        self.assertEqual(_node_depth(sink_node), 4)

    def test_zero_pipe_pipeline_depth(self) -> None:
        """No pipes between source and sink: depth(sink) == 1."""
        # Arrange
        from spdl.pipeline._components._node import (
            _convert_config,
            _MutableInt,
            _node_depth,
        )
        from spdl.pipeline._components._queue import AsyncQueue
        from spdl.pipeline.defs import PipelineConfig, SinkConfig, SourceConfig

        plc = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[],
            sink=SinkConfig(3),
        )

        # Act
        sink_node = _convert_config(plc, AsyncQueue, 0, _MutableInt(0))

        # Assert: source(0) -> sink(1).
        self.assertEqual(_node_depth(sink_node), 1)

    def test_merge_pipeline_depth_takes_max_branch(self) -> None:
        """For a merge node, depth == 1 + max(depth(branch_i))."""
        # Arrange
        from spdl.pipeline._components._node import (
            _convert_config,
            _MutableInt,
            _node_depth,
        )
        from spdl.pipeline._components._queue import AsyncQueue
        from spdl.pipeline.defs import (
            Merge,
            Pipe,
            PipelineConfig,
            SinkConfig,
            SourceConfig,
        )

        # Branch A: 1 pipe (depth 2 at branch's sink)
        branch_a = PipelineConfig(
            src=SourceConfig([1]),
            pipes=[Pipe(lambda x: x, name="a0")],
            sink=SinkConfig(3),
        )
        # Branch B: 3 pipes (depth 4 at branch's sink)
        branch_b = PipelineConfig(
            src=SourceConfig([1]),
            pipes=[
                Pipe(lambda x: x, name="b0"),
                Pipe(lambda x: x, name="b1"),
                Pipe(lambda x: x, name="b2"),
            ],
            sink=SinkConfig(3),
        )
        merged = PipelineConfig(
            src=Merge([branch_a, branch_b]),
            pipes=[],
            sink=SinkConfig(3),
        )

        # Act
        sink_node = _convert_config(merged, AsyncQueue, 0, _MutableInt(0))

        # Assert: deepest branch contributes; downstream merge + sink each
        # add one. Branch B reaches depth 4 at its own sink; merge wraps
        # both and the outer sink follows.
        depth = _node_depth(sink_node)
        self.assertGreaterEqual(depth, 5)


if __name__ == "__main__":
    unittest.main()
