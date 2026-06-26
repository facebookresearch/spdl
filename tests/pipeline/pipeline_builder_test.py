# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import functools
import multiprocessing as mp
import os
import pickle
import platform
import random
import re
import sys
import threading
import time
import unittest
import warnings
from collections.abc import Iterator
from concurrent.futures import (
    BrokenExecutor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from contextlib import asynccontextmanager
from functools import partial
from multiprocessing import Process
from typing import Any, cast, TypeVar

from parameterized import parameterized
from spdl.pipeline import (
    AsyncQueue,
    PipelineBuilder,
    PipelineFailure,
    PriorityThreadPoolExecutor,
    run_pipeline_in_subprocess,
    TaskHook,
    TaskStatsHook,
)
from spdl.pipeline._common._convert import _is_process_pool
from spdl.pipeline._components import _get_global_id, _set_global_id
from spdl.pipeline._components._common import _EOF, StageInfo
from spdl.pipeline._components._hook import _periodic_dispatch
from spdl.pipeline._components._pipe import (
    _FailCounter,
    _get_fail_counter,
    _pipe,
    _PipeArgs,
)
from spdl.pipeline._components._sink import _sink
from spdl.pipeline._components._source import _source
from spdl.pipeline._executor_proxy import (
    _ExecutorProxy,
    _interpreter_pool_kwargs,
    _make_config_executors_picklable,
)
from spdl.pipeline._subprocess_worker_pool import (
    _hoist_process_pools,
    _RemoteExecutor,
    _shutdown_pools,
    _WorkerPool,
)
from spdl.pipeline.defs import (
    Aggregator,
    Merge,
    MergeConfig,
    PathVariants,
    PathVariantsConfig,
    Pipe,
    PipeConfig,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)
from spdl.source.utils import embed_shuffle

T = TypeVar("T")


def _ignore_warnings(*filters):
    """Decorator that wraps a test in `warnings.catch_warnings()` and applies
    the given filters. Each ``filter`` is a dict of kwargs forwarded to
    ``warnings.filterwarnings``.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                for f in filters:
                    warnings.filterwarnings("ignore", **f)
                return fn(*args, **kwargs)

        return wrapper

    return decorator


_FORK_WARNING = {
    "message": (
        r"This process \(pid=\d+\) is multi-threaded, use of fork\(\) "
        r"may lead to deadlocks in the child"
    ),
    "category": DeprecationWarning,
}

_RUN_PIPELINE_DEPRECATION = {
    "message": (
        r"Passing a `PipelineBuilder` object directly to "
        r"`run_pipeline_in_subprocess` is now deprecated\..*"
    ),
    "category": UserWarning,
}

_UNAWAITED_COROUTINE = {
    "message": "coroutine .* was never awaited",
    "category": RuntimeWarning,
}

_PROCESSPOOL_FORK_WARNING = {
    "message": r"Hoisting a ProcessPoolExecutor .* 'fork' start method .* can deadlock\..*",
    "category": RuntimeWarning,
}


def _SI(name: str) -> StageInfo:
    """Shorthand for creating a test StageInfo."""
    return StageInfo(pipeline_id=0, stage_id="0", stage_name=name)


def _put_aqueue(queue, vals, *, eof):
    for val in vals:
        queue.put_nowait(val)
    if eof:
        queue.put_nowait(_EOF)


def _flush_aqueue(queue):
    ret = []
    while not queue.empty():
        ret.append(queue.get_nowait())
    return ret


async def no_op(val):
    return val


################################################################################
# _source
################################################################################


class TestSource(unittest.TestCase):
    def test_async_enqueue_empty(self) -> None:
        """_async_enqueue can handle empty iterator"""
        queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="foo"), buffer_size=0
        )
        coro = _source([], queue)
        asyncio.run(coro)
        self.assertEqual(_flush_aqueue(queue), [_EOF])

    def test_async_enqueue_simple(self) -> None:
        """_async_enqueue should put the values in the queue."""
        src = list(range(6))
        queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="foo"), buffer_size=0
        )
        coro = _source(src, queue)
        asyncio.run(coro)
        vals = _flush_aqueue(queue)
        self.assertEqual(vals, [*src, _EOF])

    def test_async_enqueue_iterator_failure(self) -> None:
        """When `iterator` fails, the exception is propagated."""

        def src():
            yield from range(10)
            raise RuntimeError("Failing the iterator.")

        coro = _source(
            src(),
            AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="foo"), buffer_size=0
            ),
        )

        with self.assertRaises(RuntimeError):
            asyncio.run(coro)  # Not raising

    def test_async_enqueue_cancel(self) -> None:
        """_async_enqueue is cancellable."""

        async def _test():
            queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="foo"), buffer_size=1
            )

            src = list(range(3))

            coro = _source(src, queue)
            task = asyncio.create_task(coro)

            await asyncio.sleep(0.1)

            task.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(_test())


################################################################################
# _sink
################################################################################


class TestSink(unittest.TestCase):
    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_async_sink_simple(self, empty: bool) -> None:
        """_sink pass the contents from input_queue to output_queue"""
        input_queue: AsyncQueue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue: AsyncQueue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        data = [] if empty else list(range(3))
        _put_aqueue(input_queue, data, eof=True)

        coro = _sink(input_queue, output_queue)

        asyncio.run(coro)
        results = _flush_aqueue(output_queue)

        self.assertEqual(results, data)

    def test_async_sink_cancel(self) -> None:
        """_async_sink is cancellable."""

        async def _test():
            input_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="input")
            )
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )

            coro = _sink(input_queue, output_queue)
            task = asyncio.create_task(coro)

            await asyncio.sleep(0.1)

            task.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(_test())


################################################################################
# _pipe
################################################################################


async def adouble(val: int):
    return 2 * val


async def aplus1(val: int):
    return val + 1


async def passthrough(val):
    print("passthrough:", val)
    return val


class TestPipe(unittest.IsolatedAsyncioTestCase):
    def test_async_pipe(self) -> None:
        """_pipe processes the data in input queue and pass it to output queue."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        async def test():
            ref = list(range(6))
            _put_aqueue(input_queue, ref, eof=True)

            await _pipe(
                _SI("adouble"),
                input_queue,
                output_queue,
                _PipeArgs(op=adouble),
                _FailCounter(),
                [],
                False,
            )

            result = _flush_aqueue(output_queue)

            self.assertEqual(result, [v * 2 for v in ref] + [_EOF])

        asyncio.run(test())

    def test_async_pipe_skip(self) -> None:
        """_pipe skips the result if it's None."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        async def skip_even(v):
            if v % 2:
                return v

        async def test():
            _put_aqueue(input_queue, range(10), eof=True)

            await _pipe(
                _SI("skip_even"),
                input_queue,
                output_queue,
                _PipeArgs(op=skip_even),
                _FailCounter(),
                [],
                False,
            )

            result = _flush_aqueue(output_queue)

            self.assertEqual(result, [*list(range(1, 10, 2)), _EOF])

        asyncio.run(test())

    def test_async_pipe_wrong_task_signature(self) -> None:
        """_pipe fails immediately if user provided incompatible iterator/afunc."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        async def _2args(val: int, _):
            return val

        async def test():
            ref = list(range(6))
            _put_aqueue(input_queue, ref, eof=False)

            with self.assertRaises(TypeError):
                await _pipe(
                    _SI("_2args"),
                    input_queue,
                    output_queue,
                    _PipeArgs(op=_2args, concurrency=3),
                    _FailCounter(),
                    [],
                    False,
                )

            remaining = _flush_aqueue(input_queue)
            self.assertEqual(remaining, ref[1:])

            result = _flush_aqueue(output_queue)
            self.assertEqual(result, [_EOF])

        asyncio.run(test())

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_async_pipe_cancel(self, full: bool) -> None:
        """_pipe is cancellable."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=1
        )

        _put_aqueue(input_queue, list(range(3)), eof=False)

        if full:
            output_queue.put_nowait(None)

        cancelled = False

        async def astuck(i):
            try:
                await asyncio.sleep(10)
                return i
            except asyncio.CancelledError:
                nonlocal cancelled
                cancelled = True
                raise

        async def test():
            coro = _pipe(
                _SI("astuck"),
                input_queue,
                output_queue,
                _PipeArgs(op=astuck),
                _FailCounter(),
                [],
                False,
            )
            task = asyncio.create_task(coro)

            await asyncio.sleep(0.5)

            task.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertFalse(cancelled)
        asyncio.run(test())
        self.assertTrue(cancelled)

    def test_async_pipe_concurrency(self) -> None:
        """Changing concurrency changes the number of items fetched and processed."""

        async def delay(val):
            await asyncio.sleep(0.5)
            return val

        async def test(concurrency):
            input_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="input"),
                buffer_size=0,
            )
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output"),
                buffer_size=0,
            )

            ref = [1, 2, 3, 4]
            _put_aqueue(input_queue, ref, eof=False)

            coro = _pipe(
                _SI("delay"),
                input_queue,
                output_queue,
                _PipeArgs(
                    op=delay,
                    concurrency=concurrency,
                ),
                _FailCounter(),
                [],
                False,
            )

            task = asyncio.create_task(coro)
            await asyncio.sleep(0.8)
            task.cancel()

            return _flush_aqueue(input_queue), _flush_aqueue(output_queue)

        # With concurrency==1, there should be
        # 1 in output_queue, 2 is in flight, 3 and 4 remain in input_queue
        remain, output = asyncio.run(test(1))
        self.assertEqual(remain, [3, 4])
        self.assertEqual(output, [1])

        # With concurrency==4, there should be
        # 1, 2, 3 and 4 in output_queue.
        remain, output = asyncio.run(test(4))
        self.assertEqual(remain, [])
        self.assertEqual(set(output), {1, 2, 3, 4})

    def test_async_pipe_concurrency_runs_ops_in_parallel(self) -> None:
        """A pool pipe runs up to `concurrency` ops simultaneously.

        Every op increments a shared counter and then waits on a single event that is
        only set once the counter reaches `concurrency` -- i.e. once all `concurrency`
        ops are in flight at the same time. If the pipe ran them serially the event
        would never be set and the wait would time out, proving real parallelism without
        depending on wall-clock timing.

        A hand-rolled counter+event is used rather than ``asyncio.Barrier`` because the
        latter is Python 3.11+ and SPDL supports 3.10.
        """
        concurrency = 4
        all_running = asyncio.Event()
        num_running = 0

        async def op(val):
            nonlocal num_running
            num_running += 1
            if num_running == concurrency:
                all_running.set()
            await asyncio.wait_for(all_running.wait(), timeout=30)
            return val

        async def test():
            input_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="input"),
                buffer_size=0,
            )
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output"),
                buffer_size=0,
            )

            ref = [4, 5, 6, 7, _EOF]
            _put_aqueue(input_queue, ref, eof=False)

            await _pipe(
                _SI("op"),
                input_queue,
                output_queue,
                _PipeArgs(
                    op=op,
                    concurrency=concurrency,
                ),
                _FailCounter(),
                [],
                False,
            )

            result = _flush_aqueue(output_queue)

            self.assertEqual(set(result), set(ref))
            self.assertEqual(result[-1], ref[-1])
            self.assertEqual(result[-1], _EOF)

        asyncio.run(test())


################################################################################
# Pipeline
################################################################################


class TestPipeline(unittest.TestCase):
    def test_pipeline_stage_hook_wrong_def1(self) -> None:
        """Pipeline fails if stage_hook is not properly overrode."""

        class _hook(TaskHook):
            # missing asynccontextmanager
            async def stage_hook(self):
                yield

            @asynccontextmanager
            async def task_hook(self, input_item=None):
                yield

        with self.assertRaises(ValueError):
            (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(passthrough)
                .add_sink()
                # pyre-ignore
                .build(num_threads=1, task_hook_factory=lambda _: [_hook()])
            )

    def test_pipeline_stage_hook_wrong_def2(self) -> None:
        """Pipeline fails if task_hook is not properly overrode."""

        class _hook(TaskHook):
            # missing asynccontextmanager and async keyword
            def stage_hook(self):
                yield

            @asynccontextmanager
            async def task_hook(self, input_item=None):
                yield

        with self.assertRaises(ValueError):
            (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(passthrough)
                .add_sink()
                # pyre-ignore
                .build(num_threads=1, task_hook_factory=lambda _: [_hook()])
            )


class CountHook(TaskHook):
    def __init__(self):
        self._enter_task_called = 0
        self._enter_stage_called = 0
        self._exit_task_called = 0
        self._exit_stage_called = 0

    @asynccontextmanager
    async def stage_hook(self):
        self._enter_stage_called += 1
        yield
        self._exit_stage_called += 1

    @asynccontextmanager
    async def task_hook(self, input_item=None):
        self._enter_task_called += 1
        try:
            yield
        finally:
            self._exit_task_called += 1


class TestPipelineHook(unittest.TestCase):
    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_pipeline_hook_drop_last(self, drop_last: bool) -> None:
        """Hook is executed properly"""

        h1, h2, h3 = CountHook(), CountHook(), CountHook()

        def hook_factory(name) -> list[TaskHook]:
            sname = str(name)
            if "adouble" in sname:
                return [h1]
            if "aggregate" in sname:
                return [h2]
            if "_fail" in sname:
                return [h3]
            raise RuntimeError(f"Unexpected name: {sname}")

        async def _fail(_):
            raise RuntimeError("Failing")

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(adouble)
            .aggregate(5, drop_last=drop_last)
            .pipe(_fail)
            .add_sink(1000)
            .build(num_threads=1, task_hook_factory=hook_factory)
        )

        with pipeline.auto_stop():
            self.assertEqual([], list(pipeline.get_iterator(timeout=30)))

        self.assertEqual(h1._enter_stage_called, 1)
        self.assertEqual(h1._exit_stage_called, 1)
        self.assertEqual(h1._enter_task_called, 10)
        self.assertEqual(h1._exit_task_called, 10)

        self.assertEqual(h2._enter_stage_called, 1)
        self.assertEqual(h2._exit_stage_called, 1)
        # When drop_last=False, EOF is passed to the aggregation operator (11 calls: 10 items + 1 EOF)
        # When drop_last=True, EOF is NOT passed to the aggregation operator (10 calls: 10 items only)
        expected_h2_calls = 10 if drop_last else 11
        self.assertEqual(h2._enter_task_called, expected_h2_calls)
        self.assertEqual(h2._exit_task_called, expected_h2_calls)

        # Even when the stage task fails,
        # the exit_stage and exit_task are still called.
        self.assertEqual(h3._enter_stage_called, 1)
        self.assertEqual(h3._exit_stage_called, 1)
        self.assertEqual(h3._enter_task_called, 2)
        self.assertEqual(h3._exit_task_called, 2)

    def test_pipeline_hook_multiple(self) -> None:
        """Multiple hooks are executed properly"""

        class _hook(TaskHook):
            def __init__(self):
                self._enter_task_called = 0
                self._enter_stage_called = 0
                self._exit_task_called = 0
                self._exit_stage_called = 0

            @asynccontextmanager
            async def stage_hook(self):
                self._enter_stage_called += 1
                yield
                self._exit_stage_called += 1

            @asynccontextmanager
            async def task_hook(self, input_item=None):
                self._enter_task_called += 1
                try:
                    yield
                finally:
                    self._exit_task_called += 1

        hooks = [_hook(), _hook(), _hook()]

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1000)
            # pyre-ignore[6]
            .build(num_threads=1, task_hook_factory=lambda _: hooks)
        )

        with pipeline.auto_stop():
            self.assertEqual(list(range(10)), list(pipeline.get_iterator(timeout=30)))

        for h in hooks:
            self.assertEqual(h._enter_stage_called, 1)
            self.assertEqual(h._exit_stage_called, 1)
            self.assertEqual(h._enter_task_called, 10)
            self.assertEqual(h._exit_task_called, 10)

    @_ignore_warnings({"category": RuntimeWarning})
    @_ignore_warnings(_UNAWAITED_COROUTINE)
    def test_pipeline_hook_failure_enter_stage(self) -> None:
        """If enter_stage fails, the pipeline is aborted."""

        class _enter_stage_fail(TaskHook):
            @asynccontextmanager
            async def stage_hook(self):
                raise RuntimeError("failing")

            @asynccontextmanager
            async def task_hook(self, input_item=None):
                yield

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1000)
            # pyre-ignore[6]
            .build(num_threads=1, task_hook_factory=lambda _: [_enter_stage_fail()])
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(vals, [])

    @_ignore_warnings({"category": RuntimeWarning})
    @_ignore_warnings(_UNAWAITED_COROUTINE)
    def test_pipeline_hook_failure_exit_stage(self) -> None:
        """If exit_stage fails, the error is propagated to the front end."""

        class _exit_stage_fail(TaskHook):
            @asynccontextmanager
            async def stage_hook(self):
                yield
                raise RuntimeError("failing")

            @asynccontextmanager
            async def task_hook(self, input_item=None):
                yield

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1000)
            # pyre-ignore[6]
            .build(num_threads=1, task_hook_factory=lambda _: [_exit_stage_fail()])
        )
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))
        self.assertEqual(vals, list(range(10)))

    @_ignore_warnings({"category": RuntimeWarning})
    def test_pipeline_hook_failure_enter_task(self) -> None:
        """If enter_task fails, the pipeline does not fail."""

        class _hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                raise RuntimeError("failing enter_task")

            @asynccontextmanager
            async def stage_hook(self, *_):
                yield

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1000)
            # pyre-ignore[6]
            .build(num_threads=1, task_hook_factory=lambda _: [_hook()])
        )

        with pipeline.auto_stop():
            self.assertEqual([], list(pipeline.get_iterator(timeout=30)))

    @_ignore_warnings({"category": RuntimeWarning})
    def test_pipeline_hook_failure_exit_task(self) -> None:
        """If exit_task fails, the pipeline does not fail.

        IMPORTANT: The result is dropped.
        """

        class _exit_stage_fail(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                yield
                raise RuntimeError("failing exit_task")

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1000)
            # pyre-ignore[6]
            .build(num_threads=1, task_hook_factory=lambda _: [_exit_stage_fail()])
        )

        with pipeline.auto_stop():
            self.assertEqual(list(pipeline.get_iterator(timeout=30)), [])

    def test_pipeline_hook_exit_task_capture_error(self) -> None:
        """If task fails exit_task captures the error."""

        exc_info = None

        class _capture(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                try:
                    yield
                except Exception as e:
                    nonlocal exc_info
                    exc_info = e

        err = RuntimeError("failing")

        async def _fail(_):
            raise err

        pipeline = (
            PipelineBuilder()
            .add_source([None])
            .pipe(_fail)
            .add_sink(100)
            .build(
                num_threads=1,
                # pyre-ignore[6]
                task_hook_factory=lambda _: [_capture()],
            )
        )

        with pipeline.auto_stop():
            self.assertEqual(list(pipeline.get_iterator(timeout=30)), [])

        self.assertTrue(exc_info is err)

    def test_pipeline_hook_receives_input_item(self) -> None:
        """task_hook receives the input_item being processed."""

        received_items = []

        class _item_capture_hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                received_items.append(input_item)
                yield

        pipeline = (
            PipelineBuilder()
            .add_source(range(5))
            .pipe(passthrough)
            .add_sink(1000)
            # pyre-ignore[6]
            .build(num_threads=1, task_hook_factory=lambda _: [_item_capture_hook()])
        )

        with pipeline.auto_stop():
            output = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(output, list(range(5)))
        self.assertEqual(received_items, list(range(5)))

    def test_pipeline_hook_receives_input_item_on_failure(self) -> None:
        """task_hook receives input_item even when the task fails."""

        captured_items_on_failure = []

        class _failure_capture_hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                try:
                    yield
                except StopAsyncIteration:
                    raise
                except Exception:
                    captured_items_on_failure.append(input_item)
                    raise

        def fail_on_even(x: int) -> int:
            if x % 2 == 0:
                raise RuntimeError(f"fail on {x}")
            return x

        pipeline = (
            PipelineBuilder()
            .add_source(range(6))
            .pipe(fail_on_even)
            .add_sink(1000)
            .build(
                num_threads=1,
                task_hook_factory=lambda _: [_failure_capture_hook()],  # pyre-ignore[6]
            )
        )

        with pipeline.auto_stop():
            output = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(output, [1, 3, 5])
        self.assertEqual(captured_items_on_failure, [0, 2, 4])

    def test_ordered_pipe_hook_receives_input_item(self) -> None:
        """task_hook receives input_item in ordered pipe."""

        received_items = []

        class _item_capture_hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                received_items.append(input_item)
                yield

        pipeline = (
            PipelineBuilder()
            .add_source(range(5))
            .pipe(passthrough, output_order="input", concurrency=4)
            .add_sink(1000)
            # pyre-ignore[6]
            .build(num_threads=4, task_hook_factory=lambda _: [_item_capture_hook()])
        )

        with pipeline.auto_stop():
            output = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(output, list(range(5)))
        self.assertEqual(received_items, list(range(5)))

    def test_ordered_pipe_hook_receives_input_item_on_failure(self) -> None:
        """task_hook receives input_item on failure in ordered pipe."""

        captured_items_on_failure = []

        class _failure_capture_hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                try:
                    yield
                except StopAsyncIteration:
                    raise
                except Exception:
                    captured_items_on_failure.append(input_item)
                    raise

        def fail_on_even(x: int) -> int:
            if x % 2 == 0:
                raise RuntimeError(f"fail on {x}")
            return x

        pipeline = (
            PipelineBuilder()
            .add_source(range(6))
            .pipe(fail_on_even, output_order="input", concurrency=4)
            .add_sink(1000)
            .build(
                num_threads=4,
                task_hook_factory=lambda _: [_failure_capture_hook()],  # pyre-ignore[6]
            )
        )

        with pipeline.auto_stop():
            output = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(output, [1, 3, 5])
        self.assertEqual(sorted(captured_items_on_failure), [0, 2, 4])


################################################################################
# TaskStatsHook
################################################################################


class TestTaskStatsHook(unittest.TestCase):
    def test_task_stats(self) -> None:
        """TaskStatsHook logs the interval of each task."""

        hook = TaskStatsHook(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="foo"), 1
        )

        async def _test():
            async with hook.stage_hook():
                for _ in range(3):
                    async with hook.task_hook():
                        await asyncio.sleep(0.5)

                self.assertEqual(hook.num_tasks, 3)
                self.assertEqual(hook.num_success, 3)
                self.assertGreater(hook.ave_time, 0.3)
                self.assertLess(hook.ave_time, 0.7)

                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        async with hook.task_hook():
                            await asyncio.sleep(1.0)
                            raise RuntimeError("failing")

                self.assertEqual(hook.num_tasks, 5)
                self.assertEqual(hook.num_success, 3)
                self.assertGreater(hook.ave_time, 0.45)
                self.assertLess(hook.ave_time, 0.9)

        asyncio.run(_test())


class TestPeriodicDispatch(unittest.TestCase):
    def test_periodic_dispatch_smoke_test(self) -> None:
        """_periodic_dispatch runs functions with the given interval."""

        calls = []

        async def afun():
            print("afun: ", time.time())
            calls.append(time.monotonic())

        async def _test():
            done = asyncio.Event()
            task = asyncio.create_task(_periodic_dispatch(afun, done, 1))

            await asyncio.sleep(3.2)

            done.set()
            await task

        print("start: ", time.time())
        asyncio.run(_test())

        self.assertEqual(len(calls), 3)
        self.assertGreater(calls[1] - calls[0], 0.9)
        self.assertLess(calls[1] - calls[0], 1.1)
        self.assertGreater(calls[2] - calls[1], 0.9)
        self.assertLess(calls[2] - calls[1], 1.1)

    def test_task_stats_log_interval_stats(self) -> None:
        """Smoke test for _log_interval_stats."""

        hook = TaskStatsHook(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="foo"), 1
        )
        asyncio.run(hook._log_interval_stats())


################################################################################
# __str__
################################################################################


class TestPipelineStr(unittest.TestCase):
    def test_pipeline_str_smoke(self) -> None:
        async def passthrough(i):
            return i

        builder = PipelineBuilder()

        print(builder)

        builder = builder.add_source(range(10))

        print(builder)

        builder = builder.pipe(passthrough)

        print(builder)

        builder = builder.aggregate(1)

        print(builder)

        builder = builder.pipe(passthrough, output_order="input")

        print(builder)

        builder = builder.aggregate(1)

        print(builder)

        builder = builder.add_sink(100)

        print(builder)


################################################################################
# AsyncPipeline - resume
################################################################################


class TestPipelineResume(unittest.TestCase):
    def test_pipeline_reiterate(self) -> None:
        """Pipeline can be iterated multiple times as long as it's not stopped"""

        pipeline = (
            PipelineBuilder().add_source(range(20)).add_sink(1000).build(num_threads=1)
        )

        with pipeline.auto_stop():
            for i in range(5):
                for j, val in enumerate(pipeline.get_iterator(timeout=30)):
                    self.assertEqual(val, (i * 4) + j)

            # Now it's empty
            with self.assertRaises(StopIteration):
                next(pipeline.get_iterator(timeout=30))

    def test_pipeline_resume(self) -> None:
        """AsyncPipeline can execute the source partially, then resumed"""

        # Note
        # If we pass `range(10)` directly, new iterator is created at every run.
        src = iter(range(10))

        pipeline = PipelineBuilder().add_source(src).add_sink(1000).build(num_threads=1)

        with pipeline.auto_stop():
            iterator = pipeline.get_iterator(timeout=30)
            self.assertEqual([0, 1], [next(iterator) for _ in range(2)])

            iterator = pipeline.get_iterator(timeout=30)
            self.assertEqual([2, 3, 4], [next(iterator) for _ in range(3)])

            iterator = pipeline.get_iterator(timeout=30)
            self.assertEqual([5, 6, 7, 8, 9], [next(iterator) for _ in range(5)])

            with self.assertRaises(StopIteration):
                next(iterator)

    def test_pipeline_infinite_loop(self) -> None:
        """AsyncPipeline can execute infinite iterable"""

        def src(i=-1):
            while True:
                yield (i := i + 1)

        pipeline = (
            PipelineBuilder().add_source(src()).add_sink(1000).build(num_threads=1)
        )

        with pipeline.auto_stop():
            i = 0
            for _ in range(10):
                num_items = random.randint(1, 128)
                for j, item in enumerate(pipeline.get_iterator(timeout=30)):
                    self.assertEqual(item, i)
                    i += 1

                    if num_items == j:
                        break


################################################################################
# AsyncPipeline - order
################################################################################


class TestPipelineOrder(unittest.TestCase):
    def test_pipeline_order_complete(self) -> None:
        """The output is in the order of completion."""

        async def _sleep(i):
            await asyncio.sleep(i / 10)
            return i

        src = list(reversed(range(10)))
        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .pipe(_sleep, concurrency=10, output_order="completion")
            .add_sink(100)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            self.assertEqual(list(pipeline.get_iterator(timeout=30)), list(range(10)))

    def test_pipeline_order_input(self) -> None:
        """The output is in the order of the input."""

        async def _sleep(i):
            print(f"Sleeping: {i}")
            await asyncio.sleep(i / 10)
            print(f"Returning: {i}")
            return i

        src = list(reversed(range(10)))
        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .pipe(_sleep, concurrency=10, output_order="input")
            .add_sink(100)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            self.assertEqual(src, list(pipeline.get_iterator(timeout=30)))

    def test_pipeline_order_input_sync_func(self) -> None:
        """The output is in the order of the input."""

        def _sleep(i):
            print(f"Sleeping: {i}")
            time.sleep(i / 10)
            print(f"Returning: {i}")
            return i

        src = list(reversed(range(10)))
        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .pipe(_sleep, concurrency=10, output_order="input")
            .add_sink(100)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            self.assertEqual(list(pipeline.get_iterator(timeout=30)), src)

    @_ignore_warnings({"category": RuntimeWarning})
    def test_pipeline_order_input_filter_none(self) -> None:
        """Ordered pipe filters out None values returned by the pipe operation."""

        pipeline = (
            PipelineBuilder()
            .add_source(list(range(10)))
            .pipe(lambda x: None if x % 2 == 0 else x, output_order="input")
            .add_sink(2)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            result = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(result, [1, 3, 5, 7, 9])

    def test_pipeline_order_input_filter_none_async(self) -> None:
        """Ordered pipe filters out None values with async function."""

        async def filter_even(x):
            await asyncio.sleep(0.01)
            return None if x % 2 == 0 else x

        pipeline = (
            PipelineBuilder()
            .add_source(list(range(10)))
            .pipe(filter_even, output_order="input", concurrency=3)
            .add_sink(2)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            result = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(result, [1, 3, 5, 7, 9])

    def test_pipeline_order_input_filter_none_with_concurrency(self) -> None:
        """Ordered pipe filters out None values with high concurrency."""

        def slow_filter(x):
            time.sleep(0.05)
            return None if x % 3 == 0 else x

        pipeline = (
            PipelineBuilder()
            .add_source(list(range(15)))
            .pipe(slow_filter, output_order="input", concurrency=5)
            .add_sink(10)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            result = list(pipeline.get_iterator(timeout=30))
            # Filters out 0, 3, 6, 9, 12
            self.assertEqual(result, [1, 2, 4, 5, 7, 8, 10, 11, 13, 14])

    def test_pipeline_order_input_all_none(self) -> None:
        """Ordered pipe handles case where all values are None."""

        pipeline = (
            PipelineBuilder()
            .add_source(list(range(5)))
            .pipe(lambda _: None, output_order="input")
            .add_sink(2)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            result = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(result, [])

    def test_pipeline_order_input_mixed_none_and_values(self) -> None:
        """Ordered pipe correctly handles mixed None and values in specific pattern."""

        def pattern_filter(x):
            if x < 2:
                return None
            if x < 5:
                return x
            if x < 7:
                return None
            return x

        pipeline = (
            PipelineBuilder()
            .add_source(list(range(10)))
            .pipe(pattern_filter, output_order="input")
            .add_sink(5)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            result = list(pipeline.get_iterator(timeout=30))
            # Returns 2, 3, 4 (x < 5), and 7, 8, 9 (x >= 7)
            self.assertEqual(result, [2, 3, 4, 7, 8, 9])


################################################################################
# AsyncPipeline2
################################################################################


class TestPipelineNoop(unittest.TestCase):
    def test_pipeline_noop(self) -> None:
        """AsyncPipeline2 functions without pipe."""

        apl = PipelineBuilder().add_source(range(10)).add_sink(1).build(num_threads=1)

        with apl.auto_stop():
            for i in range(10):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=30))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=30)

        with self.assertRaises(EOFError):
            apl.get_item(timeout=30)


class TestPipelinePassthrough(unittest.TestCase):
    def test_pipeline_passthrough(self) -> None:
        """AsyncPipeline2 can passdown items operation."""

        apl = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1)
            .build(num_threads=1)
        )

        with apl.auto_stop():
            for i in range(10):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=30))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=30)

        with self.assertRaises(EOFError):
            apl.get_item(timeout=30)


class TestPipelineSkip(unittest.TestCase):
    def test_pipeline_skip(self) -> None:
        """AsyncPipeline2 does not output None items."""

        src = list(range(10))

        async def odd(i):
            if i % 2:
                return i

        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .pipe(odd)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for i in range(5):
                self.assertEqual(i * 2 + 1, pipeline.get_item(timeout=30))

            with self.assertRaises(EOFError):
                pipeline.get_item(timeout=30)


class TestPipelineLambda(unittest.TestCase):
    def test_pipeline_lambda(self) -> None:
        """AsyncPipeline2 pipe supports lambda items operation."""

        apl = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(lambda x: x)
            .add_sink(1)
            .build(num_threads=1)
        )

        with apl.auto_stop():
            for i in range(10):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=30))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=30)

        with self.assertRaises(EOFError):
            apl.get_item(timeout=30)


class TestPipelineSimple(unittest.TestCase):
    def test_pipeline_simple(self) -> None:
        """AsyncPipeline2 can perform simple operation."""
        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(adouble)
            .pipe(aplus1)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for i, item in enumerate(pipeline.get_iterator(timeout=30)):
                self.assertEqual(item, i * 2 + 1)


class TestPipelineAggregate(unittest.TestCase):
    def test_pipeline_aggregate(self) -> None:
        """AsyncPipeline aggregates the input"""

        src = list(range(13))

        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .aggregate(4)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(
                results, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12]]
            )

    def test_pipeline_aggregate_drop_last(self) -> None:
        """AsyncPipeline aggregates the input and drop the last"""

        src = list(range(13))

        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .aggregate(4, drop_last=True)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(results, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    @parameterized.expand([(False,), (True,)])
    def test_pipeline_aggregate_custom_op(self, drop_last: bool) -> None:
        """AsyncPipeline aggregates with custom operation that concatenates when threshold is exceeded"""

        # Custom aggregation: concatenate strings when total size exceeds threshold
        class CustomAggregator(Aggregator):
            def __init__(self, size_threshold: int = 10):
                self.size_threshold = size_threshold
                self.buffer: list[str] = []
                self.total_size = 0

            def _flush(self) -> str:
                result = "".join(self.buffer)
                self.buffer = []
                self.total_size = 0
                return result

            def flush(self) -> str | None:
                # Emit remaining buffer when EOF is reached
                if self.buffer:
                    return self._flush()
                return None  # _SKIP

            def accumulate(self, item: str) -> str | None:
                self.buffer.append(item)
                self.total_size += len(item)

                if self.total_size >= self.size_threshold:
                    return self._flush()
                return None  # _SKIP

        src = ["a", "bb", "ccc", "dddd", "e", "ff", "ggg", "h"]

        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .aggregate(CustomAggregator(size_threshold=10), drop_last=drop_last)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))
            # "a", "bb", "ccc", "dddd" = 10 chars -> first result
            if drop_last:
                # When drop_last=True, flush is not called,
                # so the remaining buffer ["e", "ff", "ggg", "h"] is dropped
                self.assertEqual(results, ["abbcccdddd"])
            else:
                # When drop_last=False, flush is called,
                # so remaining buffer is emitted: "e", "ff", "ggg", "h"
                self.assertEqual(results, ["abbcccdddd", "effgggh"])


class TestPipelineDisaggregate(unittest.TestCase):
    def test_pipeline_disaggregate(self) -> None:
        """AsyncPipeline disaggregates the input"""

        src = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12]]

        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .disaggregate()
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))
        self.assertEqual(results, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


class TestPipelineSource(unittest.TestCase):
    def test_pipeline_source_failure(self) -> None:
        """AsyncPipeline continues when source fails.

        note: the front end will propagate the error at the end of the `stop`.
        before that, the pipeline should continue functioning.
        """

        def failing_range(i):
            yield from range(i)
            raise ValueError("Iterator failed")

        pipeline = (
            PipelineBuilder()
            .add_source(failing_range(10))
            .pipe(adouble)
            .pipe(aplus1)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                results = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(results, [1 + 2 * i for i in range(10)])


class TestPipelineType(unittest.TestCase):
    def test_pipeline_type_error(self) -> None:
        """AsyncPipeline immediately fails if pipe function has wrong signature"""

        async def wrong_sig(i, _):
            return i

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            # pyrefly: ignore [bad-argument-type]
            .pipe(wrong_sig)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(vals, [])


class TestPipelineTask(unittest.TestCase):
    def test_pipeline_task_failure(self) -> None:
        """AsyncPipeline is robust against task-level failure."""

        async def areject_m3(i):
            if i % 3 == 0:
                raise ValueError(f"Multiple of 3 is prohibited: {i}")
            return i

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(areject_m3)
            .pipe(adouble)
            .pipe(aplus1)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(results, [1 + 2 * i for i in range(10) if i % 3])


class TestPipelineCancel(unittest.TestCase):
    def test_pipeline_cancel_empty(self) -> None:
        """AsyncPipeline2 can be cancelled while it's blocked on the pipeline."""

        apl = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1)
            .build(num_threads=1)
        )
        # The nuffer and coroutinues will be blocked holding items as follows:
        #
        # [src] --|queue(1)|--> [passthrough] --|queue(1)|--> [sink] -->|queue(1)|
        #   i+5        i+4           i+3            i+2         i+1        i
        #

        with apl.auto_stop():
            for i in range(5):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=30))

            # Ensure that buffers are filled and the pipeline is blocked.
            time.sleep(0.1)
            # At this point, the output queue holds 5.

        # Only the "5" is retrievable.
        self.assertEqual(5, apl.get_item(timeout=30))

        # The background thread is stopped, so no more data is coming.
        for _ in range(3):
            with self.assertRaises(EOFError):
                apl.get_item(timeout=30)


class TestPipelineFail(unittest.TestCase):
    def test_pipeline_fail_middle(self) -> None:
        """When a stage in the middle fails, downstream stages are not failing."""

        async def fail(i, _):
            return i

        class PassthroughWithCache:
            def __init__(self):
                self.cache = []

            async def __call__(self, i):
                self.cache.append(i)
                return i

        pwc = PassthroughWithCache()

        apl = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            # pyrefly: ignore [bad-argument-type]
            .pipe(fail)
            .pipe(pwc)
            .add_sink(1)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure):
            with apl.auto_stop():
                with self.assertRaises(EOFError):
                    apl.get_item(timeout=30)

        self.assertEqual(pwc.cache, [])

        # The background thread is stopped, and the output queue is empty.
        for _ in range(3):
            with self.assertRaises(EOFError):
                apl.get_item(timeout=30)


class TestPipelineEof(unittest.TestCase):
    def test_pipeline_eof_stop(self) -> None:
        """APL2 can be closed after reaching EOF."""
        apl = (
            PipelineBuilder()
            .add_source(range(2))
            .pipe(passthrough)
            .add_sink(1000)
            .build(num_threads=1)
        )
        with apl.auto_stop():
            for i in range(2):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=30))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=30)


class TestPipelineIterator(unittest.TestCase):
    def test_pipeline_iterator(self) -> None:
        """Can iterate the pipeline."""

        apl = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(passthrough)
            .add_sink(1)
            .build(num_threads=1)
        )

        with apl.auto_stop():
            for i, item in enumerate(apl.get_iterator(timeout=30)):
                print(i, item)
                self.assertEqual(i, item)


class TestPipelineIter(unittest.TestCase):
    def test_pipeline_iter_and_next(self) -> None:
        """Pipeline `iter` and `next` fulfill the following contracts

        1. `iter(pipeline)` creates a new iterator.
        2. `next(iterator)` gives the next item in the pipeline.
        3. one ca call `next` multiple times on an iterator.
        4. If iterator is exhausted, it raises `StopIteration`.
        5. Iterator object can be discarded without an side-effect by itself.
           (Other side-effects might be happening but it's independent from iterator)
        6. Multiple instances of iterators can be created by
           repeatedly calling `iter(pipeline)`.
        7. We do not define the beheviors for multiple iterators exist at the same
           time. We only consider the case where one iterator is created and
           discarded, then another is created.
        8. A new iterator should return items that are sequel to items generated by
           the previous iterator.
        """

        apl = PipelineBuilder().add_source(range(12)).add_sink(1).build(num_threads=1)

        with apl.auto_stop():
            iterator = iter(apl)
            self.assertEqual(next(iterator), 0)
            self.assertEqual(next(iterator), 1)
            self.assertEqual(next(iterator), 2)

            iterator = iter(apl)
            self.assertEqual(next(iterator), 3)
            self.assertEqual(next(iterator), 4)
            self.assertEqual(next(iterator), 5)

            iterator = iter(apl)
            self.assertEqual(next(iterator), 6)
            self.assertEqual(next(iterator), 7)
            self.assertEqual(next(iterator), 8)

            iterator = iter(apl)
            self.assertEqual(next(iterator), 9)
            self.assertEqual(next(iterator), 10)
            self.assertEqual(next(iterator), 11)

            iterator = iter(apl)
            with self.assertRaises(StopIteration):
                next(iterator)


class TestPipelineStuck(unittest.TestCase):
    def test_pipeline_stuck(self) -> None:
        """`get_item` waits for slow pipeline."""

        async def delay(i):
            print(f"Sleeping: {i}")
            await asyncio.sleep(0.5)
            print(f"Sleeping: {i} - done")
            return i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(delay)
            .add_sink(1)
            .build(num_threads=1)
        )

        with apl.auto_stop():
            for i, item in enumerate(apl.get_iterator(timeout=30)):
                print(i, item)
                self.assertEqual(i, item)


class TestPipelinePipe(unittest.TestCase):
    def test_pipeline_pipe_agen(self) -> None:
        """pipe works with async generator function"""

        async def dup_increment(v):
            for i in range(3):
                yield v + i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            .build(num_threads=1)
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(expected, output)

    def test_pipeline_pipe_sync_gen(self) -> None:
        """pipe works with sync generator function"""

        def dup_increment(v):
            for i in range(3):
                yield v + i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            .build(num_threads=1)
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(expected, output)


class TestCallableGenerator(unittest.TestCase):
    def test_callable_generator(self) -> None:
        """pipe works with sync callable class returning generator"""

        class DupIncrement:
            def __init__(self) -> None:
                pass

            def __call__(self, v: int) -> Iterator[int]:
                for i in range(3):
                    yield v + i

        dup_increment = DupIncrement()

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            .build(num_threads=1)
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(expected, output)

    def test_pipeline_pipe_agen_max_failures(self) -> None:
        """pipe works with async generator function and max_failure"""

        async def dup_increment(v):
            for i in range(3):
                yield v + i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            .build(num_threads=1, max_failures=1)
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(output, expected)

    def test_pipeline_pipe_gen(self) -> None:
        """pipe works with sync generator function"""

        def dup_increment(v):
            for i in range(3):
                yield v + i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            .build(num_threads=1)
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(output, expected)

    def test_pipeline_pipe_gen_max_failures(self) -> None:
        """pipe works with sync generator function and max_failure"""

        def dup_increment(v):
            for i in range(3):
                yield v + i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            .build(num_threads=1, max_failures=1)
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(output, expected)

    @unittest.skipIf(
        platform.system() == "Darwin" and "CI" in os.environ,
        reason="GitHub macOS CI is not timely enough.",
    )
    def test_pipeline_pipe_gen_incremental(self) -> None:
        """pipe returns output of generator function immediately if not in ProcessPoolExecutor"""

        # We introduce delay in each iteration, so that, if the pipeline is returning
        # the yielded value immediately, the output will be obtained quickly.
        # If the pipeline is not returning the yielded value immediately, the output
        # won't be available until the iteration ends, and by that time
        # the foreground pipeline should timeout.
        def dup_increment(v):
            for i in range(3):
                time.sleep(0.1)
                yield v + i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            .build(num_threads=1)
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(output, expected)

    def test_pipeline_pipe_agen_wrong_hook(self) -> None:
        """pipe works with async generator function, even when hook abosrb the StopAsyncIteration"""

        class _Hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self, input_item=None):
                try:
                    yield
                except StopAsyncIteration:
                    pass

        async def dup_increment(v):
            for i in range(3):
                yield v + i

        apl = (
            PipelineBuilder()
            .add_source(range(3))
            .pipe(dup_increment)
            .add_sink(1)
            # pyre-ignore[6]
            .build(num_threads=1, task_hook_factory=lambda _: [_Hook()])
        )

        expected = [0, 1, 2, 1, 2, 3, 2, 3, 4]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(output, expected)

    def test_pipeline_source_agen(self) -> None:
        """source works with async generator function"""

        async def source():
            for i in range(3):
                yield i

        apl = PipelineBuilder().add_source(source()).add_sink(1).build(num_threads=1)

        expected = [0, 1, 2]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=30))
        self.assertEqual(output, expected)


class TestPipelineStart(unittest.TestCase):
    def test_pipeline_start_multiple_times(self) -> None:
        """`Pipeline.start` cannot be called multiple times."""

        pipeline = (
            PipelineBuilder().add_source(range(10)).add_sink(1).build(num_threads=1)
        )

        with pipeline.auto_stop():
            with self.assertRaises(RuntimeError):
                pipeline.start()


class TestPipelineStop(unittest.TestCase):
    def test_pipeline_stop_multiple_times(self) -> None:
        """`Pipeline.stop` can be called multiple times."""

        pipeline = (
            PipelineBuilder().add_source(range(10)).add_sink(1).build(num_threads=1)
        )

        pipeline.stop()
        pipeline.stop()
        pipeline.stop()

        with pipeline.auto_stop():
            pipeline.stop()
            pipeline.stop()
            pipeline.stop()

        pipeline.stop()
        pipeline.stop()
        pipeline.stop()


def _run_pipeline_without_closing():
    pipeline = PipelineBuilder().add_source(range(10)).add_sink(1).build(num_threads=1)
    pipeline.start()


def get_pid(_):
    import os

    time.sleep(0.5)
    pid = os.getpid()
    print(f"{pid=}")
    return pid


def _range(item):
    print(item)
    for i in range(item):
        print(f"yielding {item} - {i}")
        yield i


class TestPipelineNo(unittest.TestCase):
    def test_pipeline_no_close(self) -> None:
        """Python interpreter can terminate even when Pipeline is not explicitly closed."""

        p = Process(target=_run_pipeline_without_closing)
        p.start()
        p.join(timeout=10)

        if p.exitcode is None:
            p.kill()
            raise RuntimeError("Process did not self-terminate.")


class TestPipelineCustom(unittest.TestCase):
    def test_pipeline_custom_pipe_executor(self) -> None:
        """`pipe` accepts custom ThreadPoolExecutor.

        The primal goal of custom executor is to make it easy to use
        thread local storages.

        So in this test, we initialize a custom executor with some thread
        local storages, and then we access it without any check (hasattr)
        in pipe function.
        """
        num_threads = 10

        ref = set(range(num_threads))

        ref_copy = ref.copy()
        thread_local_storage = threading.local()

        def init_storage():
            print("Initializing thread:", threading.get_ident())
            thread_local_storage.value = ref_copy.pop()

        executor = ThreadPoolExecutor(
            max_workers=num_threads,
            initializer=init_storage,
        )

        # Block every op until all `num_threads` of them are running at once, forcing the
        # pool to actually use all its threads (the point of this test). If they did not run
        # concurrently the barrier would never reach its count and wait() would raise
        # BrokenBarrierError, failing the pipeline -- deterministic, no wall-clock timing.
        barrier = threading.Barrier(num_threads)

        def op(i: int) -> int:
            barrier.wait(timeout=30)
            print(i, thread_local_storage.value)
            return thread_local_storage.value

        pipeline = (
            PipelineBuilder()
            .add_source(range(num_threads))
            .pipe(op, executor=executor, concurrency=num_threads)
            .add_sink(1)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(0, len(ref_copy))
        self.assertEqual(ref, set(vals))

    def test_pipeline_custom_pipe_executor_process(self) -> None:
        """`pipe` accepts custom ProcessPoolExecutor."""
        num_processes = 5

        executor = ProcessPoolExecutor(max_workers=num_processes)

        pipeline = (
            PipelineBuilder()
            .add_source(range(num_processes))
            .pipe(get_pid, executor=executor, concurrency=num_processes)
            .add_sink(1)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(num_processes, len(set(vals)))

    def test_pipeline_custom_pipe_executor_process_generator(self) -> None:
        """`pipe` accepts custom ProcessPoolExecutor and generator function."""
        executor = ProcessPoolExecutor(max_workers=1)

        pipeline = (
            PipelineBuilder()
            .add_source(range(4))
            .pipe(_range, executor=executor, concurrency=1)
            .add_sink(1)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))
        self.assertEqual([0, 0, 1, 0, 1, 2], vals)

    def test_pipeline_custom_pipe_executor_async(self) -> None:
        """pipe rejects custom executor if op is async"""

        async def op(i: int) -> int:
            return i

        with self.assertRaises(ValueError):
            PipelineBuilder().add_source(range(10)).pipe(
                op, executor=ThreadPoolExecutor()
            ).add_sink(1).build(num_threads=1)

    def test_pipeline_pipe_list(self) -> None:
        """pipe supports list as op."""

        op = [i + 1 for i in range(10)]

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            # pyre-ignore[6]
            .pipe(op)
            .add_sink(1)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(op, vals)

    def test_pipeline_pipe_tuple(self) -> None:
        """pipe supports list as op."""

        op = tuple(i + 1 for i in range(10))

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            # pyre-ignore[6]
            .pipe(op)
            .add_sink(1)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual([i + 1 for i in range(10)], vals)

    def test_pipeline_pipe_dict(self) -> None:
        """pipe supports dict as op."""

        op = {i: i + 1 for i in range(10)}

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            # pyre-ignore[6]
            .pipe(op)
            .add_sink(1)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual([i + 1 for i in range(10)], vals)


class _PicklableSource:
    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> Iterator[int]:
        yield from range(self.n)


class _ValidatePipelineId:
    def __init__(self, val: int) -> None:
        self.val = val

    def __iter__(self) -> Iterator[int]:
        if (v := _get_global_id()) != self.val:
            raise AssertionError(f"_node._PIPELINE_ID={v} != {self.val=}")
        yield 0


def plusN(x: int, N: int) -> int:
    return x + N


def _sync_double(x: int) -> int:
    return 2 * x


def _route_zero(_: int) -> int:
    return 0


def _raise_value_error(_: int) -> int:
    raise ValueError("boom")


def _sync_double_gen(x: int):
    yield 2 * x
    yield 2 * x + 1


def _noop_initializer(_: int) -> None:
    pass


class _FakeInterpreterWorkerContext:
    """Mimics an InterpreterPoolExecutor worker context (carries ``initdata``)."""

    def __init__(self, initdata: object) -> None:
        self.initdata = initdata


class _FakeInterpreterPoolExecutor:
    """Stub with the attributes ``_interpreter_pool_kwargs`` reads (3.14 layout).

    Lets the recovery logic be tested on any Python version without a real
    InterpreterPoolExecutor (3.14+ only).
    """

    _max_workers = 3
    _thread_name_prefix = "interp"

    def __init__(self, initdata: object) -> None:
        self._initdata = initdata

    def _create_worker_context(self) -> _FakeInterpreterWorkerContext:
        return _FakeInterpreterWorkerContext(self._initdata)


def _failing_initializer() -> None:
    raise ValueError("init boom")


def hook_factory(_: StageInfo) -> list[TaskHook]:
    return [CountHook()]


class TestPipelinebuilderPicklable(unittest.TestCase):
    @_ignore_warnings(_RUN_PIPELINE_DEPRECATION, _FORK_WARNING, _UNAWAITED_COROUTINE)
    def test_pipelinebuilder_picklable(self) -> None:
        """PipelineBuilder can be passed to subprocess (==picklable)"""

        builder = (
            PipelineBuilder()
            .add_source(_PicklableSource(10))
            .pipe(
                adouble,
                concurrency=3,
            )
            .pipe(
                aplus1,
                concurrency=3,
            )
            .pipe(partial(plusN, N=3))
            .pipe(passthrough)
            .aggregate(3)
            .disaggregate()
            .add_sink(10)
        )

        results = list(
            run_pipeline_in_subprocess(
                # pyre-ignore[6]
                builder,
                num_threads=5,
                buffer_size=-1,
                task_hook_factory=hook_factory,
            )
        )

        def _ref(x: int) -> int:
            return 2 * x + 1 + 3

        self.assertEqual([_ref(i) for i in range(10)], sorted(results))


class TestFailureCounter(unittest.TestCase):
    def test_failure_counter_global_countes(self) -> None:
        """_get_fail_counter creates _FailCounter subclass with different class valiable"""

        FC1 = _get_fail_counter()
        FC2 = _get_fail_counter()

        fc1_1 = FC1(-1, -1)
        fc1_2 = FC1(-1, -1)

        fc2_1 = FC2(-1, -1)
        fc2_2 = FC2(-1, -1)

        self.assertEqual(0, _FailCounter._num_global_failures)
        self.assertEqual(0, FC1._num_global_failures)
        self.assertEqual(0, FC2._num_global_failures)
        self.assertTrue(fc1_1._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc1_2._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc2_1._num_global_failures is FC2._num_global_failures)
        self.assertTrue(fc2_2._num_global_failures is FC2._num_global_failures)
        self.assertEqual(0, fc1_1._num_global_failures)
        self.assertEqual(0, fc1_2._num_global_failures)
        self.assertEqual(0, fc1_1._num_stage_failures)
        self.assertEqual(0, fc1_2._num_stage_failures)
        self.assertEqual(0, fc2_1._num_global_failures)
        self.assertEqual(0, fc2_2._num_global_failures)
        self.assertEqual(0, fc2_1._num_stage_failures)
        self.assertEqual(0, fc2_2._num_stage_failures)

        fc1_1.__class__._num_global_failures += 1
        fc1_1._num_stage_failures += 1

        self.assertEqual(0, _FailCounter._num_global_failures)
        self.assertEqual(1, FC1._num_global_failures)
        self.assertEqual(0, FC2._num_global_failures)
        self.assertTrue(fc1_1._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc1_2._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc2_1._num_global_failures is FC2._num_global_failures)
        self.assertTrue(fc2_2._num_global_failures is FC2._num_global_failures)
        self.assertEqual(1, fc1_1._num_global_failures)
        self.assertEqual(1, fc1_2._num_global_failures)
        self.assertEqual(1, fc1_1._num_stage_failures)
        self.assertEqual(0, fc1_2._num_stage_failures)
        self.assertEqual(0, fc2_1._num_global_failures)
        self.assertEqual(0, fc2_2._num_global_failures)
        self.assertEqual(0, fc2_1._num_stage_failures)
        self.assertEqual(0, fc2_2._num_stage_failures)

        fc1_1.__class__._num_global_failures += 1
        fc1_1._num_stage_failures += 1

        self.assertEqual(0, _FailCounter._num_global_failures)
        self.assertEqual(2, FC1._num_global_failures)
        self.assertEqual(0, FC2._num_global_failures)
        self.assertTrue(fc1_1._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc1_2._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc2_1._num_global_failures is FC2._num_global_failures)
        self.assertTrue(fc2_2._num_global_failures is FC2._num_global_failures)
        self.assertEqual(2, fc1_1._num_global_failures)
        self.assertEqual(2, fc1_2._num_global_failures)
        self.assertEqual(2, fc1_1._num_stage_failures)
        self.assertEqual(0, fc1_2._num_stage_failures)
        self.assertEqual(0, fc2_1._num_global_failures)
        self.assertEqual(0, fc2_2._num_global_failures)
        self.assertEqual(0, fc2_1._num_stage_failures)
        self.assertEqual(0, fc2_2._num_stage_failures)

        fc1_2.__class__._num_global_failures += 1
        fc1_2._num_stage_failures += 1

        self.assertEqual(0, _FailCounter._num_global_failures)
        self.assertEqual(3, FC1._num_global_failures)
        self.assertEqual(0, FC2._num_global_failures)
        self.assertTrue(fc1_1._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc1_2._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc2_1._num_global_failures is FC2._num_global_failures)
        self.assertTrue(fc2_2._num_global_failures is FC2._num_global_failures)
        self.assertEqual(3, fc1_1._num_global_failures)
        self.assertEqual(3, fc1_2._num_global_failures)
        self.assertEqual(2, fc1_1._num_stage_failures)
        self.assertEqual(1, fc1_2._num_stage_failures)
        self.assertEqual(0, fc2_1._num_global_failures)
        self.assertEqual(0, fc2_2._num_global_failures)
        self.assertEqual(0, fc2_1._num_stage_failures)
        self.assertEqual(0, fc2_2._num_stage_failures)

        fc2_1.__class__._num_global_failures += 1
        fc2_1._num_stage_failures += 1

        self.assertEqual(0, _FailCounter._num_global_failures)
        self.assertEqual(3, FC1._num_global_failures)
        self.assertEqual(1, FC2._num_global_failures)
        self.assertTrue(fc1_1._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc1_2._num_global_failures is FC1._num_global_failures)
        self.assertTrue(fc2_1._num_global_failures is FC2._num_global_failures)
        self.assertTrue(fc2_2._num_global_failures is FC2._num_global_failures)
        self.assertEqual(3, fc1_1._num_global_failures)
        self.assertEqual(3, fc1_2._num_global_failures)
        self.assertEqual(2, fc1_1._num_stage_failures)
        self.assertEqual(1, fc1_2._num_stage_failures)
        self.assertEqual(1, fc2_1._num_global_failures)
        self.assertEqual(1, fc2_2._num_global_failures)
        self.assertEqual(1, fc2_1._num_stage_failures)
        self.assertEqual(0, fc2_2._num_stage_failures)


class TestPipelineMax(unittest.TestCase):
    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_max_failures(self, output_order: str) -> None:
        """max_failures stop the pipeline."""

        def fail_odd(x):
            if x % 2:
                raise ValueError(f"Only evan numbers are allowed. {x}")
            return x

        builder = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(fail_odd, output_order=output_order)
            .add_sink(1)
        )

        pipeline = builder.build(num_threads=1)
        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual([0, 2, 4, 6, 8], vals)

        pipeline = builder.build(num_threads=1, max_failures=3)
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))

        self.assertEqual([0, 2, 4, 6], vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_max_failures_multiple_pipeline(self, output_order: str) -> None:
        """When using multiple pipelines with different error caps, they work

        Note: FailCounter uses class method to combines the errors
        from the different stages. We create a different child class
        for each pipeline construction so that class counter are separate for
        each pipeline object. This test ensures that.
        """

        def fail_odd(x):
            if x % 2:
                raise ValueError(f"Only evan numbers are allowed. {x}")
            return x

        src = range(10)

        builder = (
            PipelineBuilder()
            .add_source(src)
            .pipe(fail_odd, output_order=output_order)
            .add_sink(1)
        )

        pipeline1 = builder.build(num_threads=1, max_failures=2)
        pipeline2 = builder.build(num_threads=1, max_failures=3)

        with self.assertRaises(PipelineFailure):
            with pipeline2.auto_stop():
                vals = list(pipeline2.get_iterator(timeout=30))

        self.assertEqual([0, 2, 4, 6], vals)

        with self.assertRaises(PipelineFailure):
            with pipeline1.auto_stop():
                vals = list(pipeline1.get_iterator(timeout=30))

        self.assertEqual([0, 2, 4], vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_max_failures_pipe_override_strict(
        self, output_order: str
    ) -> None:
        """max_failures at pipe overrides the global threshold."""

        def fail_odd(x):
            if x % 2:
                raise ValueError(f"Only evan numbers are allowed. {x}")
            return x

        builder = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(fail_odd, output_order=output_order, max_failures=2)
            .add_sink(1)
        )

        pipeline = builder.build(num_threads=1, max_failures=-1)
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))
        self.assertEqual([0, 2, 4], vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_max_failures_pipe_override_loose(self, output_order: str) -> None:
        """max_failures at pipe overrides the global threshold."""

        def fail_odd(x):
            if x % 2:
                raise ValueError(f"Only evan numbers are allowed. {x}")
            return x

        builder = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(fail_odd, output_order=output_order, concurrency=3, max_failures=-1)
            .add_sink(1)
        )

        pipeline = builder.build(num_threads=1, max_failures=2)
        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=30))
        self.assertEqual([0, 2, 4, 6, 8], vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_max_failures_pipe_override_multiple(
        self, output_order: str
    ) -> None:
        """max_failures at pipe overrides the global threshold."""

        # Remove odd values
        def fail_odd(x):
            if x % 2:
                raise ValueError(f"Only evan numbers are allowed. {x}")
            return x

        # Remove multiplier of 6s
        def fail_six(x):
            if (x % 6) == 0:
                raise ValueError(f"Values divisible by 6 are not allowed. {x}")
            return x

        builder = (
            PipelineBuilder()
            .add_source(range(20))
            .pipe(fail_odd, output_order=output_order, max_failures=-1)
            .pipe(fail_six, output_order=output_order, max_failures=3)
            .add_sink(1)
        )

        # fail_odd fails more often, but it is allowed to fail any number of times.
        # fail_six fails less often, but at the fourth failure (18),
        # it should shutdown the pipeline.

        pipeline = builder.build(num_threads=1, max_failures=2)
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))
        self.assertEqual([2, 4, 8, 10, 14, 16], vals)


class TestPipelinePropagate(unittest.TestCase):
    def test_pipeline_propagate_source_failure(self) -> None:
        """When source itrator fails, the exception is propagated to the front end"""

        def failure_source():
            raise RuntimeError("Foo")
            yield None

        pipeline = (
            PipelineBuilder()
            .add_source(failure_source())
            .add_sink()
            .build(num_threads=1)
        )
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                # Consume so the source actually runs; otherwise surfacing its failure
                # races against auto_stop() cancelling the not-yet-run source task.
                list(pipeline.get_iterator(timeout=30))


class TestIterableWithShuffle:
    __test__ = False

    def __init__(self, n: int) -> None:
        self.vals = list(range(n))
        self.seed = 0

    def shuffle(self, *, seed: int) -> None:
        self.vals = self.vals[1:] + self.vals[:1]
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        yield from self.vals


class TestRunPipeline(unittest.TestCase):
    @_ignore_warnings(_RUN_PIPELINE_DEPRECATION, _FORK_WARNING, _UNAWAITED_COROUTINE)
    def test_run_pipeline_in_subprocess_state(self) -> None:
        """The status of the source is maintained and propagated properly in subprocess"""
        n = 5

        # pyre-ignore[6]
        src = embed_shuffle(TestIterableWithShuffle(n))
        builder = PipelineBuilder().add_source(src).add_sink()
        # pyre-ignore[6]
        iterable = run_pipeline_in_subprocess(builder, num_threads=1)

        self.assertEqual([1, 2, 3, 4, 0], list(iterable))
        self.assertEqual([2, 3, 4, 0, 1], list(iterable))
        self.assertEqual([3, 4, 0, 1, 2], list(iterable))

        # since the src is copied to the subprocess iterating it yields the original state

        self.assertEqual([1, 2, 3, 4, 0], list(src))
        # pyre-ignore[16]
        self.assertEqual(0, src.src.seed)
        self.assertEqual([2, 3, 4, 0, 1], list(src))
        self.assertEqual(1, src.src.seed)
        self.assertEqual([3, 4, 0, 1, 2], list(src))
        self.assertEqual(2, src.src.seed)

    @_ignore_warnings(_RUN_PIPELINE_DEPRECATION, _FORK_WARNING, _UNAWAITED_COROUTINE)
    def test_run_pipeline_in_subprocess_pipeline_id(self) -> None:
        """The pipeline construdted in a subprocess inherits the global ID from the main process"""

        # Set to a number that's not zero and something unlikely to happen during the testing
        _set_global_id(123456)
        ref = _get_global_id() + 1

        builder = PipelineBuilder().add_source(_ValidatePipelineId(ref)).add_sink()

        # pyre-ignore[6]
        iterable = run_pipeline_in_subprocess(builder, num_threads=1)

        for _ in iterable:
            pass


def _interpreter_pool_available() -> bool:
    if sys.version_info < (3, 14):
        return False
    try:
        from concurrent.futures.interpreter import InterpreterPoolExecutor  # noqa: F401
    except ImportError:
        return False
    return True


def _config_with_executor(executor) -> PipelineConfig:
    return (
        PipelineBuilder()
        .add_source(_PicklableSource(10))
        .pipe(_sync_double, executor=executor)
        .add_sink(10)
        .get_config()
    )


def _pipe_executor(pipe):
    return cast(PipeConfig, pipe)._args.executor


def _first_pipe_executor(config: PipelineConfig):
    return _pipe_executor(config.pipes[0])


class TestExecutorProxy(unittest.TestCase):
    """Surgery that makes stdlib executors in a config picklable for subprocess."""

    def test_proxy_submit_after_shutdown_raises(self) -> None:
        """submit() after shutdown() raises, even when the executor was never built."""
        proxy = _ExecutorProxy(
            ThreadPoolExecutor,
            {
                "max_workers": 2,
                "thread_name_prefix": "",
                "initializer": None,
                "initargs": (),
            },
        )
        # Shut down before any submit, so the underlying executor was never constructed.
        proxy.shutdown()
        with self.assertRaises(RuntimeError):
            proxy.submit(_sync_double, 1)
        # Nothing should have been constructed by the rejected submit.
        self.assertIsNone(proxy._executor)

    def test_interpreter_pool_kwargs_recovers_initializer(self) -> None:
        """initializer/initargs are recovered from an interpreter pool's initdata."""
        executor = _FakeInterpreterPoolExecutor(initdata=(_noop_initializer, (7,), {}))
        kwargs = _interpreter_pool_kwargs(executor)
        self.assertEqual(kwargs["max_workers"], 3)
        self.assertEqual(kwargs["thread_name_prefix"], "interp")
        self.assertIs(kwargs["initializer"], _noop_initializer)
        self.assertEqual(kwargs["initargs"], (7,))

    def test_interpreter_pool_kwargs_without_initializer(self) -> None:
        """With no initializer (initdata is None), no initializer kwargs are emitted."""
        executor = _FakeInterpreterPoolExecutor(initdata=None)
        kwargs = _interpreter_pool_kwargs(executor)
        self.assertEqual(kwargs, {"max_workers": 3, "thread_name_prefix": "interp"})

    def test_proxy_pickle_roundtrip(self) -> None:
        """An `_ExecutorProxy` survives pickling and builds the right executor lazily."""
        proxy = _ExecutorProxy(
            ThreadPoolExecutor,
            {
                "max_workers": 3,
                "thread_name_prefix": "",
                "initializer": None,
                "initargs": (),
            },
        )
        restored = pickle.loads(pickle.dumps(proxy))
        try:
            self.assertIsInstance(restored, _ExecutorProxy)
            self.assertIs(restored._executor_class, ThreadPoolExecutor)
            self.assertEqual(restored._kwargs["max_workers"], 3)
            # The underlying executor is built lazily on first use.
            self.assertEqual(restored.submit(_sync_double, 21).result(), 42)
            self.assertIsInstance(restored._executor, ThreadPoolExecutor)
            self.assertEqual(restored._executor._max_workers, 3)
        finally:
            restored.shutdown()

    def test_threadpool_replaced_and_original_untouched(self) -> None:
        """ThreadPoolExecutor on a pipe is replaced; the input config is not mutated."""
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            config = _config_with_executor(executor)
            new_config = _make_config_executors_picklable(config)

            proxy = _first_pipe_executor(new_config)
            self.assertIsInstance(proxy, _ExecutorProxy)
            self.assertIs(proxy._executor_class, ThreadPoolExecutor)
            self.assertEqual(proxy._kwargs["max_workers"], 2)
            # Original config is left untouched.
            self.assertIs(_first_pipe_executor(config), executor)
            # The whole rewritten config is now picklable.
            restored = pickle.loads(pickle.dumps(new_config))
            restored_proxy = _first_pipe_executor(restored)
            self.assertIsInstance(restored_proxy, _ExecutorProxy)
            self.assertIs(restored_proxy._executor_class, ThreadPoolExecutor)
            self.assertEqual(restored_proxy._kwargs["max_workers"], 2)
        finally:
            executor.shutdown()

    def test_threadpool_initializer_preserved(self) -> None:
        """A ThreadPoolExecutor's initializer/initargs survive surgery and pickling.

        Guards the version-specific extraction: Python 3.14 moved these off the executor's
        ``_initializer``/``_initargs`` attributes into a worker-context factory.
        """
        executor = ThreadPoolExecutor(
            max_workers=2, initializer=_noop_initializer, initargs=(7,)
        )
        try:
            config = _config_with_executor(executor)
            new_config = _make_config_executors_picklable(config)
            proxy = _first_pipe_executor(new_config)
            self.assertIs(proxy._kwargs["initializer"], _noop_initializer)
            self.assertEqual(proxy._kwargs["initargs"], (7,))
            # The recovered initializer survives the pickle round-trip too.
            restored_proxy = pickle.loads(pickle.dumps(proxy))
            self.assertIs(restored_proxy._kwargs["initializer"], _noop_initializer)
            self.assertEqual(restored_proxy._kwargs["initargs"], (7,))
        finally:
            executor.shutdown()

    def test_priority_executor_left_untouched(self) -> None:
        """Already-picklable executors (e.g. Priority*) pass through unchanged."""
        pool = PriorityThreadPoolExecutor(max_workers=2)
        try:
            entrypoint = pool.get_executor()
            config = _config_with_executor(entrypoint)
            new_config = _make_config_executors_picklable(config)
            self.assertIs(new_config, config)
            self.assertIs(_first_pipe_executor(new_config), entrypoint)
        finally:
            pool.shutdown()

    def test_no_executor_returns_same_config(self) -> None:
        """A config with no custom executor is returned unchanged."""
        config = (
            PipelineBuilder()
            .add_source(_PicklableSource(10))
            .pipe(_sync_double)
            .add_sink(10)
            .get_config()
        )
        self.assertIs(_make_config_executors_picklable(config), config)

    def test_nested_path_variants_replaced(self) -> None:
        """Executors inside PathVariants paths are rewritten too."""
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            config = PipelineConfig(
                src=SourceConfig(_PicklableSource(10)),
                pipes=[
                    PathVariants(
                        router=_route_zero,
                        paths=[[Pipe(_sync_double, executor=executor)]],
                    ),
                ],
                sink=SinkConfig(buffer_size=10),
            )
            new_config = _make_config_executors_picklable(config)
            nested_pipe = cast(PathVariantsConfig, new_config.pipes[0]).paths[0][0]
            self.assertIsInstance(_pipe_executor(nested_pipe), _ExecutorProxy)
            # Original untouched.
            orig_pipe = cast(PathVariantsConfig, config.pipes[0]).paths[0][0]
            self.assertIs(_pipe_executor(orig_pipe), executor)
        finally:
            executor.shutdown()

    def test_nested_merge_replaced(self) -> None:
        """Executors inside Merge sub-configs are rewritten too."""
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            plc = _config_with_executor(executor)
            config = PipelineConfig(
                src=Merge([plc]),
                pipes=[],
                sink=SinkConfig(buffer_size=10),
            )
            new_config = _make_config_executors_picklable(config)
            nested = cast(MergeConfig, new_config.src).pipeline_configs[0]
            self.assertIsInstance(_first_pipe_executor(nested), _ExecutorProxy)
        finally:
            executor.shutdown()

    @_ignore_warnings(_FORK_WARNING, _UNAWAITED_COROUTINE)
    def test_run_in_subprocess_with_threadpool(self) -> None:
        """run_pipeline_in_subprocess works with a ThreadPoolExecutor on a pipe."""
        config = _config_with_executor(ThreadPoolExecutor(max_workers=2))
        results = list(run_pipeline_in_subprocess(config, num_threads=1))
        self.assertEqual(sorted(results), [2 * i for i in range(10)])

    @unittest.skipUnless(
        _interpreter_pool_available(),
        "InterpreterPoolExecutor requires Python 3.14+",
    )
    @_ignore_warnings(_FORK_WARNING, _UNAWAITED_COROUTINE)
    def test_run_in_subprocess_with_interpreterpool(self) -> None:
        """run_pipeline_in_subprocess works with an InterpreterPoolExecutor on a pipe."""
        import importlib

        interpreter_mod = importlib.import_module("concurrent.futures.interpreter")
        executor = interpreter_mod.InterpreterPoolExecutor(max_workers=2)

        config = _config_with_executor(executor)
        results = list(run_pipeline_in_subprocess(config, num_threads=1))
        self.assertEqual(sorted(results), [2 * i for i in range(10)])

    def test_used_thread_pool_is_rejected(self) -> None:
        """A ThreadPoolExecutor that already ran work is rejected (must be freshly built)."""
        tpe = ThreadPoolExecutor(max_workers=2)
        try:
            tpe.submit(_sync_double, 1).result(timeout=30)  # spawns a worker thread
            with self.assertRaises(ValueError):
                _make_config_executors_picklable(_config_with_executor(tpe))
        finally:
            tpe.shutdown()


class TestHoistProcessPools(unittest.TestCase):
    """run_pipeline_in_subprocess hoists ProcessPoolExecutor workers into the main process."""

    def test_hoist_replaces_with_remote_executor(self) -> None:
        """A stdlib ProcessPoolExecutor is replaced by a _RemoteExecutor; original untouched."""
        ppe = ProcessPoolExecutor(max_workers=2)
        try:
            config = _config_with_executor(ppe)
            new_config, pools = _hoist_process_pools(config)
            try:
                self.assertEqual(len(pools), 1)
                self.assertIsInstance(_first_pipe_executor(new_config), _RemoteExecutor)
                # Original config is left untouched.
                self.assertIs(_first_pipe_executor(config), ppe)
            finally:
                _shutdown_pools(pools)
        finally:
            ppe.shutdown()

    def test_hoist_dedupes_shared_pool(self) -> None:
        """The same ProcessPoolExecutor on two pipes maps to one shared pool/executor."""
        ppe = ProcessPoolExecutor(max_workers=2)
        try:
            config = (
                PipelineBuilder()
                .add_source(_PicklableSource(10))
                .pipe(_sync_double, executor=ppe)
                .pipe(_sync_double, executor=ppe)
                .add_sink(10)
                .get_config()
            )
            new_config, pools = _hoist_process_pools(config)
            try:
                self.assertEqual(len(pools), 1)
                ex0 = _pipe_executor(new_config.pipes[0])
                ex1 = _pipe_executor(new_config.pipes[1])
                self.assertIsInstance(ex0, _RemoteExecutor)
                self.assertIs(ex0, ex1)
            finally:
                _shutdown_pools(pools)
        finally:
            ppe.shutdown()

    def test_hoist_leaves_threadpool_untouched(self) -> None:
        """Non-ProcessPoolExecutor executors are not hoisted (no pools spawned)."""
        tpe = ThreadPoolExecutor(max_workers=2)
        try:
            config = _config_with_executor(tpe)
            new_config, pools = _hoist_process_pools(config)
            self.assertEqual(pools, [])
            self.assertIs(new_config, config)
        finally:
            tpe.shutdown()

    def test_remote_executor_is_detected_as_process_pool(self) -> None:
        """_RemoteExecutor must look like a process pool (for sync-generator batching)."""
        ctx = mp.get_context()
        pool = _WorkerPool(ctx, 2, None, ())
        try:
            self.assertTrue(_is_process_pool(pool.make_executor()))
        finally:
            _shutdown_pools([pool])

    def test_remote_executor_exposes_max_workers(self) -> None:
        """The proxy mirrors ProcessPoolExecutor._max_workers (read by stats logging)."""
        ctx = mp.get_context()
        pool = _WorkerPool(ctx, 3, None, ())
        try:
            ex = pool.make_executor()
            self.assertEqual(ex._max_workers, 3)
            # __setstate__ restores it (the queues themselves only pickle via process
            # inheritance, so exercise the state round-trip directly rather than pickling).
            restored = _RemoteExecutor.__new__(_RemoteExecutor)
            restored.__setstate__(ex.__getstate__())
            self.assertEqual(restored._max_workers, 3)
        finally:
            _shutdown_pools([pool])

    def test_hoist_rejects_used_process_pool(self) -> None:
        """A ProcessPoolExecutor that already ran work is rejected (must be freshly built)."""
        ppe = ProcessPoolExecutor(max_workers=2)
        try:
            ppe.submit(_sync_double, 1).result(timeout=30)  # spawns worker processes
            with self.assertRaises(ValueError):
                _hoist_process_pools(_config_with_executor(ppe))
        finally:
            ppe.shutdown()

    def test_worker_pool_roundtrip_and_teardown(self) -> None:
        """Workers run submitted tasks, propagate exceptions, and are reaped on shutdown."""
        ctx = mp.get_context()
        pool = _WorkerPool(ctx, 2, None, ())
        try:
            ex = pool.make_executor()
            self.assertEqual(ex.submit(_sync_double, 21).result(timeout=30), 42)
            with self.assertRaises(ValueError):
                ex.submit(_raise_value_error, 0).result(timeout=30)
        finally:
            _shutdown_pools([pool])
        for p in pool._procs:
            self.assertFalse(p.is_alive())

    def test_worker_pool_initializer_failure_fails_tasks(self) -> None:
        """If the pool initializer raises, every task fails with a picklable error (no hang)."""
        ctx = mp.get_context()
        pool = _WorkerPool(ctx, 2, _failing_initializer, ())
        try:
            ex = pool.make_executor()
            # Submit more tasks than workers: each must get its own failure response, proving
            # the workers keep serving (rather than dying on an unpicklable re-send) and that
            # no future is left hanging.
            futures = [ex.submit(_sync_double, i) for i in range(5)]
            for fut in futures:
                with self.assertRaises(RuntimeError):
                    fut.result(timeout=30)
        finally:
            _shutdown_pools([pool])

    def test_remote_executor_breaks_after_router_exit(self) -> None:
        """When the router exits on a closed out queue, pending and future submits fail fast."""
        ctx = mp.get_context()
        pool = _WorkerPool(ctx, 1, None, ())
        try:
            ex = pool.make_executor()
            # Register a pending future, then simulate the router thread exiting because
            # ``_out_q`` was closed (EOFError/OSError) before the result came back.
            pending: Future[int] = Future()
            ex._futures[next(ex._counter)] = pending
            ex._fail_pending("out queue closed")
            # The previously pending future is failed rather than left hanging forever.
            self.assertIsInstance(pending.exception(timeout=5), BrokenExecutor)
            # A later submit must fail fast (the router will not restart to drain its result)
            # rather than silently registering a future that never resolves.
            with self.assertRaises(BrokenExecutor):
                ex.submit(_sync_double, 1)
        finally:
            _shutdown_pools([pool])

    def test_worker_pool_cleans_up_on_partial_start_failure(self) -> None:
        """If a worker fails to start partway, already-started workers are torn down."""
        real_ctx = mp.get_context()
        started: list[Process] = []

        class _FlakyCtx:
            """Wraps a real context but fails the second ``Process.start`` call."""

            def Queue(self, *args: Any, **kwargs: Any) -> object:
                return real_ctx.Queue(*args, **kwargs)

            def Process(self, *args: Any, **kwargs: Any) -> Process:
                proc = real_ctx.Process(*args, **kwargs)

                # Proxy the process rather than monkeypatching ``proc.start``: under the
                # forkserver start method ``start()`` pickles the ``Process`` object, and a
                # closure stashed on its instance dict would be unpicklable. The real process
                # stays clean (only it is ever started); the proxy enforces the failure.
                class _Proxy:
                    def start(self_inner) -> None:
                        if len(started) >= 1:
                            raise OSError("cannot allocate worker")
                        proc.start()
                        started.append(proc)

                    def __getattr__(self_inner, name: str) -> Any:
                        return getattr(proc, name)

                return cast(Process, _Proxy())

        with self.assertRaises(OSError):
            _WorkerPool(cast(mp.context.BaseContext, _FlakyCtx()), 3, None, ())
        # The worker that did start is not left running with no owner.
        self.assertEqual(len(started), 1)
        for proc in started:
            self.assertFalse(proc.is_alive())

    @_ignore_warnings(_FORK_WARNING, _UNAWAITED_COROUTINE, _PROCESSPOOL_FORK_WARNING)
    def test_run_in_subprocess_with_processpool(self) -> None:
        """run_pipeline_in_subprocess works with a ProcessPoolExecutor on a pipe."""
        config = _config_with_executor(ProcessPoolExecutor(max_workers=2))
        results = list(run_pipeline_in_subprocess(config, num_threads=1))
        self.assertEqual(sorted(results), [2 * i for i in range(10)])

    @_ignore_warnings(_FORK_WARNING, _UNAWAITED_COROUTINE, _PROCESSPOOL_FORK_WARNING)
    def test_run_in_subprocess_with_processpool_generator(self) -> None:
        """A sync generator op + ProcessPoolExecutor runs end-to-end (batch materialized)."""
        config = (
            PipelineBuilder()
            .add_source(_PicklableSource(5))
            .pipe(_sync_double_gen, executor=ProcessPoolExecutor(max_workers=2))
            .add_sink(10)
            .get_config()
        )
        results = list(run_pipeline_in_subprocess(config, num_threads=1))
        expected = []
        for i in range(5):
            expected.extend([2 * i, 2 * i + 1])
        self.assertEqual(sorted(results), sorted(expected))

    @unittest.skipUnless(
        "fork" in mp.get_all_start_methods(),
        "fork start method is not available on this platform",
    )
    def test_hoist_warns_on_fork_from_multithreaded(self) -> None:
        """fork start method + a live extra thread in the parent → deadlock warning."""
        ppe = ProcessPoolExecutor(max_workers=2)
        started, release = threading.Event(), threading.Event()

        def _hold() -> None:
            started.set()
            release.wait()

        t = threading.Thread(target=_hold, daemon=True)
        t.start()
        started.wait()
        try:
            with self.assertWarns(RuntimeWarning):
                _, pools = _hoist_process_pools(_config_with_executor(ppe), "fork")
            _shutdown_pools(pools)
        finally:
            release.set()
            t.join()
            ppe.shutdown()

    def test_hoist_no_warn_with_spawn(self) -> None:
        """The spawn start method never triggers the fork-deadlock warning."""
        ppe = ProcessPoolExecutor(max_workers=2)
        started, release = threading.Event(), threading.Event()

        def _hold() -> None:
            started.set()
            release.wait()

        t = threading.Thread(target=_hold, daemon=True)
        t.start()
        started.wait()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, pools = _hoist_process_pools(_config_with_executor(ppe), "spawn")
            _shutdown_pools(pools)
            self.assertEqual(
                [w for w in caught if "can deadlock" in str(w.message)], []
            )
        finally:
            release.set()
            t.join()
            ppe.shutdown()

    @_ignore_warnings(_UNAWAITED_COROUTINE)
    def test_run_in_subprocess_with_processpool_spawn(self) -> None:
        """Non-fork (spawn) mp_context works end-to-end with a ProcessPoolExecutor."""
        config = _config_with_executor(ProcessPoolExecutor(max_workers=2))
        results = list(
            run_pipeline_in_subprocess(config, num_threads=1, mp_context="spawn")
        )
        self.assertEqual(sorted(results), [2 * i for i in range(10)])

    @_ignore_warnings(_FORK_WARNING, _UNAWAITED_COROUTINE, _PROCESSPOOL_FORK_WARNING)
    def test_run_in_subprocess_processpool_reused_across_epochs(self) -> None:
        """A ProcessPoolExecutor pipe survives re-iteration: pools are reaped at
        teardown, not after each epoch."""
        config = _config_with_executor(ProcessPoolExecutor(max_workers=2))
        src = run_pipeline_in_subprocess(config, num_threads=1)
        expected = [2 * i for i in range(10)]
        for _ in range(3):
            self.assertEqual(sorted(src), expected)


class TestOverrideStage(unittest.TestCase):
    @_ignore_warnings(_UNAWAITED_COROUTINE)
    def test_override_stage_id(self) -> None:
        """Providing `stage_id` overrides the index of stages."""
        ref = 12345

        class CheckNameQueue(AsyncQueue):
            index = ref

            def __init__(self, name, *, buffer_size: int = 1) -> None:
                print(name)
                # pyrefly: ignore [missing-attribute]
                id = re.match(r"\d+:(\d+):.*", str(name)).group(1)
                assert id == str(self.index)
                CheckNameQueue.index += 1
                super().__init__(name, buffer_size=buffer_size)

        (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(lambda x: x)
            .pipe(lambda x: x)
            .pipe(lambda x: x)
            .add_sink()
            .build(num_threads=1, queue_class=CheckNameQueue, stage_id=ref)
        )


class TestPipelineFailureStructure(unittest.TestCase):
    def _build_failing_pipeline(self):
        def failing_range(n):
            yield from range(n)
            raise ValueError("Iterator failed")

        return (
            PipelineBuilder()
            .add_source(failing_range(3))
            .pipe(passthrough)
            .add_sink(1000)
            .build(num_threads=1)
        )

    def test_pipeline_failure_is_exception_group(self) -> None:
        pipeline = self._build_failing_pipeline()

        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        pf = ctx.exception
        if sys.version_info >= (3, 11):
            self.assertIsInstance(pf, ExceptionGroup)
        else:
            self.assertIsInstance(pf, RuntimeError)
        self.assertGreaterEqual(len(pf.exceptions), 1)
        exception_types = {type(e) for e in pf.exceptions}
        self.assertTrue(exception_types & {ValueError})

    def test_pipeline_failure_individual_exceptions(self) -> None:
        def failing_range(n):
            yield from range(n)
            raise TypeError("source failed")

        pipeline = (
            PipelineBuilder()
            .add_source(failing_range(3))
            .pipe(passthrough)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        pf = ctx.exception
        if sys.version_info >= (3, 11):
            self.assertIsInstance(pf, ExceptionGroup)
        else:
            self.assertIsInstance(pf, RuntimeError)
        self.assertGreaterEqual(len(pf.exceptions), 1)
        self.assertTrue(
            any(isinstance(e, TypeError) for e in pf.exceptions),
        )

    @unittest.skipIf(sys.version_info < (3, 11), "ExceptionGroup requires Python 3.11+")
    def test_pipeline_failure_subgroup(self) -> None:
        def failing_range(n):
            yield from range(n)
            raise ValueError("Iterator failed")

        pipeline = (
            PipelineBuilder()
            .add_source(failing_range(3))
            .pipe(passthrough)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        pf = ctx.exception
        sub = pf.subgroup(ValueError)
        self.assertIsNotNone(sub)
        self.assertIsInstance(sub, PipelineFailure)
        self.assertTrue(all(isinstance(e, ValueError) for e in sub.exceptions))

    @unittest.skipIf(sys.version_info < (3, 11), "ExceptionGroup requires Python 3.11+")
    def test_pipeline_failure_notes_contain_stage_name(self) -> None:
        def failing_range(n):
            yield from range(n)
            raise ValueError("Iterator failed")

        pipeline = (
            PipelineBuilder()
            .add_source(failing_range(3))
            .pipe(passthrough)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        pf = ctx.exception
        for exc in pf.exceptions:
            notes = getattr(exc, "__notes__", [])
            self.assertTrue(
                any(note.startswith("Pipeline stage:") for note in notes),
            )
