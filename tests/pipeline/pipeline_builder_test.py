# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import os
import platform
import random
import re
import threading
import time
import unittest
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from multiprocessing import Process
from typing import TypeVar

from parameterized import parameterized
from spdl.pipeline import (
    AsyncQueue,
    PipelineBuilder,
    PipelineFailure,
    run_pipeline_in_subprocess,
    TaskHook,
    TaskStatsHook,
)
from spdl.pipeline._components import _get_global_id, _set_global_id
from spdl.pipeline._components._common import _EOF
from spdl.pipeline._components._hook import _periodic_dispatch
from spdl.pipeline._components._pipe import (
    _FailCounter,
    _get_fail_counter,
    _pipe,
    _PipeArgs,
)
from spdl.pipeline._components._sink import _sink
from spdl.pipeline._components._source import _source
from spdl.source.utils import embed_shuffle

T = TypeVar("T")


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
        queue = AsyncQueue(name="foo", buffer_size=0)
        coro = _source([], queue)
        asyncio.run(coro)
        self.assertEqual(_flush_aqueue(queue), [_EOF])

    def test_async_enqueue_simple(self) -> None:
        """_async_enqueue should put the values in the queue."""
        src = list(range(6))
        queue = AsyncQueue(name="foo", buffer_size=0)
        coro = _source(src, queue)
        asyncio.run(coro)
        vals = _flush_aqueue(queue)
        self.assertEqual(vals, [*src, _EOF])

    def test_async_enqueue_iterator_failure(self) -> None:
        """When `iterator` fails, the exception is propagated."""

        def src():
            yield from range(10)
            raise RuntimeError("Failing the iterator.")

        coro = _source(src(), AsyncQueue(name="foo", buffer_size=0))

        with self.assertRaises(RuntimeError):
            asyncio.run(coro)  # Not raising

    def test_async_enqueue_cancel(self) -> None:
        """_async_enqueue is cancellable."""

        async def _test():
            queue = AsyncQueue(name="foo", buffer_size=1)

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
        input_queue: AsyncQueue = AsyncQueue(name="input", buffer_size=0)
        output_queue: AsyncQueue = AsyncQueue(name="output", buffer_size=0)

        data = [] if empty else list(range(3))
        _put_aqueue(input_queue, data, eof=True)

        coro = _sink(input_queue, output_queue)

        asyncio.run(coro)
        results = _flush_aqueue(output_queue)

        self.assertEqual(results, data)

    def test_async_sink_cancel(self) -> None:
        """_async_sink is cancellable."""

        async def _test():
            input_queue = AsyncQueue(name="input")
            output_queue = AsyncQueue(name="output")

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
        input_queue = AsyncQueue(name="input", buffer_size=0)
        output_queue = AsyncQueue(name="output", buffer_size=0)

        async def test():
            ref = list(range(6))
            _put_aqueue(input_queue, ref, eof=True)

            await _pipe(
                "adouble",
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
        input_queue = AsyncQueue(name="input", buffer_size=0)
        output_queue = AsyncQueue(name="output", buffer_size=0)

        async def skip_even(v):
            if v % 2:
                return v

        async def test():
            _put_aqueue(input_queue, range(10), eof=True)

            await _pipe(
                "skip_even",
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
        input_queue = AsyncQueue(name="input", buffer_size=0)
        output_queue = AsyncQueue(name="output", buffer_size=0)

        async def _2args(val: int, _):
            return val

        async def test():
            ref = list(range(6))
            _put_aqueue(input_queue, ref, eof=False)

            with self.assertRaises(TypeError):
                await _pipe(
                    "_2args",
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
        input_queue = AsyncQueue(name="input", buffer_size=0)
        output_queue = AsyncQueue(name="output", buffer_size=1)

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
                "astuck",
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
            input_queue = AsyncQueue(name="input", buffer_size=0)
            output_queue = AsyncQueue(name="output", buffer_size=0)

            ref = [1, 2, 3, 4]
            _put_aqueue(input_queue, ref, eof=False)

            coro = _pipe(
                "delay",
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

    def test_async_pipe_concurrency_throughput(self) -> None:
        """increasing concurrency improves the throughput."""

        async def delay(val):
            await asyncio.sleep(0.5)
            return val

        async def test(concurrency):
            input_queue = AsyncQueue(name="input", buffer_size=0)
            output_queue = AsyncQueue(name="output", buffer_size=0)

            ref = [4, 5, 6, 7, _EOF]
            _put_aqueue(input_queue, ref, eof=False)

            t0 = time.monotonic()
            await _pipe(
                "delay",
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
            elapsed = time.monotonic() - t0

            result = _flush_aqueue(output_queue)

            self.assertEqual(set(result), set(ref))
            self.assertEqual(result[-1], ref[-1])
            self.assertEqual(result[-1], _EOF)

            return elapsed

        elapsed1 = asyncio.run(test(1))
        elapsed4 = asyncio.run(test(4))

        self.assertGreater(elapsed1, 1.8)
        self.assertLess(elapsed4, 1)


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
            async def task_hook(self):
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
            async def task_hook(self):
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
    async def task_hook(self):
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

        def hook_factory(name: str) -> list[TaskHook]:
            if "adouble" in name:
                return [h1]
            if "aggregate" in name:
                return [h2]
            if "_fail" in name:
                return [h3]
            raise RuntimeError("Unexpected")

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
            self.assertEqual([], list(pipeline.get_iterator(timeout=10)))

        self.assertEqual(h1._enter_stage_called, 1)
        self.assertEqual(h1._exit_stage_called, 1)
        self.assertEqual(h1._enter_task_called, 10)
        self.assertEqual(h1._exit_task_called, 10)

        self.assertEqual(h2._enter_stage_called, 1)
        self.assertEqual(h2._exit_stage_called, 1)
        self.assertEqual(h2._enter_task_called, 11)
        self.assertEqual(h2._exit_task_called, 11)

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
            async def task_hook(self):
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
            self.assertEqual(list(range(10)), list(pipeline.get_iterator(timeout=10)))

        for h in hooks:
            self.assertEqual(h._enter_stage_called, 1)
            self.assertEqual(h._exit_stage_called, 1)
            self.assertEqual(h._enter_task_called, 10)
            self.assertEqual(h._exit_task_called, 10)

    def test_pipeline_hook_failure_enter_stage(self) -> None:
        """If enter_stage fails, the pipeline is aborted."""

        class _enter_stage_fail(TaskHook):
            @asynccontextmanager
            async def stage_hook(self):
                raise RuntimeError("failing")

            @asynccontextmanager
            async def task_hook(self):
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
                vals = list(pipeline.get_iterator(timeout=10))

        self.assertEqual(vals, [])

    def test_pipeline_hook_failure_exit_stage(self) -> None:
        """If exit_stage fails, the error is propagated to the front end."""

        class _exit_stage_fail(TaskHook):
            @asynccontextmanager
            async def stage_hook(self):
                yield
                raise RuntimeError("failing")

            @asynccontextmanager
            async def task_hook(self):
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
                vals = list(pipeline.get_iterator(timeout=10))
        self.assertEqual(vals, list(range(10)))

    def test_pipeline_hook_failure_enter_task(self) -> None:
        """If enter_task fails, the pipeline does not fail."""

        class _hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self):
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
            self.assertEqual([], list(pipeline.get_iterator(timeout=10)))

    def test_pipeline_hook_failure_exit_task(self) -> None:
        """If exit_task fails, the pipeline does not fail.

        IMPORTANT: The result is dropped.
        """

        class _exit_stage_fail(TaskHook):
            @asynccontextmanager
            async def task_hook(self):
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
            self.assertEqual(list(pipeline.get_iterator(timeout=10)), [])

    def test_pipeline_hook_exit_task_capture_error(self) -> None:
        """If task fails exit_task captures the error."""

        exc_info = None

        class _capture(TaskHook):
            @asynccontextmanager
            async def task_hook(self):
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
            self.assertEqual(list(pipeline.get_iterator(timeout=10)), [])

        self.assertTrue(exc_info is err)


################################################################################
# TaskStatsHook
################################################################################


class TestTaskStatsHook(unittest.TestCase):
    def test_task_stats(self) -> None:
        """TaskStatsHook logs the interval of each task."""

        hook = TaskStatsHook("foo", 1)

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

        hook = TaskStatsHook("foo", 1)
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
                for j, val in enumerate(pipeline.get_iterator(timeout=10)):
                    self.assertEqual(val, (i * 4) + j)

            # Now it's empty
            with self.assertRaises(StopIteration):
                next(pipeline.get_iterator(timeout=10))

    def test_pipeline_resume(self) -> None:
        """AsyncPipeline can execute the source partially, then resumed"""

        # Note
        # If we pass `range(10)` directly, new iterator is created at every run.
        src = iter(range(10))

        pipeline = PipelineBuilder().add_source(src).add_sink(1000).build(num_threads=1)

        with pipeline.auto_stop():
            iterator = pipeline.get_iterator(timeout=10)
            self.assertEqual([0, 1], [next(iterator) for _ in range(2)])

            iterator = pipeline.get_iterator(timeout=10)
            self.assertEqual([2, 3, 4], [next(iterator) for _ in range(3)])

            iterator = pipeline.get_iterator(timeout=10)
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
                for j, item in enumerate(pipeline.get_iterator(timeout=10)):
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
            self.assertEqual(list(pipeline.get_iterator(timeout=10)), list(range(10)))

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
            self.assertEqual(src, list(pipeline.get_iterator(timeout=10)))

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
            self.assertEqual(list(pipeline.get_iterator(timeout=10)), src)

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
            result = list(pipeline.get_iterator(timeout=10))
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
            result = list(pipeline.get_iterator(timeout=10))
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
            result = list(pipeline.get_iterator(timeout=5))
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
            result = list(pipeline.get_iterator(timeout=10))
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
            result = list(pipeline.get_iterator(timeout=10))
            # Returns 2, 3, 4 (x < 5), and 7, 8, 9 (x >= 7)
            self.assertEqual(result, [2, 3, 4, 7, 8, 9])


################################################################################
# AsyncPipeline2
################################################################################


class TestPipelineNoop(unittest.TestCase):
    def test_pipeline_noop(self) -> None:
        """AsyncPipeline2 functions without pipe."""

        apl = PipelineBuilder().add_source(range(10)).add_sink(1).build(num_threads=1)
        with self.assertRaises(RuntimeError):
            apl.get_item(timeout=1)

        with apl.auto_stop():
            for i in range(10):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=1))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=1)

        with self.assertRaises(EOFError):
            apl.get_item(timeout=1)


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
        with self.assertRaises(RuntimeError):
            apl.get_item(timeout=1)

        with apl.auto_stop():
            for i in range(10):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=1))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=1)

        with self.assertRaises(EOFError):
            apl.get_item(timeout=1)


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
                self.assertEqual(i * 2 + 1, pipeline.get_item(timeout=10))

            with self.assertRaises(EOFError):
                pipeline.get_item(timeout=10)


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
        with self.assertRaises(RuntimeError):
            apl.get_item(timeout=1)

        with apl.auto_stop():
            for i in range(10):
                print("fetching", i)
                self.assertEqual(i, apl.get_item(timeout=1))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=1)

        with self.assertRaises(EOFError):
            apl.get_item(timeout=1)


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
            for i, item in enumerate(pipeline.get_iterator(timeout=10)):
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
            results = list(pipeline.get_iterator(timeout=10))
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
            results = list(pipeline.get_iterator(timeout=10))
            self.assertEqual(results, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])


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
            results = list(pipeline.get_iterator(timeout=10))
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
                results = list(pipeline.get_iterator(timeout=10))

        self.assertEqual(results, [1 + 2 * i for i in range(10)])


class TestPipelineType(unittest.TestCase):
    def test_pipeline_type_error(self) -> None:
        """AsyncPipeline immediately fails if pipe function has wrong signature"""

        async def wrong_sig(i, _):
            return i

        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(wrong_sig)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))

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
            results = list(pipeline.get_iterator(timeout=10))
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
                self.assertEqual(i, apl.get_item(timeout=1))

            # Ensure that buffers are filled and the pipeline is blocked.
            time.sleep(0.1)
            # At this point, the output queue holds 5.

        # Only the "5" is retrievable.
        self.assertEqual(5, apl.get_item(timeout=1))

        # The background thread is stopped, so no more data is coming.
        for _ in range(3):
            with self.assertRaises(EOFError):
                apl.get_item(timeout=1)


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
            .pipe(fail)
            .pipe(pwc)
            .add_sink(1)
            .build(num_threads=1)
        )

        with self.assertRaises(RuntimeError):
            apl.get_item(timeout=1)

        with self.assertRaises(PipelineFailure):
            with apl.auto_stop():
                with self.assertRaises(EOFError):
                    apl.get_item(timeout=1)

        self.assertEqual(pwc.cache, [])

        # The background thread is stopped, and the output queue is empty.
        for _ in range(3):
            with self.assertRaises(EOFError):
                apl.get_item(timeout=1)


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
                self.assertEqual(i, apl.get_item(timeout=1))

            with self.assertRaises(EOFError):
                apl.get_item(timeout=1)


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
            for i, item in enumerate(apl.get_iterator(timeout=1)):
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
            for i, item in enumerate(apl.get_iterator(timeout=10)):
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
            output = list(apl.get_iterator(timeout=10))
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
            output = list(apl.get_iterator(timeout=10))
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
            output = list(apl.get_iterator(timeout=10))
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
            output = list(apl.get_iterator(timeout=10))
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
            output = list(apl.get_iterator(timeout=10))
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
            output = list(apl.get_iterator(timeout=10))
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
            output = list(apl.get_iterator(timeout=0.2))
        self.assertEqual(output, expected)

    def test_pipeline_pipe_agen_wrong_hook(self) -> None:
        """pipe works with async generator function, even when hook abosrb the StopAsyncIteration"""

        class _Hook(TaskHook):
            @asynccontextmanager
            async def task_hook(self):
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
            output = list(apl.get_iterator(timeout=10))
        self.assertEqual(output, expected)

    def test_pipeline_source_agen(self) -> None:
        """source works with async generator function"""

        async def source():
            for i in range(3):
                yield i

        apl = PipelineBuilder().add_source(source()).add_sink(1).build(num_threads=1)

        expected = [0, 1, 2]
        with apl.auto_stop():
            output = list(apl.get_iterator(timeout=10))
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
        sleep = 0.5

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

        def op(i: int) -> int:
            # sleep to block this thread, so that
            # we use all the threads in the pool
            time.sleep(sleep)
            print(i, thread_local_storage.value)
            return thread_local_storage.value

        pipeline = (
            PipelineBuilder()
            .add_source(range(num_threads))
            .pipe(op, executor=executor, concurrency=num_threads)
            .add_sink(1)
            .build(num_threads=1)
        )

        t0 = time.monotonic()
        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))
        elapsed = time.monotonic() - t0

        self.assertEqual(0, len(ref_copy))
        self.assertEqual(ref, set(vals))
        self.assertLess(elapsed, sleep * 2)

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
            vals = list(pipeline.get_iterator(timeout=10))

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
            vals = list(pipeline.get_iterator(timeout=10))
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
            vals = list(pipeline.get_iterator(timeout=10))

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
            vals = list(pipeline.get_iterator(timeout=10))

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
            vals = list(pipeline.get_iterator(timeout=10))

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


def hook_factory(_: str) -> list[TaskHook]:
    return [CountHook()]


class TestPipelinebuilderPicklable(unittest.TestCase):
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

        fc1_1._increment()

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

        fc1_1._increment()

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

        fc1_2._increment()

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

        fc2_1._increment()

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
            vals = list(pipeline.get_iterator(timeout=10))

        self.assertEqual([0, 2, 4, 6, 8], vals)

        pipeline = builder.build(num_threads=1, max_failures=3)
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))

        self.assertEqual([0, 2, 4], vals)

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
                vals = list(pipeline2.get_iterator(timeout=10))

        self.assertEqual([0, 2, 4], vals)

        with self.assertRaises(PipelineFailure):
            with pipeline1.auto_stop():
                vals = list(pipeline1.get_iterator(timeout=10))

        self.assertEqual([0, 2], vals)

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
                vals = list(pipeline.get_iterator(timeout=10))
        self.assertEqual([0, 2], vals)

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
            vals = list(pipeline.get_iterator(timeout=10))
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
        # fail_six fails less often, but at the third failure (12),
        # it should shutdown the pipeline.

        pipeline = builder.build(num_threads=1, max_failures=2)
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))
        self.assertEqual([2, 4, 8, 10], vals)


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
        with self.assertRaises(RuntimeError):
            with pipeline.auto_stop():
                pass


class TestIterableWithShuffle:
    def __init__(self, n: int) -> None:
        self.vals = list(range(n))
        self.seed = 0

    def shuffle(self, *, seed: int) -> None:
        self.vals = self.vals[1:] + self.vals[:1]
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        yield from self.vals


class TestRunPipeline(unittest.TestCase):
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


class TestOverrideStage(unittest.TestCase):
    def test_override_stage_id(self) -> None:
        """Providing `stage_id` overrides the index of stages."""
        ref = 12345

        class CheckNameQueue(AsyncQueue):
            index = ref

            def __init__(self, name: str, *, buffer_size: int = 1) -> None:
                print(name)
                id = re.match(r"\d+:(\d+):.*", name).group(1)
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
