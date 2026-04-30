# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Diff 3b tests: ``Pipeline._resize_concurrency_async`` (INTERNAL).

Per DESIGN v6 / Decision 13, the public foreground-thread
``Pipeline.resize_concurrency`` sync wrapper, ``Pipeline.list_stages``
debug helper, and the lifecycle gating that supported them are dropped.
Only the in-loop async source-of-truth remains. The intended caller is
an in-loop adaptive-concurrency controller running as a
:py:class:`BackgroundTask`. Coverage:

- Async resize from a :py:class:`BackgroundTask` mutates the registered
  semaphore and the sibling ``_dynamic_concurrency`` dict atomically
  (single asyncio turn — no awaits between the two writes).
- ``KeyError`` on unknown stage; valid names are listed in the error.
- ``ValueError`` on ``new_value < 1``.
- Continuous-mode regression: the in-loop async resize takes effect
  across an epoch sentinel boundary.
"""

import asyncio
import inspect
import threading
import time
import unittest
from collections.abc import Iterator
from typing import Any

import later.unittest
from spdl.pipeline import build_pipeline, Pipeline
from spdl.pipeline._bg_task import BackgroundTask
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig


def _identity(x: int) -> int:
    return x


class AsyncResizeFromBackgroundTaskTest(later.unittest.TestCase):
    """Async resize via a BackgroundTask updates semaphore + dict."""

    async def test_async_resize_from_bg_task_succeeds(self) -> None:
        # Capture both the semaphore.max_value AND the sibling
        # _dynamic_concurrency entry to confirm both writes landed in
        # the same event-loop turn.
        applied: dict[str, int] = {"sem": -1, "dict": -1}
        done: threading.Event = threading.Event()

        class ResizerTask(BackgroundTask):
            def __init__(self, pipeline_ref: list[Any]) -> None:
                self._pipeline_ref = pipeline_ref

            async def run(self) -> None:
                await asyncio.sleep(0.05)
                pipeline = self._pipeline_ref[0]
                await pipeline._resize_concurrency_async("s1", 7)
                applied["sem"] = pipeline._impl._semaphore_registry["s1"].max_value
                applied["dict"] = pipeline._impl._dynamic_concurrency["s1"]
                done.set()

        pipeline_ref: list[Any] = [None]

        config = PipelineConfig(
            src=SourceConfig(iter(range(1000))),
            pipes=[Pipe(_identity, concurrency=3, name="s1")],
            sink=SinkConfig(buffer_size=8),
        )
        pipeline = build_pipeline(
            config,
            num_threads=4,
            background_tasks=[lambda: ResizerTask(pipeline_ref)],
            _install_semaphores_for_test=True,
        )
        pipeline_ref[0] = pipeline

        with pipeline.auto_stop():
            it = pipeline.get_iterator()
            for _ in range(10):
                next(it)
            await asyncio.get_running_loop().run_in_executor(None, done.wait, 5.0)

        self.assertTrue(done.is_set(), "Resizer task did not run")
        self.assertEqual(applied["sem"], 7)
        self.assertEqual(applied["dict"], 7)


class AsyncResizeValidationTest(later.unittest.TestCase):
    """``_resize_concurrency_async`` raises on invalid arguments."""

    async def _drive_with_bg_task(
        self,
        config: PipelineConfig[int],
        bg_body: Any,  # async callable taking the pipeline
        result_holder: dict[str, BaseException | None],
    ) -> None:
        invoked: threading.Event = threading.Event()
        pipeline_ref: list[Any] = [None]

        class CheckerTask(BackgroundTask):
            def __init__(self, ref: list[Any]) -> None:
                self._ref = ref

            async def run(self) -> None:
                await asyncio.sleep(0.05)
                try:
                    await bg_body(self._ref[0])
                except Exception as e:
                    result_holder["err"] = e
                finally:
                    invoked.set()

        pipeline = build_pipeline(
            config,
            num_threads=4,
            background_tasks=[lambda: CheckerTask(pipeline_ref)],
            _install_semaphores_for_test=True,
        )
        pipeline_ref[0] = pipeline

        with pipeline.auto_stop():
            it = pipeline.get_iterator()
            for _ in range(10):
                next(it)
            await asyncio.get_running_loop().run_in_executor(None, invoked.wait, 5.0)

        self.assertTrue(invoked.is_set(), "Checker task did not run")

    async def test_unknown_qualified_name_raises_keyerror(self) -> None:
        result_holder: dict[str, BaseException | None] = {"err": None}

        async def body(pipeline: Pipeline[int]) -> None:
            await pipeline._resize_concurrency_async("nope", 4)

        config = PipelineConfig(
            src=SourceConfig(iter(range(1000))),
            pipes=[Pipe(_identity, concurrency=2, name="s1")],
            sink=SinkConfig(buffer_size=4),
        )
        await self._drive_with_bg_task(config, body, result_holder)
        err = result_holder["err"]
        self.assertIsInstance(err, KeyError)
        # KeyError repr wraps the message in quotes, so str(ex) shows it.
        self.assertIn("'s1'", str(err))

    async def test_new_value_zero_raises_valueerror(self) -> None:
        result_holder: dict[str, BaseException | None] = {"err": None}

        async def body(pipeline: Pipeline[int]) -> None:
            await pipeline._resize_concurrency_async("s1", 0)

        config = PipelineConfig(
            src=SourceConfig(iter(range(1000))),
            pipes=[Pipe(_identity, concurrency=2, name="s1")],
            sink=SinkConfig(buffer_size=4),
        )
        await self._drive_with_bg_task(config, body, result_holder)
        self.assertIsInstance(result_holder["err"], ValueError)

    async def test_new_value_negative_raises_valueerror(self) -> None:
        result_holder: dict[str, BaseException | None] = {"err": None}

        async def body(pipeline: Pipeline[int]) -> None:
            await pipeline._resize_concurrency_async("s1", -1)

        config = PipelineConfig(
            src=SourceConfig(iter(range(1000))),
            pipes=[Pipe(_identity, concurrency=2, name="s1")],
            sink=SinkConfig(buffer_size=4),
        )
        await self._drive_with_bg_task(config, body, result_holder)
        self.assertIsInstance(result_holder["err"], ValueError)


class AsyncResizeAtomicityTest(unittest.TestCase):
    """``_resize_concurrency_async`` is one event-loop turn (no awaits).

    The body's atomicity claim is a STRUCTURAL invariant: there is no
    ``await`` between :py:meth:`ResizableSemaphore.resize` and the
    ``_dynamic_concurrency`` dict assignment, so asyncio cannot schedule
    any other coroutine between the two writes. We assert that by
    inspecting the source for a single ``await`` token (the dispatch from
    the caller is the only entry).
    """

    def test_no_await_between_semaphore_resize_and_dict_assignment(
        self,
    ) -> None:
        source = inspect.getsource(Pipeline._resize_concurrency_async)
        # Locate the two writes; assert no `await` keyword sits between
        # them. ``sem.resize(`` and ``self._impl._dynamic_concurrency[``
        # are unique anchors in the body.
        resize_index = source.index("sem.resize(")
        dict_index = source.index("self._impl._dynamic_concurrency[")
        self.assertLess(resize_index, dict_index)
        between = source[resize_index:dict_index]
        # Tokenise on whitespace boundaries to avoid matching ``await`` as
        # a substring of a longer identifier (defensive — there are none
        # today).
        for token in between.split():
            self.assertNotEqual(
                token,
                "await",
                "Found `await` between sem.resize() and dict assignment "
                "— atomicity invariant of _resize_concurrency_async is "
                "broken. The controller's _apply_decision contract "
                "depends on this method completing in one asyncio turn.",
            )


class AsyncResizeContinuousModeTest(unittest.TestCase):
    """In-loop async resize works across continuous-mode epoch boundaries.

    Continuous mode emits epoch-end sentinels but never EOF; the pipeline
    runs until ``pipeline.stop()``. Verify:
      (a) No crash on epoch sentinel propagation when the registry is non-empty.
      (b) Resize at iteration N takes effect for iterations > N.
      (c) In-flight items at the time of resize complete normally.
    """

    EPOCH_SIZE: int = 50
    NUM_EPOCHS: int = 3
    INITIAL_CONCURRENCY: int = 2
    RESIZE_TO: int = 5

    def test_async_resize_works_across_epoch_boundary(self) -> None:
        # Track in-flight count; reset between observation windows.
        in_flight: list[int] = [0]
        peak: list[int] = [0]
        observe_lock: threading.Lock = threading.Lock()
        resize_event: threading.Event = threading.Event()
        resize_done: threading.Event = threading.Event()

        def slow_op(x: int) -> int:
            with observe_lock:
                in_flight[0] += 1
                if in_flight[0] > peak[0]:
                    peak[0] = in_flight[0]
            time.sleep(0.005)
            with observe_lock:
                in_flight[0] -= 1
            return x

        def epoch_source() -> Iterator[int]:
            for _ in range(self.NUM_EPOCHS):
                yield from range(self.EPOCH_SIZE)
                # Continuous-mode pipeline auto-emits an epoch sentinel
                # at the end of each pass over the source iterator.

        # The resize is driven from inside the pipeline event loop via
        # a BackgroundTask — that's the only supported caller for the
        # async resize API.
        class ResizerTask(BackgroundTask):
            def __init__(self, pipeline_ref: list[Any]) -> None:
                self._pipeline_ref = pipeline_ref

            async def run(self) -> None:
                # Wait until the foreground signals it is ready for the
                # resize (after the first epoch has been drained).
                while not resize_event.is_set():
                    await asyncio.sleep(0.01)
                pipeline = self._pipeline_ref[0]
                await pipeline._resize_concurrency_async(
                    "slow", AsyncResizeContinuousModeTest.RESIZE_TO
                )
                resize_done.set()

        pipeline_ref: list[Any] = [None]

        config = PipelineConfig(
            src=SourceConfig(epoch_source(), continuous=True),
            pipes=[
                Pipe(
                    slow_op,
                    concurrency=self.INITIAL_CONCURRENCY,
                    name="slow",
                )
            ],
            sink=SinkConfig(buffer_size=4),
        )
        pipeline = build_pipeline(
            config,
            num_threads=8,
            background_tasks=[lambda: ResizerTask(pipeline_ref)],
            _install_semaphores_for_test=True,
        )
        pipeline_ref[0] = pipeline

        with pipeline.auto_stop():
            it = pipeline.get_iterator()

            # (a) First epoch: confirm initial cap is honoured.
            for _ in range(self.EPOCH_SIZE):
                next(it)
            self.assertLessEqual(peak[0], self.INITIAL_CONCURRENCY)

            # (b)+(c) Trigger resize from the BG task and wait for it.
            with observe_lock:
                peak[0] = 0
            resize_event.set()
            self.assertTrue(
                resize_done.wait(timeout=5.0),
                "Resizer task did not complete the in-loop resize",
            )
            for _ in range(self.EPOCH_SIZE):
                next(it)
            self.assertLessEqual(peak[0], self.RESIZE_TO)
            self.assertGreater(
                peak[0],
                self.INITIAL_CONCURRENCY,
                "Expected resize to allow more concurrency.",
            )

            # (a) Run a third epoch to confirm no crash on sentinel propagation.
            with observe_lock:
                peak[0] = 0
            for _ in range(self.EPOCH_SIZE):
                next(it)
            self.assertLessEqual(peak[0], self.RESIZE_TO)

        # The registry survives epoch sentinel propagation.
        self.assertIn("slow", pipeline._impl._semaphore_registry)
