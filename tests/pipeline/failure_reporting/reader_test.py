# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import queue
import threading
import unittest
import warnings
from collections.abc import Iterator

from spdl.pipeline import PipelineBuilder, PipelineFailure
from spdl.pipeline._components import (
    _EPOCH,
    _EPOCH_DONE,
    _ERROR,
    _ITEM,
    _POOL_SHUTDOWN,
    _RESULT,
)
from spdl.pipeline._subprocess_pipeline_pool import _run_continuous
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig
from spdl.pipeline.defs._defs import Pipe, PipelineConfig, SinkConfig, SourceConfig


def _fail_on_item(x: int) -> int:
    raise ValueError("fatal item failure")


def _fail_second_epoch(x: int) -> int:
    if x >= 100:
        raise ValueError("second epoch failed")
    return x


class _TaggedSource:
    def __init__(self) -> None:
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        value = self.epoch * 100
        self.epoch += 1
        yield value


class _ObservedQueue:
    def __init__(self) -> None:
        self.queue: queue.Queue[tuple[int, object]] = queue.Queue()
        self.parked = threading.Event()

    def get(self, *, timeout: float) -> tuple[int, object]:
        if self.queue.empty():
            self.parked.set()
        return self.queue.get(timeout=timeout)


class ReaderFailureTest(unittest.TestCase):
    def test_warm_failure_with_source_waiting_for_next_epoch(self) -> None:
        """A failed worker stops even when its source is already reading a future epoch."""
        source = _ObservedQueue()
        output: queue.Queue[tuple[int, object]] = queue.Queue()
        for message in ((_ITEM, [0]), (_EPOCH, 0)):
            source.queue.put(message)
        worker_id = 0

        def fail_after_source_parks(value: int) -> int:
            if value == 100:
                if not source.parked.wait(10):
                    raise TimeoutError("Source did not start its next epoch read.")
                raise ValueError("second epoch failed with source waiting")
            return value

        config = PipelineConfig(
            src=SourceConfig([]),
            pipes=[Pipe(fail_after_source_parks, concurrency=1, max_failures=0)],
            sink=SinkConfig(3),
        )
        worker = threading.Thread(
            target=_run_continuous,
            args=(source, output, worker_id, config, {"num_threads": 2}, 3),
        )
        worker.start()
        try:
            self.assertEqual(output.get(timeout=10), (_RESULT, [0]))
            self.assertEqual(output.get(timeout=10), (_EPOCH_DONE, (0, worker_id)))
            for message in ((_ITEM, [100]), (_EPOCH, 1)):
                source.queue.put(message)
            kind, error = output.get(timeout=10)
            self.assertEqual(kind, _ERROR)
            self.assertTrue(source.parked.is_set())
            self.assertIsInstance(error, RuntimeError)
            self.assertIn("fail_after_source_parks", str(error))
            self.assertIn("exceeds the threshold (0)", str(error))
            worker.join(timeout=10)
            self.assertFalse(
                worker.is_alive(), "Worker did not finish failure cleanup."
            )
        finally:
            source.queue.put((_POOL_SHUTDOWN, None))
            worker.join(timeout=10)

    def _check_failure(
        self, thread_queue: bool, continuous: bool, nowait: bool
    ) -> None:
        pipeline = (
            PipelineBuilder()
            .add_source(range(1), continuous=continuous)
            .pipe(_fail_on_item)
            .add_sink(1)
            .build(num_threads=1, max_failures=0, use_thread_output_queue=thread_queue)
        )
        pipeline.start()
        try:
            if nowait:
                self.assertTrue(pipeline._impl._event_loop._task_completed.wait(10))
            with self.assertRaises(PipelineFailure):
                if nowait:
                    pipeline._get_item_nowait()
                else:
                    pipeline.get_item(timeout=10)
        finally:
            try:
                pipeline.stop(timeout=10)
            except PipelineFailure:
                pass  # Shutdown must not mask a missing reader error.

    def test_blocking_read_propagates_failure_before_stop(self) -> None:
        """Both sink backends report finite and continuous failures before shutdown."""
        for thread in (False, True):
            for continuous in (False, True):
                with self.subTest(thread_queue=thread, continuous=continuous):
                    self._check_failure(thread, continuous, nowait=False)

    def test_nowait_read_propagates_failure_before_stop(self) -> None:
        """Polling distinguishes failed pipelines from successful exhaustion on both sinks."""
        for thread in (False, True):
            for continuous in (False, True):
                with self.subTest(thread_queue=thread, continuous=continuous):
                    self._check_failure(thread, continuous, nowait=True)

    def test_warm_region_failure_is_reported_to_reader(self) -> None:
        """A warm continuous region relays a fatal worker error before explicit shutdown."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pipeline = (
                PipelineBuilder()
                .add_source(_TaggedSource(), continuous=True)
                .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="fork"))
                .pipe(_fail_second_epoch, concurrency=1, max_failures=0)
                .to(MAIN_PROCESS)
                .add_sink(1)
                .build(num_threads=1, max_failures=0)
            )
            pipeline.start()
        try:
            self.assertEqual(list(pipeline.get_iterator(timeout=10)), [0])
            with self.assertRaises(PipelineFailure):
                list(pipeline.get_iterator(timeout=10))
        finally:
            try:
                pipeline.stop(timeout=10)
            except PipelineFailure:
                pass  # Shutdown must not mask a missing reader error.
