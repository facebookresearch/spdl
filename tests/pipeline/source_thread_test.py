# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import unittest
from collections.abc import Iterator

from spdl.pipeline import PipelineBuilder, PipelineFailure


class _ThreadRecordingSource:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def __iter__(self) -> Iterator[int]:
        for item in range(3):
            self.thread_ids.append(threading.get_ident())
            yield item


class _CloseFailingIterator:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self._yielded = False
        self.close_calls = 0

    def __iter__(self) -> "_CloseFailingIterator":
        return self

    def __next__(self) -> int:
        if not self._yielded:
            self._yielded = True
            return 1
        if self._fail:
            raise RuntimeError("source failure")
        raise StopIteration

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("close failure")


class SyncSourceThreadTest(unittest.TestCase):
    def test_continuous_source_keeps_thread_affinity(self) -> None:
        """A reusable synchronous source stays on one thread across epochs."""
        source = _ThreadRecordingSource()
        pipeline = (
            PipelineBuilder()
            .add_source(source, continuous=True)
            .add_sink(buffer_size=2)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for _ in range(3):
                self.assertEqual(
                    list(pipeline.get_iterator(timeout=30)),
                    [0, 1, 2],
                )

        self.assertGreaterEqual(len(source.thread_ids), 9)
        self.assertEqual(len(set(source.thread_ids)), 1)
        self.assertNotEqual(source.thread_ids[0], threading.get_ident())

    def test_cancelled_source_closes_iterator_and_stops_thread(self) -> None:
        """Pipeline cancellation closes the iterator and stops its source thread."""
        finished = threading.Event()
        source_threads: list[threading.Thread] = []

        def source() -> Iterator[int]:
            source_threads.append(threading.current_thread())
            try:
                while True:
                    yield 1
            finally:
                finished.set()

        pipeline = (
            PipelineBuilder()
            .add_source(source())
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        pipeline.start()
        try:
            self.assertEqual(next(pipeline.get_iterator(timeout=30)), 1)
        finally:
            pipeline.stop(timeout=5)

        self.assertEqual(len(source_threads), 1)
        source_threads[0].join(timeout=5)
        self.assertFalse(source_threads[0].is_alive())
        self.assertTrue(finished.is_set())

    def test_failed_source_closes_iterator_and_stops_thread(self) -> None:
        """A source failure closes the iterator and stops its source thread."""
        finished = threading.Event()
        source_threads: list[threading.Thread] = []

        def source() -> Iterator[int]:
            source_threads.append(threading.current_thread())
            try:
                yield 1
                raise RuntimeError("source failure")
            finally:
                finished.set()

        pipeline = (
            PipelineBuilder()
            .add_source(source())
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        self.assertEqual(len(source_threads), 1)
        source_threads[0].join(timeout=5)
        self.assertFalse(source_threads[0].is_alive())
        self.assertTrue(finished.is_set())

    def test_exhausted_source_is_not_closed_again(self) -> None:
        """Natural exhaustion does not invoke an iterator's optional close method."""
        source = _CloseFailingIterator()
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            self.assertEqual(list(pipeline.get_iterator(timeout=30)), [1])

        self.assertEqual(source.close_calls, 0)

    def test_source_failure_is_not_masked_by_close_failure(self) -> None:
        """Iterator cleanup cannot replace the source's original exception."""
        source = _CloseFailingIterator(fail=True)
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

        self.assertEqual(source.close_calls, 1)
        self.assertEqual(len(ctx.exception.exceptions), 1)
        self.assertEqual(str(ctx.exception.exceptions[0]), "source failure")

    def test_blocked_source_does_not_delay_pipeline_stop(self) -> None:
        """Pipeline shutdown remains responsive while a source is blocked."""
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        source_threads: list[threading.Thread] = []

        def source() -> Iterator[int]:
            try:
                source_threads.append(threading.current_thread())
                started.set()
                release.wait(timeout=30)
                yield 1
            finally:
                finished.set()

        pipeline = (
            PipelineBuilder()
            .add_source(source())
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        pipeline.start()
        try:
            self.assertTrue(started.wait(timeout=5))
            pipeline.stop(timeout=1)
        finally:
            release.set()
            pipeline.stop(timeout=5)

        self.assertEqual(len(source_threads), 1)
        source_threads[0].join(timeout=5)
        self.assertFalse(source_threads[0].is_alive())
        self.assertTrue(finished.is_set())
