# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Iterable
from typing import Any

from spdl.pipeline import AsyncQueue, PipelineBuilder
from spdl.pipeline._components._aggregate import _aggregate
from spdl.pipeline._components._common import _EOF, StageInfo
from spdl.pipeline._components._pipe import _get_fail_counter
from spdl.pipeline.defs import Aggregator

_TEST_INFO = StageInfo(pipeline_id=0, stage_id="0", stage_name="test")


def _put_aqueue(queue: AsyncQueue, vals: Iterable[object], *, eof: bool) -> None:
    for val in vals:
        queue.put_nowait(val)
    if eof:
        queue.put_nowait(_EOF)


def _flush_aqueue(queue: AsyncQueue) -> list[object]:
    ret = []
    while not queue.empty():
        ret.append(queue.get_nowait())
    return ret


class _TrackingAggregator(Aggregator):
    """Aggregator that collects items into batches of size N and records
    accumulate call order for verifying drain behavior."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.buffer: list[Any] = []
        self.accumulate_log: list[Any] = []

    def accumulate(self, item: Any) -> list[Any] | None:
        self.accumulate_log.append(item)
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            result = self.buffer
            self.buffer = []
            return result
        return None

    def flush(self) -> list[Any] | None:
        if self.buffer:
            result = self.buffer
            self.buffer = []
            return result
        return None


class AggregatePipeBulkDrainTest(unittest.IsolatedAsyncioTestCase):
    async def test_stops_draining_on_emit(self) -> None:
        """After the aggregator emits, bulk draining stops and remaining
        items stay in the input queue for the next drain cycle."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        agg = _TrackingAggregator(batch_size=3)
        # Pre-fill 7 items + EOF. The aggregator emits after every 3 items.
        # With stop-on-emit, each drain cycle processes exactly 3 items
        # (emit batch), then stops. The remaining items stay in the input
        # queue for the next cycle. The last item is flushed at EOF.
        _put_aqueue(input_queue, list(range(7)), eof=True)

        await _aggregate(
            _TEST_INFO,
            input_queue,
            output_queue,
            agg,
            _get_fail_counter()(),
            [],
            op_requires_eof=True,
        )

        results = _flush_aqueue(output_queue)
        # Two full batches of 3 + flush of remainder [6] + EOF
        self.assertEqual(results, [[0, 1, 2], [3, 4, 5], [6], _EOF])
        # All 7 items were accumulated
        self.assertEqual(agg.accumulate_log, [0, 1, 2, 3, 4, 5, 6])

    async def test_drains_without_blocking_when_no_emit(self) -> None:
        """When the aggregator doesn't emit, items are drained from the
        queue without blocking (via get_nowait)."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        # batch_size=10 means 5 items won't trigger an emit
        agg = _TrackingAggregator(batch_size=10)

        # Pre-fill 5 items + EOF. None will trigger an emit during
        # accumulate, but flush will emit the remaining buffer.
        _put_aqueue(input_queue, list(range(5)), eof=True)

        await _aggregate(
            _TEST_INFO,
            input_queue,
            output_queue,
            agg,
            _get_fail_counter()(),
            [],
            op_requires_eof=True,
        )

        results = _flush_aqueue(output_queue)
        # flush emits the remaining 5 items
        self.assertEqual(results, [[0, 1, 2, 3, 4], _EOF])
        self.assertEqual(agg.accumulate_log, [0, 1, 2, 3, 4])

    async def test_drop_last(self) -> None:
        """When op_requires_eof=False, EOF stops processing and flush
        is not called, dropping incomplete batches."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        agg = _TrackingAggregator(batch_size=3)

        # 5 items: batch at 3, then 2 remaining are dropped
        _put_aqueue(input_queue, list(range(5)), eof=True)

        await _aggregate(
            _TEST_INFO,
            input_queue,
            output_queue,
            agg,
            _get_fail_counter()(),
            [],
            op_requires_eof=False,
        )

        results = _flush_aqueue(output_queue)
        # Only the complete batch + EOF from queue_stage_hook
        self.assertEqual(results, [[0, 1, 2], _EOF])

    async def test_exception_propagates(self) -> None:
        """Exceptions from the aggregator propagate instead of being
        silently swallowed."""

        class FailingAggregator(Aggregator):
            def accumulate(self, item: Any) -> None:
                raise ValueError("aggregation failed")

            def flush(self) -> None:
                return None

        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        _put_aqueue(input_queue, [1], eof=True)

        with self.assertRaises(ValueError, msg="aggregation failed"):
            await _aggregate(
                _TEST_INFO,
                input_queue,
                output_queue,
                FailingAggregator(),
                _get_fail_counter()(),
                [],
                op_requires_eof=False,
            )

    async def test_single_item_no_emit(self) -> None:
        """A single item that doesn't trigger emit is flushed at EOF."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        agg = _TrackingAggregator(batch_size=5)

        _put_aqueue(input_queue, [42], eof=True)

        await _aggregate(
            _TEST_INFO,
            input_queue,
            output_queue,
            agg,
            _get_fail_counter()(),
            [],
            op_requires_eof=True,
        )

        results = _flush_aqueue(output_queue)
        self.assertEqual(results, [[42], _EOF])

    async def test_eof_with_empty_flush(self) -> None:
        """When items evenly divide batch_size, flush() returns None at EOF.
        The function must still return instead of blocking."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        agg = _TrackingAggregator(batch_size=3)

        # 6 items evenly divides batch_size=3, so flush() returns None
        _put_aqueue(input_queue, list(range(6)), eof=True)

        await _aggregate(
            _TEST_INFO,
            input_queue,
            output_queue,
            agg,
            _get_fail_counter()(),
            [],
            op_requires_eof=True,
        )

        results = _flush_aqueue(output_queue)
        self.assertEqual(results, [[0, 1, 2], [3, 4, 5], _EOF])

    async def test_emit_on_every_item(self) -> None:
        """When batch_size=1, every item triggers an emit and each
        drain cycle processes exactly one item."""
        input_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="input"), buffer_size=0
        )
        output_queue = AsyncQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="output"), buffer_size=0
        )

        agg = _TrackingAggregator(batch_size=1)

        _put_aqueue(input_queue, [10, 20, 30], eof=True)

        await _aggregate(
            _TEST_INFO,
            input_queue,
            output_queue,
            agg,
            _get_fail_counter()(),
            [],
            op_requires_eof=False,
        )

        results = _flush_aqueue(output_queue)
        self.assertEqual(results, [[10], [20], [30], _EOF])


class AggregatePipeEndToEndTest(unittest.TestCase):
    def test_aggregate_bulk_drain_correctness(self) -> None:
        """End-to-end pipeline test: aggregate produces correct batches."""
        src = list(range(10))

        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .aggregate(3)
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(results, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])

    def test_aggregate_custom_op_bulk_drain(self) -> None:
        """End-to-end: custom aggregator with bulk drain produces correct output."""

        class SizeAggregator(Aggregator):
            def __init__(self, threshold: int) -> None:
                self.threshold = threshold
                self.buffer: list[str] = []
                self.total: int = 0

            def accumulate(self, item: str) -> str | None:
                self.buffer.append(item)
                self.total += len(item)
                if self.total >= self.threshold:
                    result = "".join(self.buffer)
                    self.buffer = []
                    self.total = 0
                    return result
                return None

            def flush(self) -> str | None:
                if self.buffer:
                    result = "".join(self.buffer)
                    self.buffer = []
                    self.total = 0
                    return result
                return None

        src = ["a", "bb", "ccc", "dddd", "e", "ff", "ggg", "h"]

        pipeline = (
            PipelineBuilder()
            .add_source(src)
            .aggregate(SizeAggregator(threshold=10))
            .add_sink(1000)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(results, ["abbcccdddd", "effgggh"])
