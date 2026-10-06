# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the fused worker's transfer handling (``_subprocess_pipeline_pool``).

Input expansion and output coalescing are invisible end to end -- they change where a transfer
is unpacked and how many transfers a worker sends, not what comes out of the region -- so their
properties are asserted here against scripted queues and a stand-in for the nested pipeline.
That keeps thread placement, "blocks exactly once", "stops at the bound" and "flushes what it
already has" deterministic instead of dependent on how fast a subprocess happens to produce.
"""

import asyncio
import queue
import threading
import unittest
from collections.abc import Iterator
from typing import Any

from spdl.pipeline import AsyncQueue
from spdl.pipeline._components import (
    _EPOCH,
    _ITEM,
    _POOL_SHUTDOWN,
    _RESULT,
    _SESSION_END,
)
from spdl.pipeline._subprocess_pipeline_pool import (
    _drain_chunk,
    _DrainSource,
    _run_sessions,
    _stream_results,
)
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig

# Script markers for _FakePipeline.
_EOF: object = object()  # raise EOFError -- end of the epoch/session
_EMPTY: object = object()  # raise queue.Empty -- nothing buffered right now
_BOOM: object = object()  # raise RuntimeError -- an unexpected failure mid-chunk


class _CountingQueue(queue.Queue[Any]):
    """Queue that records how many transport messages the worker reads."""

    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        self.get_calls += 1
        return super().get(block=block, timeout=timeout)


class _ThreadRecordingList(list[Any]):
    """Transfer payload that records the thread expanding each of its items."""

    def __init__(self, items: list[Any]) -> None:
        super().__init__(items)
        self.iteration_threads: list[int] = []

    def __iter__(self) -> Iterator[Any]:
        for item in super().__iter__():
            self.iteration_threads.append(threading.get_ident())
            yield item


class InputTransferDrainTest(unittest.TestCase):
    """A worker reads one message in its pool, then expands its items on the event loop."""

    def test_session_payload_is_expanded_on_worker_event_loop(self) -> None:
        """A finite transfer is expanded beside async region ops, not by executor next calls."""
        items = _ThreadRecordingList([1, [2, 3], 4])
        in_q = _CountingQueue()
        in_q.put((_ITEM, items))
        in_q.put((_SESSION_END, None))
        in_q.put((_POOL_SHUTDOWN, None))
        out_q: queue.Queue[Any] = queue.Queue()
        op_threads: list[int] = []

        async def _record_thread(item: Any) -> Any:
            op_threads.append(threading.get_ident())
            return item

        config = PipelineConfig(
            src=SourceConfig([]),
            pipes=[Pipe(_record_thread)],
            sink=SinkConfig(8),
        )
        _run_sessions(
            in_q,
            out_q,
            config,
            {
                "num_threads": 2,
                "queue_class": AsyncQueue,
                "task_hook_factory": lambda _: [],
            },
            output_buffer_size=8,
        )

        messages = [out_q.get_nowait() for _ in range(out_q.qsize())]
        results = [
            item for kind, payload in messages if kind == _RESULT for item in payload
        ]
        self.assertEqual(results, [1, [2, 3], 4])
        self.assertEqual(in_q.get_calls, 3)
        self.assertEqual(len(items.iteration_threads), 3)
        self.assertEqual(len(set(op_threads)), 1)
        self.assertEqual(set(items.iteration_threads), set(op_threads))

    def test_continuous_source_expands_each_epoch_on_event_loop(self) -> None:
        """A continuous source reads per transfer and keeps chunks inside epoch boundaries."""
        first = _ThreadRecordingList([1, 2])
        second = _ThreadRecordingList([[3, 4]])
        in_q = _CountingQueue()
        for message in (
            (_ITEM, first),
            (_EPOCH, None),
            (_ITEM, second),
            (_EPOCH, None),
            (_POOL_SHUTDOWN, None),
        ):
            in_q.put(message)
        source = _DrainSource(in_q)

        async def _read_epochs() -> tuple[int, list[list[Any]]]:
            loop_thread = threading.get_ident()
            epochs = [[item async for item in source] for _ in range(3)]
            return loop_thread, epochs

        loop_thread, epochs = asyncio.run(_read_epochs())
        self.assertEqual(epochs, [[1, 2], [[3, 4]], []])
        self.assertEqual(in_q.get_calls, 5)
        self.assertEqual(first.iteration_threads, [loop_thread, loop_thread])
        self.assertEqual(second.iteration_threads, [loop_thread])
        self.assertTrue(source.exiting)


class _FakePipeline:
    """Nested pipeline stand-in driven by two scripted call sequences.

    ``blocking`` is consumed by ``get_item`` and ``ready`` by ``_get_item_nowait``, so a test
    states exactly which reads block and which find something already buffered.
    """

    def __init__(self, blocking: list[Any], ready: list[Any]) -> None:
        self.blocking: list[Any] = list(blocking)
        self.ready: list[Any] = list(ready)
        self.blocking_calls: int = 0
        self.nowait_calls: int = 0

    @staticmethod
    def _pop(seq: list[Any], name: str) -> Any:
        if not seq:
            raise AssertionError(f"{name} was called more times than the test scripted")
        value = seq.pop(0)
        if value is _EOF:
            raise EOFError
        if value is _EMPTY:
            raise queue.Empty
        if value is _BOOM:
            raise RuntimeError("boom")
        return value

    def get_item(self) -> Any:
        self.blocking_calls += 1
        return self._pop(self.blocking, "get_item")

    def _get_item_nowait(self) -> Any:
        self.nowait_calls += 1
        return self._pop(self.ready, "_get_item_nowait")


class DrainChunkTest(unittest.TestCase):
    """``_drain_chunk`` blocks for the first result and then takes only what is ready."""

    def test_takes_ready_items_after_one_blocking_read(self) -> None:
        """One read blocks; the rest of the chunk comes from what is already buffered."""
        pipeline = _FakePipeline(blocking=[1], ready=[2, 3, _EMPTY])
        out: list[Any] = []
        self.assertFalse(_drain_chunk(pipeline, out, 8))
        self.assertEqual(out, [1, 2, 3])
        self.assertEqual(pipeline.blocking_calls, 1)

    def test_stops_at_output_buffer_size(self) -> None:
        """The chunk is capped even when more results are already available."""
        pipeline = _FakePipeline(blocking=[1], ready=[2, 3, 4, 5])
        out: list[Any] = []
        self.assertFalse(_drain_chunk(pipeline, out, 3))
        self.assertEqual(out, [1, 2, 3])
        self.assertEqual(pipeline.ready, [4, 5])  # not over-drained

    def test_output_buffer_size_one_never_polls(self) -> None:
        """An unbuffered region pays nothing for coalescing: no non-blocking read happens."""
        pipeline = _FakePipeline(blocking=[1], ready=[])
        out: list[Any] = []
        self.assertFalse(_drain_chunk(pipeline, out, 1))
        self.assertEqual(out, [1])
        self.assertEqual(pipeline.nowait_calls, 0)

    def test_end_of_stream_on_first_read(self) -> None:
        """An epoch/session that produced nothing reports the end with an empty chunk."""
        pipeline = _FakePipeline(blocking=[_EOF], ready=[])
        out: list[Any] = []
        self.assertTrue(_drain_chunk(pipeline, out, 8))
        self.assertEqual(out, [])

    def test_end_of_stream_mid_chunk_keeps_items(self) -> None:
        """Hitting the end while filling a chunk keeps what was already collected.

        The caller flushes ``out`` before acting on the end, so these results must survive.
        """
        pipeline = _FakePipeline(blocking=[1], ready=[2, _EOF])
        out: list[Any] = []
        self.assertTrue(_drain_chunk(pipeline, out, 8))
        self.assertEqual(out, [1, 2])


class StreamResultsTest(unittest.TestCase):
    """``_stream_results`` forwards one stream as a sequence of chunked transfers."""

    def test_emits_one_transfer_per_chunk(self) -> None:
        """Each chunk becomes one ``_RESULT`` message; the empty final one is skipped."""
        pipeline = _FakePipeline(blocking=[1, 4, _EOF], ready=[2, 3, _EMPTY, _EMPTY])
        out_q: queue.Queue[Any] = queue.Queue()
        _stream_results(pipeline, out_q, 8)
        self.assertEqual(
            [out_q.get_nowait() for _ in range(out_q.qsize())],
            [(_RESULT, [1, 2, 3]), (_RESULT, [4])],
        )

    def test_flushes_collected_results_before_propagating_a_failure(self) -> None:
        """A failure mid-chunk still relays what was produced before it.

        Without the flush, buffering would silently lose results that an unbuffered region
        would have delivered.
        """
        pipeline = _FakePipeline(blocking=[1], ready=[2, _BOOM])
        out_q: queue.Queue[Any] = queue.Queue()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            _stream_results(pipeline, out_q, 8)
        self.assertEqual(out_q.get_nowait(), (_RESULT, [1, 2]))
