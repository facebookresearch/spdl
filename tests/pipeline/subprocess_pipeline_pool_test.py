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
import contextlib
import gc
import queue
import threading
import unittest
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.reduction import ForkingPickler
from typing import Any
from unittest import mock

from spdl.pipeline import AsyncQueue
from spdl.pipeline._components import (
    _DONE,
    _EPOCH,
    _EPOCH_DONE,
    _ERROR,
    _ITEM,
    _POOL_SHUTDOWN,
    _RESULT,
    _SESSION_END,
    _subprocess_pipe,
)
from spdl.pipeline._components._common import StageInfo
from spdl.pipeline._components._subprocess_pipe import (
    _check_serialization_error_relay,
    _collect,
    _serialization_failure_boundaries,
)
from spdl.pipeline._subprocess_pipeline_pool import (
    _drain_chunk,
    _DrainSource,
    _handle_queue_feeder_error,
    _install_queue_feeder_error_handler,
    _run_continuous,
    _run_sessions,
    _serialization_error,
    _SERIALIZATION_ERROR_DETAIL_LIMIT,
    _SerializationFailureState,
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
            (_EPOCH, 0),
            (_ITEM, second),
            (_EPOCH, 1),
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


def _serialization_failure_state(
    *,
    failed: bool = False,
    relay_failed: bool = False,
    non_payload_failed: bool = False,
) -> _SerializationFailureState:
    failed_event = threading.Event()
    relay_failed_event = threading.Event()
    non_payload_failed_event = threading.Event()
    if failed:
        failed_event.set()
    if relay_failed:
        relay_failed_event.set()
    if non_payload_failed:
        non_payload_failed_event.set()
    return _SerializationFailureState(
        failed_event, relay_failed_event, non_payload_failed_event
    )


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


class RunContinuousTest(unittest.TestCase):
    """Continuous workers publish each completed generation without losing the next."""

    def test_clears_completed_epoch_before_publishing_boundary(self) -> None:
        """Publishing one boundary cannot clear a concurrently completed next epoch."""
        source_ref: list[Any] = []
        completed_when_published: list[int | None] = []
        next_epoch_seen: list[int | None] = []
        messages: list[tuple[int, Any]] = []
        stream_calls = 0

        class _OutputQueue:
            def put(self, message: tuple[int, Any]) -> None:
                messages.append(message)
                if message[0] == _EPOCH_DONE:
                    source = source_ref[0]
                    completed_when_published.append(source.completed_epoch)
                    source.completed_epoch = 1

        pipeline = mock.Mock()
        pipeline.auto_stop.return_value = contextlib.nullcontext()

        def _build_pipeline(config: PipelineConfig[Any], **kwargs: Any) -> Any:
            del kwargs
            if not isinstance(config.src, SourceConfig):
                raise AssertionError("continuous worker must install one source")
            source_ref.append(config.src.source)
            return pipeline

        def _stream_one_epoch(*args: Any) -> None:
            nonlocal stream_calls
            del args
            source = source_ref[0]
            stream_calls += 1
            if stream_calls == 1:
                source.completed_epoch = 0
            else:
                next_epoch_seen.append(source.completed_epoch)
                source.exiting = True

        config = PipelineConfig(
            src=SourceConfig([]),
            pipes=[],
            sink=SinkConfig(1),
        )
        with (
            mock.patch(
                "spdl.pipeline._build.build_pipeline", side_effect=_build_pipeline
            ),
            mock.patch(
                "spdl.pipeline._subprocess_pipeline_pool._stream_results",
                side_effect=_stream_one_epoch,
            ),
        ):
            _run_continuous(queue.Queue(), _OutputQueue(), 7, config, {}, 1)

        self.assertEqual(completed_when_published, [None])
        self.assertEqual(next_epoch_seen, [1])
        self.assertEqual(messages, [(_EPOCH_DONE, (0, 7)), (_DONE, None)])


class SerializationFailureRelayTest(unittest.TestCase):
    def test_serialization_error_contains_hostile_exception_formatting(self) -> None:
        """Broken exception formatting cannot escape the queue feeder callback."""

        class _BrokenNameMeta(type):
            def __getattribute__(cls, name: str) -> Any:
                if name == "__name__":
                    raise RuntimeError("type name failed")
                return super().__getattribute__(name)

        class _UnformattableError(Exception, metaclass=_BrokenNameMeta):
            def __str__(self) -> str:
                raise RuntimeError("error detail failed")

        formatted = _serialization_error("output", _UnformattableError())

        self.assertEqual(
            str(formatted),
            "Fused subprocess output could not be serialized: "
            "<exception type unavailable>: <error message unavailable>",
        )

    def test_serialization_error_truncates_large_exception_detail(self) -> None:
        """A hostile payload cannot create an unbounded fallback message."""
        detail = "x" * (_SERIALIZATION_ERROR_DETAIL_LIMIT + 1)

        formatted = _serialization_error("output", RuntimeError(detail))

        self.assertTrue(
            str(formatted).endswith(
                "x" * _SERIALIZATION_ERROR_DETAIL_LIMIT + "... <truncated>"
            )
        )

    def test_full_output_queue_does_not_block_feeder_error_handler(self) -> None:
        """The output feeder never waits for capacity on its own bounded queue."""
        occupied = object()
        out_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        out_q.put_nowait(occupied)
        serialization_failure = _serialization_failure_state()
        callback_done = threading.Event()

        def _invoke_handler() -> None:
            try:
                _handle_queue_feeder_error(
                    out_q,
                    serialization_failure,
                    "output",
                    _RESULT,
                    TypeError("unpicklable result"),
                    (_RESULT, [object()]),
                )
            finally:
                callback_done.set()

        callback = threading.Thread(target=_invoke_handler, daemon=True)
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ):
            callback.start()
            try:
                self.assertTrue(
                    callback_done.wait(timeout=1),
                    "feeder error handler blocked on its own full output queue",
                )
                self.assertTrue(serialization_failure.failed.is_set())
                self.assertTrue(serialization_failure.relay_failed.is_set())
            finally:
                self.assertIs(out_q.get_nowait(), occupied)
                callback.join(timeout=1)

    def test_control_message_feeder_errors_are_terminal(self) -> None:
        """A dropped control message cannot let collection report success."""

        class _HostileTuple(tuple[Any, ...]):
            def __len__(self) -> int:
                raise RuntimeError("message length failed")

        class _HostileKind(int):
            def __eq__(self, other: object) -> bool:
                raise RuntimeError("message kind comparison failed")

        for obj in (
            (_ERROR, RuntimeError("worker failed")),
            (_DONE, None),
            b"serialized control message",
            _HostileTuple((_RESULT, [object()])),
            (_HostileKind(_RESULT), [object()]),
        ):
            with self.subTest(obj_type=type(obj).__name__):
                out_q: queue.Queue[Any] = queue.Queue()
                serialization_failure = _serialization_failure_state()

                with mock.patch(
                    "spdl.pipeline._subprocess_pipeline_pool.traceback.print_exception"
                ) as print_exception:
                    _handle_queue_feeder_error(
                        out_q,
                        serialization_failure,
                        "output",
                        _RESULT,
                        RuntimeError("control message was dropped"),
                        obj,
                    )

                self.assertTrue(serialization_failure.failed.is_set())
                self.assertTrue(serialization_failure.relay_failed.is_set())
                self.assertTrue(serialization_failure.non_payload_failed.is_set())
                self.assertTrue(out_q.empty())
                print_exception.assert_called_once()

    def test_feeder_error_diagnostic_formatting_cannot_escape(self) -> None:
        """A broken traceback formatter cannot kill the queue's feeder thread."""
        serialization_failure = _serialization_failure_state()

        with mock.patch(
            "spdl.pipeline._subprocess_pipeline_pool.traceback.print_exception",
            side_effect=RuntimeError("formatting failed"),
        ):
            _handle_queue_feeder_error(
                queue.Queue(),
                serialization_failure,
                "output",
                _RESULT,
                RuntimeError("unrelated feeder error"),
                object(),
            )

        self.assertTrue(serialization_failure.failed.is_set())
        self.assertTrue(serialization_failure.relay_failed.is_set())
        self.assertTrue(serialization_failure.non_payload_failed.is_set())

    def test_failed_error_fallback_marks_relay_failed(self) -> None:
        """Failure of the safe ``_ERROR`` tuple ends the relay grace promptly."""
        out_q: queue.Queue[Any] = queue.Queue()
        serialization_failure = _serialization_failure_state(failed=True)

        with mock.patch(
            "spdl.pipeline._subprocess_pipeline_pool.traceback.print_exception"
        ) as print_exception:
            _handle_queue_feeder_error(
                out_q,
                serialization_failure,
                "output",
                _RESULT,
                RuntimeError("fallback feeder error"),
                (_ERROR, RuntimeError("serialization detail")),
            )

        self.assertTrue(serialization_failure.relay_failed.is_set())
        self.assertTrue(serialization_failure.non_payload_failed.is_set())
        self.assertTrue(out_q.empty())
        print_exception.assert_called_once()

    def test_missing_feeder_hook_uses_synchronous_fallback(self) -> None:
        """A queue without CPython's private hook still transfers valid messages."""
        raw_q: queue.Queue[Any] = queue.Queue()
        failure = _serialization_failure_state()
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ):
            checked_q = _install_queue_feeder_error_handler(
                raw_q,
                queue.Queue(),
                failure,
                "output",
                _RESULT,
            )
        checked_q.put((_RESULT, [1]))

        wire_message = raw_q.get_nowait()
        self.assertEqual(
            ForkingPickler.loads(ForkingPickler.dumps(wire_message)),
            (_RESULT, [1]),
        )
        self.assertFalse(failure.failed.is_set())

    def test_synchronous_fallback_reports_unpicklable_payload(self) -> None:
        """The portable fallback preserves serialization-failure reporting."""
        raw_q: queue.Queue[Any] = queue.Queue()
        out_q: queue.Queue[Any] = queue.Queue()
        failure = _serialization_failure_state()
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ):
            checked_q = _install_queue_feeder_error_handler(
                raw_q,
                out_q,
                failure,
                "output",
                _RESULT,
            )

        checked_q.put((_RESULT, [lambda: None]))

        kind, error = out_q.get_nowait()
        self.assertEqual(kind, _ERROR)
        self.assertIn("could not be serialized", str(error))
        self.assertTrue(failure.failed.is_set())
        self.assertTrue(raw_q.empty())

    def test_synchronous_fallback_serializes_once_across_full_retries(self) -> None:
        """Backpressure retries cannot invoke a user reducer more than once."""
        calls: list[int] = []

        class _StatefulPayload:
            def __reduce__(self) -> Any:
                calls.append(1)
                return int, (7,)

        raw_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        raw_q.put_nowait(object())
        failure = _serialization_failure_state()
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ):
            checked_q = _install_queue_feeder_error_handler(
                raw_q,
                queue.Queue(),
                failure,
                "output",
                _RESULT,
            )
        message = (_RESULT, [_StatefulPayload()])

        with self.assertRaises(queue.Full):
            checked_q.put(message, timeout=0)
        self.assertEqual(calls, [1])
        raw_q.get_nowait()
        checked_q.put(message, timeout=0)

        self.assertEqual(calls, [1])
        wire_message = raw_q.get_nowait()
        self.assertEqual(
            ForkingPickler.loads(ForkingPickler.dumps(wire_message)),
            (_RESULT, [7]),
        )
        self.assertFalse(failure.failed.is_set())

    def test_synchronous_fallback_releases_abandoned_full_retry(self) -> None:
        """Teardown releases an object retained for a backpressure retry."""
        serialized = threading.Event()

        class _Payload:
            def __reduce__(self) -> Any:
                serialized.set()
                return int, (7,)

        raw_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        raw_q.put_nowait(object())
        failure = _serialization_failure_state()
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ):
            checked_q = _install_queue_feeder_error_handler(
                raw_q,
                queue.Queue(),
                failure,
                "output",
                _RESULT,
            )
        payload = _Payload()
        payload_ref = weakref.ref(payload)
        message = (_RESULT, [payload])
        stop = threading.Event()
        put_thread = threading.Thread(
            target=_subprocess_pipe._put,
            args=(checked_q, message, stop),
        )

        put_thread.start()
        try:
            self.assertTrue(serialized.wait(timeout=1))
        finally:
            stop.set()
            put_thread.join(timeout=1)

        self.assertFalse(put_thread.is_alive())
        del message, payload
        gc.collect()
        self.assertIsNone(payload_ref())

    def test_synchronous_fallback_propagates_base_exceptions(self) -> None:
        """Fallback serialization cannot swallow process-termination signals."""

        class _InterruptingPayload:
            def __reduce__(self) -> Any:
                raise KeyboardInterrupt

        failure = _serialization_failure_state()
        with self.assertLogs(
            "spdl.pipeline._subprocess_pipeline_pool", level="WARNING"
        ):
            checked_q = _install_queue_feeder_error_handler(
                queue.Queue(),
                queue.Queue(),
                failure,
                "output",
                _RESULT,
            )

        with self.assertRaises(KeyboardInterrupt):
            checked_q.put((_RESULT, [_InterruptingPayload()]))
        self.assertFalse(failure.failed.is_set())

    def test_non_payload_feeder_failure_uses_queue_diagnostic(self) -> None:
        """A dropped control/transport message is not blamed on user payloads."""
        failure = _serialization_failure_state(
            failed=True,
            relay_failed=True,
            non_payload_failed=True,
        )

        async def _scenario() -> None:
            await _subprocess_pipe._check_serialization_error_relay_after_progress(
                queue.Queue(),
                mock.Mock(),
                (("output", failure),),
                None,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "queue feeder failed while sending a protocol or transport message",
        ) as raised:
            asyncio.run(_scenario())
        self.assertNotIn("serialization failed", str(raised.exception))

    def test_missing_detailed_error_identifies_boundary_after_grace(self) -> None:
        """A bounded generic fallback identifies the failed queue boundary."""
        for boundary in ("input", "output"):
            with self.subTest(boundary=boundary):
                input_failure = _serialization_failure_state(failed=boundary == "input")
                output_failure = _serialization_failure_state(
                    failed=boundary == "output"
                )
                failures = (
                    ("input", input_failure),
                    ("output", output_failure),
                )
                with mock.patch.object(
                    _subprocess_pipe.time,
                    "monotonic",
                    side_effect=[10.0, 15.0],
                ):
                    deadline = _check_serialization_error_relay(
                        _serialization_failure_boundaries(failures), None
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        f"Fused subprocess {boundary} serialization failed",
                    ):
                        _check_serialization_error_relay(
                            _serialization_failure_boundaries(failures), deadline
                        )

    def test_missing_detailed_error_identifies_all_failed_boundaries(self) -> None:
        """The generic fallback names every queue whose feeder reported failure."""
        input_failure = _serialization_failure_state(failed=True)
        output_failure = _serialization_failure_state(failed=True)

        self.assertEqual(
            _serialization_failure_boundaries(
                (("input", input_failure), ("output", output_failure))
            ),
            "input and output",
        )

    def test_detailed_error_can_arrive_after_multiple_empty_polls(self) -> None:
        """A delayed detailed feeder error wins over the generic fallback."""

        async def scenario() -> None:
            input_failure = _serialization_failure_state(failed=True)
            failures = (
                ("input", input_failure),
                ("output", _serialization_failure_state()),
            )
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )
            responses = [
                None,
                None,
                (_ERROR, RuntimeError("detailed input failure")),
                (_DONE, None),
            ]
            with (
                ThreadPoolExecutor(max_workers=1) as executor,
                mock.patch.object(
                    _subprocess_pipe,
                    "_drain_one",
                    side_effect=responses,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "detailed input failure"):
                    await _collect(
                        object(),
                        1,
                        output_queue,
                        executor,
                        asyncio.Event(),
                        asyncio.Event(),
                        failures,
                    )

        asyncio.run(scenario())

    def test_dropped_done_after_worker_error_preserves_original_error(self) -> None:
        """A lost terminal marker cannot stall or mask an earlier worker error."""

        async def scenario() -> None:
            failure = _serialization_failure_state(failed=True, relay_failed=True)
            original_error = RuntimeError("original worker failure")
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )
            with (
                ThreadPoolExecutor(max_workers=1) as executor,
                mock.patch.object(
                    _subprocess_pipe,
                    "_drain_one",
                    side_effect=[(_ERROR, original_error), None],
                ),
                mock.patch.object(_subprocess_pipe, "_WORKER_STALL_TIMEOUT", -1.0),
            ):
                with self.assertRaisesRegex(RuntimeError, "original worker failure"):
                    await _collect(
                        object(),
                        1,
                        output_queue,
                        executor,
                        asyncio.Event(),
                        asyncio.Event(),
                        (("output", failure),),
                    )

        asyncio.run(scenario())

    def test_stall_after_worker_error_preserves_original_error(self) -> None:
        """The stall guard cannot replace an error already received from a worker."""

        async def scenario() -> None:
            original_error = RuntimeError("original worker failure")
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )
            with (
                ThreadPoolExecutor(max_workers=1) as executor,
                mock.patch.object(
                    _subprocess_pipe,
                    "_drain_one",
                    side_effect=[(_ERROR, original_error), None],
                ),
                mock.patch.object(_subprocess_pipe, "_WORKER_STALL_TIMEOUT", -1.0),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "original worker failure"
                ) as raised:
                    await _collect(
                        object(),
                        1,
                        output_queue,
                        executor,
                        asyncio.Event(),
                        asyncio.Event(),
                    )
                self.assertIs(raised.exception, original_error)
                self.assertIsNone(raised.exception.__cause__)

        asyncio.run(scenario())

    def test_queue_teardown_after_worker_error_preserves_original_error(self) -> None:
        """Output-queue teardown cannot replace an error already received from a worker."""

        async def scenario(queue_error: BaseException) -> None:
            out_q = mock.Mock()
            original_error = RuntimeError("original worker failure")
            out_q.get.side_effect = [
                (_ERROR, original_error),
                queue_error,
            ]
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                with self.assertRaisesRegex(
                    RuntimeError, "original worker failure"
                ) as raised:
                    await _collect(
                        out_q,
                        1,
                        output_queue,
                        executor,
                        asyncio.Event(),
                        asyncio.Event(),
                    )
                self.assertIs(raised.exception, original_error)
                self.assertIsNone(raised.exception.__cause__)
                self.assertTrue(raised.exception.__suppress_context__)

        for queue_error in (
            RuntimeError("interpreter queue is closed"),
            ValueError("output queue is closed"),
        ):
            with self.subTest(error=type(queue_error).__name__):
                asyncio.run(scenario(queue_error))
