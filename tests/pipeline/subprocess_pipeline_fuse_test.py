# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import json
import os
import queue as _queue
import sys
import tempfile
import time
import unittest
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from spdl.pipeline import (
    AsyncQueue,
    Pipeline,
    PipelineBuilder,
    PipelineFailure,
    run_pipeline_in_subprocess,
    TaskHook,
)
from spdl.pipeline._components import _subprocess_pipe
from spdl.pipeline._components._common import StageInfo
from spdl.pipeline.config import set_default_hook_class, set_default_queue_class


def add_one(x: int) -> int:
    return x + 1


def _raise_initializer() -> None:
    raise RuntimeError("initializer boom")


def times_two(x: int) -> int:
    return x * 2


def boom(x: int) -> int:
    raise ValueError("boom")


class _Unpicklable:
    """An object that refuses to be pickled, to prove it never crosses a process boundary."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __reduce__(self) -> Any:
        raise TypeError("_Unpicklable must not be pickled")


def wrap(x: int) -> _Unpicklable:
    return _Unpicklable(x + 1)


def unwrap(o: _Unpicklable) -> int:
    return o.value * 2


def dup(x: int) -> Iterator[int]:
    """A sync-generator op: yields two values per input (1->2 fan-out)."""
    yield x * 2
    yield x * 2 + 1


async def adup(x: int) -> AsyncIterator[int]:
    """An async-generator op: yields two values per input (1->2 fan-out)."""
    yield x
    yield x + 1


def _run(pipeline: Pipeline[Any], timeout: float = 60.0) -> list[Any]:
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=timeout))


class SubprocessPipelineFuseTest(unittest.TestCase):
    def test_two_pool_stages_fused_match_unfused(self) -> None:
        """A fused two-stage process-pool pipeline matches the unfused result."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))

        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=3)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_unpicklable_intermediate_passes_through_fused(self) -> None:
        """A fused run keeps the op->op handoff in-process, so an unpicklable value works."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))

        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(wrap, executor=ex, concurrency=2)
            .pipe(unwrap, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_generator_op_fused(self) -> None:
        """A generator op is fusable, fanning out 1->N inside the worker sub-pipeline."""
        n = 8
        # dup yields {2x, 2x+1}; add_one shifts each by one, covering 1..2n exactly.
        ref = list(range(1, 2 * n + 1))
        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(dup, executor=ex, concurrency=2)
            .pipe(add_one, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_async_generator_op_composes_with_fused(self) -> None:
        """An async-generator op (main-process) fans out downstream of a fused run."""
        n = 8
        # add_one+times_two fuse to (x+1)*2; adup then yields {v, v+1}, covering 2..2n+1.
        ref = list(range(2, 2 * n + 2))
        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .pipe(adup)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_aggregate_between_pool_stages_not_absorbed(self) -> None:
        """An aggregate between two pool stages stays in the main process (not absorbed).

        Its main-process batching semantics are unchanged: each batch holds exactly
        ``aggregate``'s size.
        """
        n = 12
        ex = ProcessPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .aggregate(3)
            .pipe(len, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        # Each aggregated batch has exactly 3 items, so every produced value is 3.
        out = _run(fused)
        self.assertTrue(all(v == 3 for v in out))
        self.assertEqual(sum(out), n)

    @unittest.skipUnless(
        sys.version_info >= (3, 14), "InterpreterPoolExecutor requires Python 3.14+"
    )
    def test_interpreter_pool_stages_fused(self) -> None:
        """Stages sharing an InterpreterPoolExecutor are recognized and fused."""
        from concurrent.futures import InterpreterPoolExecutor  # pyre-ignore[21]

        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = InterpreterPoolExecutor(max_workers=2)
        fused = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        self.assertEqual(sorted(_run(fused)), ref)

    def test_initializer_failure_surfaces_not_hangs(self) -> None:
        """A worker initializer that raises surfaces as a failure instead of hanging.

        A real failure surfaces as :py:class:`PipelineFailure`; a hung pipeline would instead
        raise :py:class:`TimeoutError` from the finite ``get_iterator`` timeout, so asserting on
        ``PipelineFailure`` proves the failed initializer is reported rather than wedging the
        collector forever.
        """
        ex = ProcessPoolExecutor(max_workers=2, initializer=_raise_initializer)
        fused = (
            PipelineBuilder()
            .add_source(range(2))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(4)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with self.assertRaises(PipelineFailure):
            _run(fused, timeout=30.0)

    def test_flag_off_is_unaffected(self) -> None:
        """Without the flag, the same pipeline runs the stages normally and matches."""
        n = 10
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=False)
        )
        self.assertEqual(sorted(_run(pipeline)), ref)


class ContinuousFuseTest(unittest.TestCase):
    """Fusion with a continuous (multi-epoch) source."""

    def test_multi_epoch_correct(self) -> None:
        """A continuous fused pipeline yields the correct set each epoch."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=3)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(3):  # three epochs from the same warm worker pool
                epoch = sorted(pipeline.get_iterator(timeout=60))
                self.assertEqual(epoch, ref)

    def test_unpicklable_intermediate_multi_epoch(self) -> None:
        """The unpicklable op->op handoff keeps working across epochs."""
        n = 10
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(wrap, executor=ex, concurrency=2)
            .pipe(unwrap, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_aggregate_not_absorbed_multi_epoch(self) -> None:
        """A non-absorbed aggregate keeps single-flush-per-epoch semantics across epochs.

        Because the aggregate runs in the main process (not per worker), each epoch produces
        full-size batches plus one combined partial, and every item is accounted for.
        """
        n = 12
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .aggregate(3)
            .pipe(len, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):  # each epoch's batches cover exactly n items
                out = list(pipeline.get_iterator(timeout=60))
                self.assertEqual(sum(out), n)

    def test_fewer_items_than_workers(self) -> None:
        """An epoch with fewer items than workers still completes (some workers run empty)."""
        n = 2
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=4)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_generator_op_multi_epoch(self) -> None:
        """A fused generator op keeps its 1->N fan-out correct across epochs."""
        n = 8
        ref = list(range(1, 2 * n + 1))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(dup, executor=ex, concurrency=2)
            .pipe(add_one, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(3):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_async_generator_op_multi_epoch(self) -> None:
        """A main-process async-generator op fans out downstream of a fused run each epoch."""
        n = 8
        ref = list(range(2, 2 * n + 2))
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .pipe(adup)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(3):
                self.assertEqual(sorted(pipeline.get_iterator(timeout=60)), ref)

    def test_continuous_op_failure_does_not_deadlock(self) -> None:
        """Op failures (dropped per SPDL default) still let each epoch's barrier complete.

        ``boom`` raises on every item, so the fused workers produce no results; the test checks
        the continuous epoch barrier still completes each epoch (workers report the boundary
        with zero results) instead of deadlocking, matching unfused drop-on-failure behavior.
        """
        n = 8
        ex = ProcessPoolExecutor(max_workers=2)
        pipeline = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(boom, executor=ex, concurrency=2)
            .add_sink(n)
            .build(num_threads=4, fuse_subprocess_stages=True)
        )
        with pipeline.auto_stop():
            for _ in range(2):
                self.assertEqual(list(pipeline.get_iterator(timeout=60)), [])

    def test_continuous_in_subprocess(self) -> None:
        """Continuous fusion composes with run_pipeline_in_subprocess across epochs."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        config = (
            PipelineBuilder()
            .add_source(range(n), continuous=True)
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=2)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(
            config, num_threads=4, fuse_subprocess_stages=True
        )
        for _ in range(3):  # one epoch per iteration
            self.assertEqual(sorted(src), ref)


class FuseInSubprocessTest(unittest.TestCase):
    """Fusion composed with whole-pipeline subprocess execution."""

    def test_fused_run_in_subprocess_matches(self) -> None:
        """A fused config run via run_pipeline_in_subprocess yields the same result."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        config = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(add_one, executor=ex, concurrency=2)
            .pipe(times_two, executor=ex, concurrency=3)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(
            config, num_threads=4, fuse_subprocess_stages=True
        )
        self.assertEqual(sorted(src), ref)

    def test_fused_unpicklable_intermediate_in_subprocess(self) -> None:
        """The fused handle survives the trip into the pipeline subprocess and the
        unpicklable op->op handoff stays inside a worker."""
        n = 12
        ref = sorted((x + 1) * 2 for x in range(n))
        ex = ProcessPoolExecutor(max_workers=2)
        config = (
            PipelineBuilder()
            .add_source(range(n))
            .pipe(wrap, executor=ex, concurrency=2)
            .pipe(unwrap, executor=ex, concurrency=2)
            .add_sink(n)
            .get_config()
        )
        src = run_pipeline_in_subprocess(
            config, num_threads=4, fuse_subprocess_stages=True
        )
        self.assertEqual(sorted(src), ref)


class FeedAbortTest(unittest.TestCase):
    """The bridge feeder must wind down promptly when the collector signals abort."""

    def test_feed_ends_session_when_aborted_while_idle(self) -> None:
        """A feeder parked on an empty input queue still emits the per-worker _SESSION_END.

        On a worker error the collector sets ``abort`` while the feeder is typically blocked
        waiting on a slow/idle upstream. The feeder must wake and send one ``_SESSION_END`` per
        worker so the collector can drain every ``_DONE`` instead of hanging until its stall
        timeout. Driving ``_feed`` directly keeps the abort-while-idle race deterministic.
        """
        num_workers = 3

        async def _scenario() -> list[Any]:
            in_q: _queue.Queue[Any] = _queue.Queue()
            input_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="input")
            )  # stays empty -> get() blocks
            abort = asyncio.Event()
            feeder_idle = asyncio.Event()
            with ThreadPoolExecutor(max_workers=2) as ex:
                task = asyncio.ensure_future(
                    _subprocess_pipe._feed(
                        input_queue, in_q, num_workers, ex, abort, feeder_idle
                    )
                )
                await asyncio.sleep(0.1)  # let the feeder park on input_queue.get()
                self.assertFalse(task.done(), "feeder should be parked on empty queue")
                abort.set()
                await asyncio.wait_for(task, timeout=5.0)
            return [in_q.get_nowait() for _ in range(in_q.qsize())]

        msgs = asyncio.run(_scenario())
        self.assertEqual(msgs, [(_subprocess_pipe._SESSION_END, None)] * num_workers)


class StallGuardTest(unittest.TestCase):
    """The collector's stall guard against an abruptly-dead worker."""

    def test_check_stall_raises_past_timeout(self) -> None:
        """``_check_stall`` raises once no message has arrived for longer than the bound."""
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 0.0
        try:
            with self.assertRaises(TimeoutError):
                _subprocess_pipe._check_stall(time.monotonic() - 1.0)
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig

    def test_check_stall_quiet_within_timeout(self) -> None:
        """``_check_stall`` does not raise while progress is within the bound."""
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 60.0
        try:
            _subprocess_pipe._check_stall(time.monotonic())  # should not raise
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig

    def test_collect_suppresses_stall_while_feeder_idle(self) -> None:
        """An idle feeder suppresses the collector's stall guard during input starvation.

        With the timeout pinned to zero, any stall check on an empty queue would trip instantly;
        the collector must instead keep draining while ``feeder_idle`` is set (nothing dispatched,
        no worker message due) and still finish once the worker reports ``_DONE``.
        """
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 0.0

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )
            abort = asyncio.Event()
            feeder_idle = asyncio.Event()
            feeder_idle.set()  # feeder parked on an idle upstream -> no message expected
            with ThreadPoolExecutor(max_workers=2) as ex:
                task = asyncio.ensure_future(
                    _subprocess_pipe._collect(
                        out_q, 1, output_queue, ex, abort, feeder_idle
                    )
                )
                await asyncio.sleep(
                    0.6
                )  # several empty poll cycles; must not trip the guard
                self.assertFalse(
                    task.done(), "idle feeder must suppress the stall guard"
                )
                out_q.put((_subprocess_pipe._DONE, None))
                await asyncio.wait_for(task, timeout=5.0)

        try:
            asyncio.run(_scenario())
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig


# Set inside each fused worker process by ``_install_recording_hooks`` (the executor
# initializer). The worker is a separate process, so the recording hooks below cannot share
# in-memory counters with the test; each instead writes a small JSON file into this directory,
# which the test reads back to prove the hooks fired inside the worker.
_EVIDENCE_DIR: str | None = None


def _write_evidence(record: dict[str, Any]) -> None:
    if (d := _EVIDENCE_DIR) is None:
        return
    # ``iid`` (the recorder's id) keeps each instance's file distinct within a process; ``pid``
    # keeps them distinct across worker processes, so no two writers ever race on one path.
    path = os.path.join(d, f"{record['kind']}-{record['pid']}-{record['iid']}.json")
    with open(path, "w") as f:
        f.write(json.dumps(record))


class _RecordingTaskHook(TaskHook):
    """A ``TaskHook`` that records how many tasks ran in its (subprocess) stage."""

    def __init__(self, info: Any, interval: float = -1) -> None:
        self.info = info
        self.n_tasks = 0

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        try:
            yield
        finally:
            _write_evidence(
                {
                    "kind": "task",
                    "name": str(self.info),
                    "pid": os.getpid(),
                    "iid": id(self),
                    "n_tasks": self.n_tasks,
                }
            )

    @asynccontextmanager
    async def task_hook(self, input_item: Any = None) -> AsyncIterator[None]:
        self.n_tasks += 1
        yield


class _RecordingQueue(AsyncQueue):
    """An ``AsyncQueue`` whose ``stage_hook`` records that it ran in the (subprocess) stage."""

    def __init__(
        self, info: Any, *, buffer_size: int = 1, interval: float = -1
    ) -> None:
        super().__init__(info, buffer_size=buffer_size)
        self.n_get = 0

    async def get(self) -> object:
        item = await super().get()
        self.n_get += 1
        return item

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        try:
            yield
        finally:
            _write_evidence(
                {
                    "kind": "queue",
                    "name": str(self.info),
                    "pid": os.getpid(),
                    "iid": id(self),
                    "n_get": self.n_get,
                }
            )


def _install_recording_hooks(evidence_dir: str) -> None:
    """Executor initializer: runs inside each fused worker before its sub-pipeline is built.

    Fusion reads this off the ``ProcessPoolExecutor`` (``_pool_params``) and runs it in every
    worker process, so the worker's nested ``build_pipeline`` — which is given no explicit
    ``task_hook_factory``/``queue_class`` — picks up these recording classes as its defaults.
    """
    global _EVIDENCE_DIR
    _EVIDENCE_DIR = evidence_dir
    set_default_hook_class(_RecordingTaskHook)
    set_default_queue_class(_RecordingQueue)


class FuseHookTest(unittest.TestCase):
    """``TaskHook`` and the queue ``stage_hook`` fire inside the fused worker subprocess.

    The fused run executes as a nested pipeline inside main-process-owned worker processes; the
    per-stage hooks/stats fire there, not in the bridge stage. These tests install recording
    hook/queue classes as the worker defaults (via the pool's ``initializer``) and assert, from
    the evidence files those recorders leave behind, that both fired in a non-main process.
    """

    def _read_evidence(self, evidence_dir: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for name in os.listdir(evidence_dir):
            with open(os.path.join(evidence_dir, name)) as f:
                records.append(json.loads(f.read()))
        return records

    def _assert_hooks_fired(self, records: list[dict[str, Any]], n: int) -> None:
        task_records = [r for r in records if r["kind"] == "task"]
        queue_records = [r for r in records if r["kind"] == "queue"]

        self.assertTrue(task_records, "TaskHook never fired in any fused worker")
        self.assertTrue(
            queue_records, "queue stage_hook never fired in any fused worker"
        )

        # Every record came from a worker process, never the main/test process — this is what
        # proves the hooks fired "in the subprocess".
        main_pid = os.getpid()
        for r in records:
            self.assertNotEqual(r["pid"], main_pid)

        # The two fused pipe stages each see every item once: ``add_one`` over n items, then
        # ``times_two`` over n items => 2n ``task_hook`` invocations, summed across workers.
        self.assertEqual(sum(r["n_tasks"] for r in task_records), 2 * n)

        # The fused sub-pipeline's queues carried the data inside the worker(s).
        self.assertGreaterEqual(sum(r["n_get"] for r in queue_records), n)

    def test_hooks_fire_in_fused_workers(self) -> None:
        """Hooks fire in the fused workers spawned by ``PipelineBuilder.build``."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        with tempfile.TemporaryDirectory() as evidence_dir:
            ex = ProcessPoolExecutor(
                max_workers=2,
                initializer=_install_recording_hooks,
                initargs=(evidence_dir,),
            )
            fused = (
                PipelineBuilder()
                .add_source(range(n))
                .pipe(add_one, executor=ex, concurrency=2)
                .pipe(times_two, executor=ex, concurrency=2)
                .add_sink(n)
                .build(num_threads=4, fuse_subprocess_stages=True)
            )
            self.assertEqual(sorted(_run(fused)), ref)
            self._assert_hooks_fired(self._read_evidence(evidence_dir), n)

    def test_hooks_fire_in_fused_workers_via_subprocess(self) -> None:
        """Hooks fire in the fused workers when the run is driven from a pipeline subprocess."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        with tempfile.TemporaryDirectory() as evidence_dir:
            ex = ProcessPoolExecutor(
                max_workers=2,
                initializer=_install_recording_hooks,
                initargs=(evidence_dir,),
            )
            config = (
                PipelineBuilder()
                .add_source(range(n))
                .pipe(add_one, executor=ex, concurrency=2)
                .pipe(times_two, executor=ex, concurrency=2)
                .add_sink(n)
                .get_config()
            )
            src = run_pipeline_in_subprocess(
                config, num_threads=4, fuse_subprocess_stages=True
            )
            # Each worker writes its evidence when its nested pipeline tears down at end of the
            # session, before the run completes — so the files are present once iteration ends.
            self.assertEqual(sorted(src), ref)
            self._assert_hooks_fired(self._read_evidence(evidence_dir), n)
