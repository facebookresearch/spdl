# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Per-stage hooks and queues fire inside the workers of a ``.to()`` region.

Ported from the removed ``subprocess_pipeline_fuse_test.py``: the region worker runs a nested
pipeline, so its default task hooks and queues must fire *in the worker process*. The worker is a
separate process and cannot share in-memory counters with the test, so recording hooks write
small JSON files that the test reads back as evidence.
"""

import itertools
import json
import os
import tempfile
import threading
import time
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from spdl.pipeline import AsyncQueue, Pipeline, PipelineBuilder, TaskHook
from spdl.pipeline.config import set_default_hook_class, set_default_queue_class
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig


def add_one(x: int) -> int:
    return x + 1


def times_two(x: int) -> int:
    return x * 2


def _run(pipeline: Pipeline[Any], timeout: float = 60.0) -> list[Any]:
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=timeout))


# Set inside each region worker process by ``_install_recording_hooks`` (the region initializer).
# The worker is a separate process, so the recording hooks below cannot share in-memory counters
# with the test; each instead writes a small JSON file into this directory, which the test reads
# back to prove the hooks fired inside the worker.
_EVIDENCE_DIR: str | None = None

# A per-process monotonic counter for evidence filenames. ``id(self)`` is only unique among
# simultaneously-live objects -- CPython recycles addresses, so two recorders created and
# destroyed in sequence could collide on one path and clobber each other's count. This counter
# (combined with ``pid``) is guaranteed distinct for every write within a process.
_EVIDENCE_SEQ = itertools.count()
_EVIDENCE_SEQ_LOCK = threading.Lock()


def _write_evidence(record: dict[str, Any]) -> None:
    if (d := _EVIDENCE_DIR) is None:
        return
    with _EVIDENCE_SEQ_LOCK:
        seq = next(_EVIDENCE_SEQ)
    # ``pid`` proves the write came from a worker process (never the main/test process); ``seq``
    # keeps each writer's file distinct within that process, so no two writers ever race on one
    # path. Write to a temp file then atomically rename, so a reader polling mid-write never sees
    # a truncated/partial file.
    path = os.path.join(d, f"{record['kind']}-{record['pid']}-{seq}.json")
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(record))
        os.replace(tmp, path)
    except OSError:
        # This runs inside the worker's stage_hook ``finally`` (stage teardown). It must never
        # raise into the stage: an I/O failure here -- e.g. the evidence ``TemporaryDirectory``
        # already torn down in a teardown race -- would otherwise crash the stage and drop items
        # from the data path the test is validating.
        pass


class _RecordingTaskHook(TaskHook):
    """A ``TaskHook`` that records how many tasks ran in its (region-worker) stage."""

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
    """An ``AsyncQueue`` whose ``stage_hook`` records that it ran in the (region-worker) stage."""

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
    """Region-worker initializer: runs inside each worker before its sub-pipeline is built.

    The region reads this off the :py:class:`ProcessPoolExecutorConfig` and runs it in every worker
    process, so the worker's nested ``build_pipeline`` -- given no explicit
    ``task_hook_factory``/``queue_class`` -- picks up these recording classes as its defaults.
    """
    global _EVIDENCE_DIR
    _EVIDENCE_DIR = evidence_dir
    set_default_hook_class(_RecordingTaskHook)
    set_default_queue_class(_RecordingQueue)


class RegionHookTest(unittest.TestCase):
    """Per-stage hooks/queues fire inside the workers of a ``.to()`` region."""

    def test_hooks_fire_in_region_workers(self) -> None:
        """Hooks fire in the workers spawned for a
        ``.to(ProcessPoolExecutorConfig(...))`` region."""
        n = 16
        ref = sorted((x + 1) * 2 for x in range(n))
        with tempfile.TemporaryDirectory() as evidence_dir:
            pipeline = (
                PipelineBuilder()
                .add_source(range(n))
                .to(
                    ProcessPoolExecutorConfig(
                        max_workers=2,
                        initializer=_install_recording_hooks,
                        initargs=(evidence_dir,),
                    )
                )
                .pipe(add_one, concurrency=2)
                .pipe(times_two, concurrency=2)
                .to(MAIN_PROCESS)
                .add_sink(n)
                .build(num_threads=4)
            )
            self.assertEqual(sorted(_run(pipeline)), ref)
            self._assert_hooks_fired(self._await_evidence(evidence_dir, n), n)

    def _read_evidence(self, evidence_dir: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for name in os.listdir(evidence_dir):
            if not name.endswith(".json"):
                continue  # skip ``.json.tmp`` files still being written
            with open(os.path.join(evidence_dir, name)) as f:
                records.append(json.loads(f.read()))
        return records

    def _await_evidence(
        self, evidence_dir: str, n: int, timeout: float = 60.0
    ) -> list[dict[str, Any]]:
        """Re-read evidence until the workers' task total lands, then return the records.

        Each worker writes its evidence files during nested-pipeline teardown, a side channel
        with no happens-before relationship to the main iterator completing. Polling until the
        recorded task total reaches the expected ``2 * n`` closes that read-too-early window
        without masking regressions: on a genuine failure (hooks never fire) the poll times out
        and the caller's assertions report the shortfall.
        """
        deadline = time.monotonic() + timeout
        records = self._read_evidence(evidence_dir)
        while sum(r["n_tasks"] for r in records if r["kind"] == "task") < 2 * n:
            if time.monotonic() > deadline:
                break  # fall through; the assertions below report the shortfall
            time.sleep(0.02)
            records = self._read_evidence(evidence_dir)
        return records

    def _assert_hooks_fired(self, records: list[dict[str, Any]], n: int) -> None:
        task_records = [r for r in records if r["kind"] == "task"]
        queue_records = [r for r in records if r["kind"] == "queue"]

        self.assertTrue(task_records, "TaskHook never fired in any region worker")
        self.assertTrue(
            queue_records, "queue stage_hook never fired in any region worker"
        )

        # Every record came from a worker process, never the main/test process -- this is what
        # proves the hooks fired "in the worker".
        main_pid = os.getpid()
        for r in records:
            self.assertNotEqual(r["pid"], main_pid)

        # The two region pipe stages each see every item once: ``add_one`` over n items, then
        # ``times_two`` over n items => 2n ``task_hook`` invocations, summed across workers.
        self.assertEqual(sum(r["n_tasks"] for r in task_records), 2 * n)

        # The region sub-pipeline's queues carried the data inside the worker(s).
        self.assertGreaterEqual(sum(r["n_get"] for r in queue_records), n)


if __name__ == "__main__":
    unittest.main()
