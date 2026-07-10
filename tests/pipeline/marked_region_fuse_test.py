# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from collections.abc import Sequence
from typing import Any

from spdl.pipeline import build_pipeline, Pipeline
from spdl.pipeline.defs import (
    Aggregate,
    InterpreterPoolExecutorConfig,
    MAIN_PROCESS,
    Pipe,
    PipelineConfig,
    PlacementConfig,
    ProcessPoolExecutorConfig,
    SinkConfig,
    SourceConfig,
)


def add_one(x: int) -> int:
    return x + 1


def times_two(x: int) -> int:
    return x * 2


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


class _PidStamp:
    """A picklable item stages stamp their ``os.getpid()`` into, to prove process-locality."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.pids: dict[str, int] = {}


def stamp_region(item: _PidStamp) -> _PidStamp:
    """A region stage: records the pid it runs in."""
    item.pids["region"] = os.getpid()
    return item


def stamp_main(item: _PidStamp) -> _PidStamp:
    """A main-process stage (after the region closes): records the pid it runs in."""
    item.pids["main"] = os.getpid()
    return item


def _cfg(src: Any, pipes: Sequence[Any], buffer: int = 16) -> PipelineConfig[Any]:
    return PipelineConfig(
        src=SourceConfig(src), pipes=list(pipes), sink=SinkConfig(buffer)
    )


def _run(pipeline: Pipeline[Any], timeout: float = 60.0) -> list[Any]:
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=timeout))


class MarkedRegionFuseTest(unittest.TestCase):
    def test_region_of_two_pipes(self) -> None:
        """Two pipes inside a subprocess region produce the same result as running inline."""
        n = 16
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=2)),
                Pipe(add_one),
                Pipe(times_two),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=4)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    def test_aggregate_inside_region(self) -> None:
        """An aggregate stage inside a region is absorbed and runs in the worker."""
        config = _cfg(
            range(10),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),
                Aggregate(3),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        result = _run(pipeline)
        # add_one over 0..9, then batched by 3 (one worker preserves order).
        self.assertEqual(result, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]])

    def test_unpicklable_intermediate_stays_in_region(self) -> None:
        """An unpicklable value handed between two region stages never crosses a boundary."""
        n = 5
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(wrap),
                Pipe(unwrap),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    def test_region_runs_in_worker_main_runs_in_main(self) -> None:
        """A region stage runs in a worker process; a stage after MAIN_PROCESS runs in main."""
        config = _cfg(
            [_PidStamp(i) for i in range(4)],
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(stamp_region),
                PlacementConfig(target=MAIN_PROCESS),
                Pipe(stamp_main),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        results = _run(pipeline)
        self.assertEqual(len(results), 4)
        for item in results:
            self.assertNotEqual(item.pids["region"], os.getpid())
            self.assertEqual(item.pids["main"], os.getpid())

    def test_no_markers_is_unchanged(self) -> None:
        """A config with no markers builds and runs entirely in the main process."""
        n = 8
        config = _cfg(range(n), [Pipe(add_one), Pipe(times_two)])
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    def test_multiple_regions(self) -> None:
        """Two separate subprocess regions around a main-process stage both fuse correctly.

        Exercises the main -> subprocess -> main -> subprocess -> main transition cycle, so a
        bug carrying stale ``target``/``region`` state across ``_flush()`` calls would surface.
        """
        n = 8
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),  # region 1: x + 1
                PlacementConfig(target=MAIN_PROCESS),
                Pipe(times_two),  # main: (x + 1) * 2
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),  # region 2: (x + 1) * 2 + 1
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(
            sorted(_run(pipeline)), sorted((x + 1) * 2 + 1 for x in range(n))
        )

    def test_region_open_at_end_of_pipes(self) -> None:
        """A region left open at the end of the pipes is flushed (no closing MAIN_PROCESS).

        Covers the ``_flush()`` after the segmentation loop, which closes a region that runs to
        the end of ``pipes``; the sink still executes in the main process.
        """
        n = 8
        config = _cfg(
            range(n),
            [
                PlacementConfig(target=ProcessPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),
                Pipe(times_two),
            ],
        )
        pipeline = build_pipeline(config, num_threads=2)
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    def test_subinterpreter_region_not_yet_supported(self) -> None:
        """A subinterpreter region raises until the subinterpreter backend lands."""
        config = _cfg(
            range(4),
            [
                PlacementConfig(target=InterpreterPoolExecutorConfig(max_workers=1)),
                Pipe(add_one),
                PlacementConfig(target=MAIN_PROCESS),
            ],
        )
        with self.assertRaises(NotImplementedError):
            build_pipeline(config, num_threads=2)
