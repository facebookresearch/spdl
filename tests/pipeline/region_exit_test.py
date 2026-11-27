# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Interpreter-exit teardown of a held `.to(ProcessPoolExecutorConfig)` region pipeline.

A held-but-unstopped pipeline is stopped at interpreter shutdown by the hook that
:py:meth:`Pipeline.start` registers. This exercises that for the worker-region topology: the
scenario holds a `.to()` region pipeline (mimicking a framework keeping the dataloader), iterates
a few items, and returns WITHOUT stopping. Run in a child process: the child must exit cleanly
rather than hang joining the region's still-running worker pool at shutdown.
"""

import multiprocessing as mp
import time
import unittest
from typing import Any, Callable

from spdl.pipeline import PipelineBuilder
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

# Strong reference that outlives the scenario function inside the child process.
_HELD: dict[str, object] = {}


def _double(x: int) -> int:
    return x * 2


def _scenario_to_region(mp_ctx: str) -> None:
    """Hold a `.to(ProcessPoolExecutorConfig)` region pipeline; never stop it."""
    p = (
        PipelineBuilder()
        .add_source(range(1000), continuous=True)
        .aggregate(8)
        .to(ProcessPoolExecutorConfig(max_workers=2, mp_context=mp_ctx))
        .pipe(_double)
        .to(MAIN_PROCESS)
        .add_sink(4)
        .build(num_threads=2)
    )
    _HELD["dl"] = p
    for i, _ in enumerate(p.get_iterator(timeout=60)):
        if i >= 10:
            break


def _wait_long(x: bytes) -> bytes:
    time.sleep(60)
    return x


def _scenario_full_worker_input(mp_ctx: str) -> None:
    """Hold a region pipeline whose busy workers leave their input pipes full; never stop it.

    Each item is larger than a pipe's buffer, so the feeder thread of a worker's input queue
    blocks writing the first item the worker does not read.
    """
    p = (
        PipelineBuilder()
        .add_source(bytes(1 << 20) for _ in range(16))
        .to(ProcessPoolExecutorConfig(max_workers=2, mp_context=mp_ctx))
        .pipe(_wait_long, concurrency=2)
        .to(MAIN_PROCESS)
        .add_sink(4)
        .build(num_threads=2)
    )
    _HELD["dl"] = p
    p.start()
    time.sleep(3)


def _scenario_stopped_region(mp_ctx: str) -> None:
    """Hold a `.to(ProcessPoolExecutorConfig)` region pipeline stopped before exit."""
    p = (
        PipelineBuilder()
        .add_source(range(1000), continuous=True)
        .to(ProcessPoolExecutorConfig(max_workers=2, mp_context=mp_ctx))
        .pipe(_double)
        .to(MAIN_PROCESS)
        .add_sink(4)
        .build(num_threads=2)
    )
    _HELD["dl"] = p
    with p.auto_stop():
        for i, _ in enumerate(p.get_iterator(timeout=60)):
            if i >= 10:
                break


def _child_exit_code(
    scenario: Callable[[str], None], mp_ctx: str, timeout: float = 120.0
) -> int | None:
    """Run ``scenario`` in a child process; return its exit code, or None if it hung."""
    ctx: Any = mp.get_context(mp_ctx)
    proc = ctx.Process(target=scenario, args=(mp_ctx,))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        return None
    return proc.exitcode


class RegionInterpreterExitTest(unittest.TestCase):
    """A held, unstopped `.to()` region pipeline exits cleanly at interpreter exit."""

    @unittest.skipUnless(
        "forkserver" in mp.get_all_start_methods(),
        "forkserver start method is unavailable on this platform (e.g. Windows)",
    )
    def test_to_region_forkserver(self) -> None:
        """`.to(ProcessPoolExecutorConfig)` region pipeline exits cleanly (forkserver)."""
        exit_code = _child_exit_code(_scenario_to_region, "forkserver")
        self.assertIsNotNone(
            exit_code,
            "Child process hung (did not exit within timeout); the region worker pool "
            "likely blocked interpreter shutdown.",
        )
        self.assertEqual(exit_code, 0)

    def test_to_region_spawn(self) -> None:
        """`.to(ProcessPoolExecutorConfig)` region pipeline exits cleanly (spawn)."""
        exit_code = _child_exit_code(_scenario_to_region, "spawn")
        self.assertIsNotNone(
            exit_code,
            "Child process hung (did not exit within timeout); the region worker pool "
            "likely blocked interpreter shutdown.",
        )
        self.assertEqual(exit_code, 0)

    @unittest.skipUnless(
        "forkserver" in mp.get_all_start_methods(),
        "forkserver start method is unavailable on this platform (e.g. Windows)",
    )
    def test_full_worker_input_forkserver(self) -> None:
        """A region with full worker input pipes lets its process exit (forkserver)."""
        exit_code = _child_exit_code(_scenario_full_worker_input, "forkserver")
        self.assertIsNotNone(
            exit_code,
            "Child process hung (did not exit within timeout); multiprocessing's exit "
            "handler likely joined a queue feeder blocked on a worker's full pipe.",
        )
        self.assertEqual(exit_code, 0)

    def test_full_worker_input_spawn(self) -> None:
        """A region with full worker input pipes lets its process exit (spawn)."""
        exit_code = _child_exit_code(_scenario_full_worker_input, "spawn")
        self.assertIsNotNone(
            exit_code,
            "Child process hung (did not exit within timeout); multiprocessing's exit "
            "handler likely joined a queue feeder blocked on a worker's full pipe.",
        )
        self.assertEqual(exit_code, 0)

    def test_stopped_region_spawn(self) -> None:
        """A region pipeline stopped before its process returns exits cleanly (spawn)."""
        exit_code = _child_exit_code(_scenario_stopped_region, "spawn")
        self.assertIsNotNone(
            exit_code, "Child process hung after stopping its region pipeline."
        )
        self.assertEqual(exit_code, 0)
