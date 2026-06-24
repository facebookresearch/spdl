# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any

from spdl.pipeline._fuse import _find_fusable_runs, _FusableRun
from spdl.pipeline.defs import Aggregate, Disaggregate, Pipe


def op(x: int) -> int:
    return x


async def aop(x: int) -> int:
    return x


class _FakeProcessPool(Executor):
    """Stand-in that ``_is_process_pool`` recognizes, without spawning processes."""

    _pool_executor_class: type[Executor] = ProcessPoolExecutor

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _FakeThreadPool(Executor):
    """Stand-in for a non-isolating (thread) executor — not a fusion target."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _pool_pipe(executor: Executor) -> Any:
    return Pipe(op, executor=executor, concurrency=2)


class FindFusableRunsTest(unittest.TestCase):
    def test_empty_pipes(self) -> None:
        """No pipes yields no fusable runs."""
        self.assertEqual(_find_fusable_runs([]), [])

    def test_two_same_executor_pipes_fuse(self) -> None:
        """Two adjacent pipes on the same pool executor fuse into one run."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), _pool_pipe(ex)])
        self.assertEqual(runs, [_FusableRun(0, 2, ex)])

    def test_aggregate_between_pool_pipes_breaks_run(self) -> None:
        """An aggregate between two pool pipes is not absorbed; the lone pipes do not fuse."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), Aggregate(4), _pool_pipe(ex)])
        self.assertEqual(runs, [])

    def test_disaggregate_between_pool_pipes_breaks_run(self) -> None:
        """A disaggregate between two pool pipes is not absorbed; the lone pipes do not fuse."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), Disaggregate(), _pool_pipe(ex)])
        self.assertEqual(runs, [])

    def test_aggregate_between_two_runs_fuses_each_side(self) -> None:
        """An aggregate splits the pipes; each adjacent same-executor pair fuses on its own."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs(
            [
                _pool_pipe(ex),
                _pool_pipe(ex),
                Aggregate(4),
                _pool_pipe(ex),
                _pool_pipe(ex),
            ]
        )
        self.assertEqual(runs, [_FusableRun(0, 2, ex), _FusableRun(3, 5, ex)])

    def test_trailing_aggregate_not_fused(self) -> None:
        """A trailing aggregate is left in the main process; only the pool pipes fuse."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), _pool_pipe(ex), Aggregate(4)])
        self.assertEqual(runs, [_FusableRun(0, 2, ex)])

    def test_single_pool_pipe_with_trailing_aggregate_not_fused(self) -> None:
        """A lone pool pipe plus a trailing aggregate has no same-executor neighbour: no run."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), Aggregate(4)])
        self.assertEqual(runs, [])

    def test_trailing_disaggregate_not_fused(self) -> None:
        """A trailing disaggregate is left in the main process; only the pool pipes fuse."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), _pool_pipe(ex), Disaggregate()])
        self.assertEqual(runs, [_FusableRun(0, 2, ex)])

    def test_different_executors_do_not_fuse(self) -> None:
        """Adjacent pool pipes on different executor instances do not fuse."""
        runs = _find_fusable_runs(
            [_pool_pipe(_FakeProcessPool()), _pool_pipe(_FakeProcessPool())]
        )
        self.assertEqual(runs, [])

    def test_thread_pool_breaks_run(self) -> None:
        """A thread-pool stage between two pool pipes breaks the run (no fusion)."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs(
            [_pool_pipe(ex), _pool_pipe(_FakeThreadPool()), _pool_pipe(ex)]
        )
        self.assertEqual(runs, [])

    def test_default_executor_breaks_run(self) -> None:
        """A stage with no executor (default thread pool) breaks the run."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), Pipe(op), _pool_pipe(ex)])
        self.assertEqual(runs, [])

    def test_async_op_breaks_run(self) -> None:
        """An async stage breaks the run."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), Pipe(aop), _pool_pipe(ex)])
        self.assertEqual(runs, [])

    def test_input_ordered_pool_pipe_not_fused(self) -> None:
        """An input-ordered pool pipe is not fusable and breaks the run."""
        ex = _FakeProcessPool()
        ordered = Pipe(op, executor=ex, output_order="input")
        runs = _find_fusable_runs([_pool_pipe(ex), ordered, _pool_pipe(ex)])
        self.assertEqual(runs, [])

    def test_lone_pool_pipe_untouched(self) -> None:
        """A single pool pipe with nothing to combine is left unchanged."""
        self.assertEqual(_find_fusable_runs([_pool_pipe(_FakeProcessPool())]), [])

    def test_leading_micro_stage_stays_in_main(self) -> None:
        """A micro-stage before the first pool pipe stays in the main process."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([Aggregate(4), _pool_pipe(ex), _pool_pipe(ex)])
        self.assertEqual(runs, [_FusableRun(1, 3, ex)])

    def test_multiple_runs(self) -> None:
        """Several disjoint fusable runs are all detected."""
        ex1, ex2 = _FakeProcessPool(), _FakeProcessPool()
        pipes = [
            _pool_pipe(ex1),
            _pool_pipe(ex1),
            Pipe(op),  # breaker
            _pool_pipe(ex2),
            _pool_pipe(ex2),
        ]
        runs = _find_fusable_runs(pipes)
        self.assertEqual(runs, [_FusableRun(0, 2, ex1), _FusableRun(3, 5, ex2)])

    def test_same_executor_separated_by_breaker_not_fused(self) -> None:
        """The same executor reused across a breaker yields no paying run."""
        ex = _FakeProcessPool()
        runs = _find_fusable_runs([_pool_pipe(ex), Pipe(op), _pool_pipe(ex)])
        self.assertEqual(runs, [])
