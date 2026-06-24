# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from spdl.pipeline import Pipeline, PipelineBuilder, run_pipeline_in_subprocess


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
