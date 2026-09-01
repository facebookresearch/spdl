# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""End-to-end tests for ``buffer_size`` -- transfer buffering at a region boundary.

The bridge packs items into one transfer on the way into the region and unpacks them on the way
out, so the region's stages (and everything downstream) still see individual items. These tests
pin that transparency: for every scenario the observable output must not depend on
``buffer_size``.

The interesting cases are the ones where the region's output cardinality differs from its
input's -- a dropped item, a generator, an ``aggregate`` -- because nothing in the protocol may
assume the two directions chunk alike. ``continuous`` sources get their own coverage since a
chunk that straddled an epoch boundary would silently move data between epochs.

Bridge-level packing/flushing is covered directly (and deterministically) in
``subprocess_pipe_bridge_test.py``; this file only checks the observable end-to-end behavior.
"""

import unittest
from collections.abc import Iterator
from typing import Any

from parameterized import parameterized  # pyre-ignore[21]
from spdl.pipeline import Pipeline, PipelineBuilder
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

# (buffer_size, max_workers). Sizes are deliberately co-prime with the item counts
# below so the tail chunk is always partial. ``(1, N)`` is the unbuffered control and
# ``(16, 2)`` covers a transfer larger than a whole epoch. Kept small on purpose: every case
# spawns a worker pool, and the suite runs these in parallel.
_MATRIX: list[tuple[int, int]] = [(1, 1), (1, 3), (3, 3), (16, 2)]

# Generous: these cases are correctness checks, not latency checks, and they compete with the
# rest of the suite for cores. A real hang still fails via the test runner's own timeout.
_TIMEOUT: float = 180.0


def add_one(x: int) -> int:
    return x + 1


def times_two(x: int) -> int:
    return x * 2


def drop_odd(x: int) -> int | None:
    """1 -> 0 for odd inputs: SPDL absorbs a ``None`` result, so fewer items come out."""
    return None if x % 2 else x


def explode(x: int) -> Iterator[int]:
    """1 -> 3: a generator op makes the region emit more items than it received."""
    yield x
    yield x + 1000
    yield x + 2000


def boom_on_five(x: int) -> int:
    if x == 5:
        raise ValueError("boom")
    return x


def _run(pipeline: Pipeline[Any]) -> list[Any]:
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=_TIMEOUT))


def _build(
    source: Any,
    buffer_size: int,
    max_workers: int,
    *,
    continuous: bool = False,
) -> PipelineBuilder[Any, Any]:
    """Open a region over ``source``; the caller adds the region's stages and closes it."""
    return (
        PipelineBuilder()
        .add_source(source, continuous=continuous)
        .to(
            ProcessPoolExecutorConfig(max_workers=max_workers),
            buffer_size=buffer_size,
        )
    )


class RegionBufferSizeTest(unittest.TestCase):
    """A buffered region produces exactly what an unbuffered one would."""

    @parameterized.expand(_MATRIX)
    def test_identity_roundtrip(self, buffer_size: int, max_workers: int) -> None:
        """Every item makes it through, whatever the transfer size.

        The item count (23) is not a multiple of any size in the matrix, so the tail
        chunk is always partial -- the case where a missing flush would silently drop items.
        """
        n = 23
        pipeline = (
            _build(range(n), buffer_size, max_workers)
            .pipe(add_one)
            .pipe(times_two)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))

    @parameterized.expand(_MATRIX)
    def test_empty_source(self, buffer_size: int, max_workers: int) -> None:
        """An empty source completes and yields nothing, rather than hanging on a chunk
        that never fills."""
        pipeline = (
            _build([], buffer_size, max_workers)
            .pipe(add_one)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        self.assertEqual(_run(pipeline), [])

    @parameterized.expand(_MATRIX)
    def test_fewer_items_than_buffer_size(
        self, buffer_size: int, max_workers: int
    ) -> None:
        """A stream shorter than one chunk is still delivered (flushed at end of stream)."""
        pipeline = (
            _build(range(1), buffer_size, max_workers)
            .pipe(add_one)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        self.assertEqual(_run(pipeline), [1])

    @parameterized.expand(_MATRIX)
    def test_region_emitting_fewer_items(
        self, buffer_size: int, max_workers: int
    ) -> None:
        """1 -> 0 stages: the output chunking must not assume one result per input."""
        n = 23
        pipeline = (
            _build(range(n), buffer_size, max_workers)
            .pipe(drop_odd)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), [x for x in range(n) if x % 2 == 0])

    @parameterized.expand(_MATRIX)
    def test_region_emitting_more_items(
        self, buffer_size: int, max_workers: int
    ) -> None:
        """1 -> 3 stages: more results than inputs, so output chunks outnumber input chunks."""
        n = 11
        pipeline = (
            _build(range(n), buffer_size, max_workers)
            .pipe(explode)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        expected = sorted(x + k for x in range(n) for k in (0, 1000, 2000))
        self.assertEqual(sorted(_run(pipeline)), expected)

    @parameterized.expand(_MATRIX)
    def test_aggregate_inside_region(self, buffer_size: int, max_workers: int) -> None:
        """N -> 1 stages: batching inside the region is independent of the transfer size.

        This is the shape the feature exists for -- the batch size and the transfer size are
        set separately. Each worker aggregates only the items routed to it, so batches may be
        partial at end of stream; the union across batches is what must be preserved.
        """
        n = 23
        batch = 4
        pipeline = (
            _build(range(n), buffer_size, max_workers)
            .aggregate(batch)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        batches = _run(pipeline)
        self.assertTrue(all(len(b) <= batch for b in batches))
        self.assertCountEqual([x for b in batches for x in b], list(range(n)))

    @parameterized.expand(_MATRIX)
    def test_drop_last_inside_region(self, buffer_size: int, max_workers: int) -> None:
        """With drop_last each worker discards its own partial batch, but only whole batches.

        The point is that the loss is a property of per-worker aggregation, not of the transfer
        size: every emitted batch is full, and nothing outside the source appears.
        """
        n = 23
        batch = 4
        pipeline = (
            _build(range(n), buffer_size, max_workers)
            .aggregate(batch, drop_last=True)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        batches = _run(pipeline)
        self.assertTrue(all(len(b) == batch for b in batches))
        emitted = [x for b in batches for x in b]
        self.assertEqual(len(emitted), len(set(emitted)))
        self.assertTrue(set(emitted) <= set(range(n)))

    @parameterized.expand(_MATRIX)
    def test_op_failure_drops_only_that_item(
        self, buffer_size: int, max_workers: int
    ) -> None:
        """A failing item is dropped without taking the rest of its transfer with it.

        SPDL's default is to drop a failed item rather than fail the pipeline, and
        ``max_failures`` is not propagated into a region's nested pipeline, so the observable
        behavior is a missing item. What matters for buffering is that the *siblings* sharing
        the failed item's chunk still arrive. The iterator timeout turns a hang into a failure.
        """
        n = 23
        pipeline = (
            _build(range(n), buffer_size, max_workers)
            .pipe(boom_on_five)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), [x for x in range(n) if x != 5])


class ContinuousRegionBufferSizeTest(unittest.TestCase):
    """Buffering must not move items across an epoch boundary."""

    @parameterized.expand(_MATRIX)
    def test_each_epoch_exact(self, buffer_size: int, max_workers: int) -> None:
        """Every epoch delivers exactly its own items -- no tail carried into the next one."""
        n = 23
        ref = sorted((x + 1) * 2 for x in range(n))
        pipeline = (
            _build(range(n), buffer_size, max_workers, continuous=True)
            .pipe(add_one)
            .pipe(times_two)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        with pipeline.auto_stop():
            for epoch in range(3):
                with self.subTest(epoch=epoch):
                    got = sorted(pipeline.get_iterator(timeout=_TIMEOUT))
                    # Exact equality, not a subset: a chunk straddling the boundary would show
                    # up as a short epoch followed by an over-long one.
                    self.assertEqual(got, ref)

    @parameterized.expand(_MATRIX)
    def test_epoch_shorter_than_buffer(
        self, buffer_size: int, max_workers: int
    ) -> None:
        """An epoch with fewer items than one chunk must still be flushed at the boundary.

        Without the flush before the ``_EPOCH`` broadcast this deadlocks: the items sit in the
        feeder's partial chunk, the workers see an empty epoch, and the epoch's results never
        arrive.
        """
        n = 2
        ref = sorted(x + 1 for x in range(n))
        pipeline = (
            _build(range(n), buffer_size, max_workers, continuous=True)
            .pipe(add_one)
            .to(MAIN_PROCESS)
            .add_sink(n)
            .build(num_threads=4)
        )
        with pipeline.auto_stop():
            for epoch in range(3):
                with self.subTest(epoch=epoch):
                    self.assertEqual(
                        sorted(pipeline.get_iterator(timeout=_TIMEOUT)), ref
                    )


class BufferSizeEquivalenceTest(unittest.TestCase):
    """The observable result is identical across transfer sizes."""

    def test_same_output_for_every_buffer_size(self) -> None:
        """A region's output does not depend on how its transfers are chunked."""
        n = 23

        def _run_with(buffer_size: int) -> list[int]:
            pipeline = (
                _build(range(n), buffer_size, 3)
                .pipe(drop_odd)
                .pipe(explode)
                .to(MAIN_PROCESS)
                .add_sink()
                .build(num_threads=4)
            )
            return sorted(_run(pipeline))

        baseline = _run_with(1)
        self.assertTrue(baseline)  # guard against the comparison being vacuous
        for buffer_size in (2, 5, 32):
            with self.subTest(buffer_size=buffer_size):
                self.assertEqual(_run_with(buffer_size), baseline)
