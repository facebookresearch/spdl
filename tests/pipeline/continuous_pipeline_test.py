# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import functools
import os
import sys
import threading
import time
import unittest
import warnings
import weakref
from collections.abc import Iterator

from spdl.pipeline import (
    build_pipeline,
    PipelineBuilder,
    PipelineFailure,
    run_pipeline_in_subinterpreter,
    run_pipeline_in_subprocess,
)
from spdl.pipeline.defs import (
    Aggregate,
    Merge,
    PathVariants,
    Pipe,
    PipelineConfig,
    SinkConfig,
)


def _ignore_fork_warning(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"This process \(pid=\d+\) is multi-threaded, use of "
                    r"fork\(\) may lead to deadlocks in the child"
                ),
                category=DeprecationWarning,
            )
            return fn(*args, **kwargs)

    return wrapper


class SourceIterable:
    """Reusable iterable that yields range(n) on each iteration."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> Iterator[int]:
        yield from range(self.n)


class EpochTaggedSource:
    """Yields a disjoint block of values per epoch (epoch K -> K*100 + range(n)).

    Successive epochs produce non-overlapping value ranges so a test can detect
    an item leaking across the epoch boundary by value alone.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        base = self._epoch * 100
        self._epoch += 1
        yield from range(base, base + self.n)


class TestContinuousPipelineBasic(unittest.TestCase):
    def test_continuous_multi_epoch(self) -> None:
        """Pipeline with continuous=True can be iterated multiple times."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [0, 1, 2, 3, 4], f"epoch {epoch}")

    def test_continuous_single_epoch(self) -> None:
        """Continuous pipeline works for a single epoch."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(3), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            result = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(result, [0, 1, 2])

    def test_continuous_empty_epoch(self) -> None:
        """Continuous pipeline handles empty iterations."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(0), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [], f"epoch {epoch}")

    def test_continuous_single_item_epoch(self) -> None:
        """Continuous pipeline works with single-item epochs."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(1), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [0], f"epoch {epoch}")


class TestContinuousPipelinePipe(unittest.TestCase):
    def test_continuous_pipe_concurrent(self) -> None:
        """Pipe with concurrency > 1 handles epoch boundaries correctly."""

        def double(x: int) -> int:
            return x * 2

        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .pipe(double, concurrency=4)
            .add_sink(buffer_size=3)
            .build(num_threads=4)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = sorted(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [0, 2, 4, 6, 8], f"epoch {epoch}")

    def test_continuous_pipe_chain(self) -> None:
        """EPOCH_END propagates through multiple pipe stages."""

        def add_one(x: int) -> int:
            return x + 1

        def double(x: int) -> int:
            return x * 2

        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(3), continuous=True)
            .pipe(add_one, concurrency=1)
            .pipe(double, concurrency=1)
            .add_sink(buffer_size=3)
            .build(num_threads=2)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [2, 4, 6], f"epoch {epoch}")


class TestContinuousPipelineAggregate(unittest.TestCase):
    def test_continuous_aggregate_exact_batch(self) -> None:
        """Aggregate with exact batch size across epochs."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(6), continuous=True)
            .aggregate(3)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [[0, 1, 2], [3, 4, 5]], f"epoch {epoch}")

    def test_continuous_aggregate_partial_batch_flushed(self) -> None:
        """Partial batch at epoch end is flushed by the default aggregator."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .aggregate(3)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                # 5 items, batch_size=3: full batch [0,1,2] + partial batch [3,4]
                self.assertEqual(result, [[0, 1, 2], [3, 4]], f"epoch {epoch}")

    def test_continuous_aggregate_drop_last(self) -> None:
        """With drop_last=True, partial batch at epoch end is discarded."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .aggregate(3, drop_last=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                # 5 items, batch_size=3, drop_last: only [0,1,2], partial [3,4] dropped
                self.assertEqual(result, [[0, 1, 2]], f"epoch {epoch}")


class TestContinuousPipelineShutdown(unittest.TestCase):
    def test_continuous_auto_stop_mid_epoch(self) -> None:
        """auto_stop() exits cleanly even if mid-epoch."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(100), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            it = pipeline.get_iterator(timeout=30)
            # Consume only a few items
            self.assertEqual(next(it), 0)
            self.assertEqual(next(it), 1)
            # auto_stop exits here — must not hang

    def test_continuous_stop_between_epochs(self) -> None:
        """stop() between epochs works cleanly."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(3), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            result = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(result, [0, 1, 2])
            # auto_stop exits here after one epoch — must not hang

    def test_continuous_get_iterator_reuse(self) -> None:
        """get_iterator() can be called multiple times within auto_stop()."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(3), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            r1 = list(pipeline.get_iterator(timeout=30))
            r2 = list(pipeline.get_iterator(timeout=30))
            r3 = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(r1, [0, 1, 2])
            self.assertEqual(r2, [0, 1, 2])
            self.assertEqual(r3, [0, 1, 2])

    def test_same_iterator_is_single_use(self) -> None:
        """An iterator covers one epoch: reuse yields nothing, in either mode."""
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                pipeline = (
                    PipelineBuilder()
                    .add_source(SourceIterable(3), continuous=continuous)
                    .add_sink(buffer_size=3)
                    .build(num_threads=1)
                )

                with pipeline.auto_stop():
                    it = pipeline.get_iterator(timeout=30)
                    self.assertEqual(list(it), [0, 1, 2])
                    # Reusing the same iterator does not start a new epoch.
                    self.assertEqual(list(it), [])

    def test_continuous_stop_with_pipe_stage(self) -> None:
        """Continuous pipeline with pipe stage can be stopped after epochs."""

        def double(x: int) -> int:
            return x * 2

        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .pipe(double, concurrency=2)
            .add_sink(buffer_size=3)
            .build(num_threads=2)
        )

        with pipeline.auto_stop():
            r1 = sorted(pipeline.get_iterator(timeout=30))
            self.assertEqual(r1, [0, 2, 4, 6, 8])
            # auto_stop exits — pipeline has items buffered for next epoch
            # stop() must drain and shut down cleanly

    @_ignore_fork_warning
    def test_continuous_stop_with_subprocess_source(self) -> None:
        """Continuous pipeline reading from subprocess can be stopped."""
        backend = (
            PipelineBuilder().add_source(SourceIterable(5)).add_sink(buffer_size=3)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=1,
            timeout=10,
        )

        pipeline = (
            PipelineBuilder()
            .add_source(source, continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            r1 = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(r1, [0, 1, 2, 3, 4])
            # auto_stop exits — must not hang on subprocess IPC

    @_ignore_fork_warning
    def test_continuous_pipeline_in_subprocess_multi_epoch(self) -> None:
        """A continuous pipeline running inside a subprocess supports
        multi-epoch iteration without recreating the subprocess."""
        backend = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .add_sink(buffer_size=3)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=1,
            timeout=10,
        )

        # Iterate 3 epochs from the parent — subprocess is reused
        for epoch in range(3):
            result = list(source)
            self.assertEqual(result, [0, 1, 2, 3, 4], f"epoch {epoch}")

    @_ignore_fork_warning
    def test_continuous_pipeline_in_subprocess_stop_mid_epoch(self) -> None:
        """Subprocess with continuous pipeline can be abandoned mid-epoch."""
        backend = (
            PipelineBuilder()
            .add_source(SourceIterable(100), continuous=True)
            .add_sink(buffer_size=3)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=1,
            timeout=10,
        )

        it = iter(source)
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 1)
        # Abandon — must not hang
        del it
        del source

    @_ignore_fork_warning
    def test_continuous_frontend_backend_multi_epoch(self) -> None:
        """Frontend continuous pipeline on top of subprocess backend
        supports multi-epoch iteration."""
        backend = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .add_sink(buffer_size=3)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=1,
            timeout=10,
        )

        pipeline = (
            PipelineBuilder()
            .add_source(source, continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [0, 1, 2, 3, 4], f"epoch {epoch}")

    @_ignore_fork_warning
    def test_continuous_frontend_backend_stop_mid_epoch(self) -> None:
        """Frontend continuous pipeline on top of subprocess backend
        can be stopped mid-epoch without hanging."""
        backend = (
            PipelineBuilder()
            .add_source(SourceIterable(100), continuous=True)
            .add_sink(buffer_size=3)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=1,
            timeout=10,
        )

        pipeline = (
            PipelineBuilder()
            .add_source(source, continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with pipeline.auto_stop():
            it = pipeline.get_iterator(timeout=30)
            self.assertEqual(next(it), 0)
            self.assertEqual(next(it), 1)
            # auto_stop exits mid-epoch — must not hang

    @_ignore_fork_warning
    def test_continuous_frontend_backend_finalizer_shutdown(self) -> None:
        """Frontend+backend pipeline cleaned up via weakref.finalize.

        Simulates the _SPDLDataLoader pattern: pipeline is started, iterated,
        then the wrapper goes out of scope. The finalizer calls stop(timeout=10).
        Must not hang.
        """

        backend = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .add_sink(buffer_size=3)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=1,
            timeout=10,
        )

        pipeline = (
            PipelineBuilder()
            .add_source(source, continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        pipeline.start()
        finalizer = weakref.finalize(pipeline, lambda p: p.stop(timeout=30), pipeline)

        # Iterate 2 epochs
        for epoch in range(2):
            result = list(pipeline.get_iterator(timeout=30))
            self.assertEqual(result, [0, 1, 2, 3, 4], f"epoch {epoch}")

        t0 = time.monotonic()
        # Trigger finalizer (simulates going out of scope)
        finalizer()
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 15, f"finalizer took {elapsed:.1f}s — likely hung")


class TestContinuousPipelinePathVariants(unittest.TestCase):
    def test_continuous_path_variants_multi_epoch(self) -> None:
        """PathVariants epoch boundary propagates across epochs (no deadlock)."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(6), continuous=True)
            .path_variants(
                router=lambda x: x % 2,
                paths=[
                    [Pipe(lambda x: x * 2)],  # path 0: even items doubled
                    [Pipe(lambda x: x + 100)],  # path 1: odd items + 100
                ],
            )
            .add_sink(buffer_size=10)
            .build(num_threads=4)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = sorted(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [0, 4, 8, 101, 103, 105], f"epoch {epoch}")

    def test_continuous_path_variants_empty_epoch(self) -> None:
        """PathVariants handles empty epochs from a continuous source."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(0), continuous=True)
            .path_variants(
                router=lambda x: x % 2,
                paths=[
                    [Pipe(lambda x: x * 2)],
                    [Pipe(lambda x: x + 100)],
                ],
            )
            .add_sink(buffer_size=10)
            .build(num_threads=2)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                self.assertEqual(result, [], f"epoch {epoch}")

    def test_continuous_nested_path_variants_multi_epoch(self) -> None:
        """Epoch boundary propagates through a PathVariants nested in a path."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(8), continuous=True)
            .path_variants(
                router=lambda x: x % 2,
                paths=[
                    # path 0 (evens): nested PathVariants splitting on x % 4
                    [
                        PathVariants(
                            router=lambda x: 0 if x % 4 == 0 else 1,
                            paths=[
                                [Pipe(lambda x: x)],  # multiples of 4 unchanged
                                [Pipe(lambda x: x * 10)],  # other evens * 10
                            ],
                        )
                    ],
                    # path 1 (odds): + 100
                    [Pipe(lambda x: x + 100)],
                ],
            )
            .add_sink(buffer_size=16)
            .build(num_threads=4)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = sorted(pipeline.get_iterator(timeout=30))
                # evens 0,4 -> 0,4; evens 2,6 -> 20,60; odds 1,3,5,7 -> 101..107
                self.assertEqual(
                    result, [0, 4, 20, 60, 101, 103, 105, 107], f"epoch {epoch}"
                )

    def test_continuous_path_variants_aggregate_partial_flush(self) -> None:
        """Partial aggregate batch inside a path is flushed at each epoch end."""
        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(6), continuous=True)
            .path_variants(
                router=lambda x: x % 2,
                paths=[
                    [Aggregate(2)],  # path 0: batch evens (0,2,4) by 2
                    [Pipe(lambda x: x + 100)],  # path 1: odds + 100
                ],
            )
            .add_sink(buffer_size=10)
            .build(num_threads=2)
        )

        with pipeline.auto_stop():
            for epoch in range(3):
                result = list(pipeline.get_iterator(timeout=30))
                # evens -> [0, 2] full batch + [4] partial flushed at epoch end;
                # odds -> 101, 103, 105. Cross-path order is nondeterministic.
                self.assertCountEqual(
                    result, [[0, 2], [4], 101, 103, 105], f"epoch {epoch}"
                )

    def test_continuous_path_variants_no_epoch_crossing(self) -> None:
        """Slow/fast path skew must not leak items across the epoch boundary."""

        async def slow(x: int) -> int:
            await asyncio.sleep(0.05)
            return x

        async def fast(x: int) -> int:
            return x

        pipeline = (
            PipelineBuilder()
            .add_source(EpochTaggedSource(4), continuous=True)
            .path_variants(
                router=lambda x: x % 2,
                paths=[
                    [Pipe(slow, concurrency=2)],  # even items: slow path
                    [Pipe(fast, concurrency=2)],  # odd items: fast path
                ],
            )
            .add_sink(buffer_size=16)
            .build(num_threads=4)
        )

        with pipeline.auto_stop():
            for epoch in range(4):
                base = epoch * 100
                result = sorted(pipeline.get_iterator(timeout=30))
                # Each epoch must contain exactly its own disjoint block. A
                # next-epoch fast-path item leaking in would show as a value
                # >= base + 100; the fan-in barrier parks the fast path at the
                # EOE until the slow path catches up, preventing that.
                self.assertEqual(
                    result, [base, base + 1, base + 2, base + 3], f"epoch {epoch}"
                )


class CustomError(ValueError):
    pass


class TestContinuousPipelineErrors(unittest.TestCase):
    def test_continuous_source_failure(self) -> None:
        """Source raising mid-epoch propagates error."""

        class FailingSource:
            def __iter__(self) -> Iterator[int]:
                yield 0
                raise CustomError("source failed")

        pipeline = (
            PipelineBuilder()
            .add_source(FailingSource(), continuous=True)
            .add_sink(buffer_size=3)
            .build(num_threads=1)
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))

    def test_continuous_pipe_failure(self) -> None:
        """Pipe function raising propagates error."""

        def failing_fn(x: int) -> int:
            if x == 2:
                raise CustomError("pipe failed")
            return x

        pipeline = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .pipe(failing_fn, concurrency=1)
            .add_sink(buffer_size=3)
            .build(num_threads=1, max_failures=0)
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))


class TestContinuousPipelineValidation(unittest.TestCase):
    def test_mixed_continuous_mode_rejected(self) -> None:
        """Mixing continuous and non-continuous sources raises ValueError."""
        plc_continuous = (
            PipelineBuilder()
            .add_source(SourceIterable(3), continuous=True)
            .add_sink()
            .get_config()
        )
        plc_normal = (
            PipelineBuilder().add_source(SourceIterable(3)).add_sink().get_config()
        )

        merged_config = PipelineConfig(
            src=Merge([plc_continuous, plc_normal]),
            pipes=[],
            sink=SinkConfig(buffer_size=3),
        )

        with self.assertRaisesRegex(ValueError, "Mixed continuous mode"):
            build_pipeline(merged_config, num_threads=1)


class _RecordIDs:
    """Pipe function that records the thread ID and process ID."""

    def __init__(self) -> None:
        self.thread_ids: list[int] = []
        self.process_ids: list[int] = []

    def __call__(self, x: int) -> int:
        self.thread_ids.append(threading.get_ident())
        self.process_ids.append(os.getpid())
        return x


class TestContinuousSubprocessPipelineReuse(unittest.TestCase):
    @_ignore_fork_warning
    def test_subprocess_pipeline_reused_across_epochs(self) -> None:
        """Thread and process IDs stay the same across epochs, proving
        the pipeline is reused rather than rebuilt."""
        recorder = _RecordIDs()

        backend = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .pipe(recorder, concurrency=1)
            .add_sink(buffer_size=3)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=1,
            timeout=10,
        )

        all_thread_ids: list[set[int]] = []
        all_process_ids: list[set[int]] = []

        for epoch in range(3):
            recorder.thread_ids.clear()
            recorder.process_ids.clear()
            result = list(source)
            self.assertEqual(sorted(result), [0, 1, 2, 3, 4], f"epoch {epoch}")
            all_thread_ids.append(set(recorder.thread_ids))
            all_process_ids.append(set(recorder.process_ids))

        # All epochs should use the same thread(s) — pipeline was reused
        self.assertEqual(
            all_thread_ids[0],
            all_thread_ids[1],
            "Thread IDs changed between epoch 0 and 1 — pipeline was rebuilt",
        )
        self.assertEqual(
            all_thread_ids[1],
            all_thread_ids[2],
            "Thread IDs changed between epoch 1 and 2 — pipeline was rebuilt",
        )

        # All epochs should run in the same subprocess
        self.assertEqual(
            all_process_ids[0],
            all_process_ids[1],
            "Process IDs changed between epoch 0 and 1",
        )
        self.assertEqual(
            all_process_ids[1],
            all_process_ids[2],
            "Process IDs changed between epoch 1 and 2",
        )

    @_ignore_fork_warning
    def test_continuous_subprocess_concurrent_aggregate_crisp_boundary(self) -> None:
        """A continuous subprocess pipeline with a concurrent pipe stage and
        aggregation keeps crisp epoch boundaries across epochs.

        Concurrency can reorder items and aggregation buffers partial batches,
        so this guards against epochs bleeding into one another (e.g. a partial
        batch from one epoch being emitted in the next) when the pipeline is
        reused across epochs in the subprocess.
        """

        def double(x: int) -> int:
            return x * 2

        backend = (
            PipelineBuilder()
            .add_source(SourceIterable(10), continuous=True)
            .pipe(double, concurrency=4)
            .aggregate(3, drop_last=False)
            .add_sink(buffer_size=2)
        )
        source = run_pipeline_in_subprocess(
            backend.get_config(),
            num_threads=4,
            timeout=10,
        )

        # 10 items, batch_size=3, drop_last=False -> batches of [3, 3, 3, 1].
        expected_items = sorted(x * 2 for x in range(10))
        for epoch in range(3):
            batches = list(source)
            self.assertEqual(len(batches), 4, f"epoch {epoch}: batch count")
            flat = sorted(v for batch in batches for v in batch)
            self.assertEqual(flat, expected_items, f"epoch {epoch}: items")


@unittest.skipIf(
    sys.version_info < (3, 14),
    "Subinterpreters require Python 3.14+",
)
class TestContinuousSubinterpreterPipelineReuse(unittest.TestCase):
    @_ignore_fork_warning
    def test_subinterpreter_pipeline_reused_across_epochs(self) -> None:
        """Thread and process IDs stay the same across epochs in subinterpreter."""
        recorder = _RecordIDs()

        config = (
            PipelineBuilder()
            .add_source(SourceIterable(5), continuous=True)
            .pipe(recorder, concurrency=1)
            .add_sink(buffer_size=3)
            .get_config()
        )
        source = run_pipeline_in_subinterpreter(
            config,
            num_threads=1,
            timeout=10,
        )

        all_thread_ids: list[set[int]] = []
        all_process_ids: list[set[int]] = []

        for epoch in range(3):
            recorder.thread_ids.clear()
            recorder.process_ids.clear()
            result = list(source)
            self.assertEqual(sorted(result), [0, 1, 2, 3, 4], f"epoch {epoch}")
            all_thread_ids.append(set(recorder.thread_ids))
            all_process_ids.append(set(recorder.process_ids))

        # All epochs should use the same thread(s) — pipeline was reused
        self.assertEqual(
            all_thread_ids[0],
            all_thread_ids[1],
            "Thread IDs changed between epoch 0 and 1 — pipeline was rebuilt",
        )
        self.assertEqual(
            all_thread_ids[1],
            all_thread_ids[2],
            "Thread IDs changed between epoch 1 and 2 — pipeline was rebuilt",
        )

        # All epochs should run in the same process (subinterpreter shares process)
        self.assertEqual(
            all_process_ids[0],
            all_process_ids[1],
            "Process IDs changed between epoch 0 and 1",
        )
