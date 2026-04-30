# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""V5.5 throughput regression test for the Diff 3a admission gate REPLACE.

Builds a small 4-stage pipeline, pushes ``NUM_ITEMS`` items through, and
compares wall time in two modes:

- **Baseline**: ``_install_semaphores_for_test=False`` — ``_pipe()`` uses
  the legacy static ``len(tasks) >= concurrency`` admission gate.
- **Treatment**: ``_install_semaphores_for_test=True`` — ``_pipe()`` uses
  the V5.6 REPLACE branch with a :py:class:`ResizableSemaphore`.

Regression > :py:data:`REGRESSION_THRESHOLD` on either p50 or p99 across
:py:data:`NUM_TRIALS` runs fails the diff (per V5.5 spec).
"""

import asyncio
import statistics
import time
import unittest

from spdl.pipeline import build_pipeline
from spdl.pipeline._common._types import StageInfo
from spdl.pipeline._components._queue import QueuePerfStats, StatsQueue
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig


def _stage_1(x: int) -> int:
    return x + 1


def _stage_2(x: int) -> int:
    return x * 2


def _stage_3(x: int) -> int:
    return x - 1


def _stage_4(x: int) -> int:
    return x // 2


class AdmissionGateThroughputRegressionTest(unittest.TestCase):
    """V5.5: admission gate REPLACE must not regress throughput by > 2%.

    A 4-stage sync pipeline is built with concurrency=4 per stage, fed
    ``NUM_ITEMS`` items, and run ``NUM_TRIALS`` times in each mode.

    Note (perf-reviewer WARN-1): this regression test covers sync stages
    only — extending coverage to async/sync_iter handlers is a follow-up
    after Diff 3a lands.
    """

    NUM_ITEMS: int = 10_000
    NUM_TRIALS: int = 5
    REGRESSION_THRESHOLD: float = 0.02  # 2%

    def _build_config(self) -> PipelineConfig[int]:
        return PipelineConfig(
            src=SourceConfig(iter(range(self.NUM_ITEMS))),
            pipes=[
                Pipe(_stage_1, concurrency=4),
                Pipe(_stage_2, concurrency=4),
                Pipe(_stage_3, concurrency=4),
                Pipe(_stage_4, concurrency=4),
            ],
            sink=SinkConfig(buffer_size=64),
        )

    def _run_once(self, *, install_semaphores: bool) -> float:
        config = self._build_config()
        pipeline = build_pipeline(
            config,
            num_threads=8,
            _install_semaphores_for_test=install_semaphores,
        )
        t0 = time.perf_counter()
        with pipeline.auto_stop():
            count = 0
            for _ in pipeline.get_iterator():
                count += 1
            self.assertEqual(count, self.NUM_ITEMS)
        return time.perf_counter() - t0

    def test_admission_gate_overhead_within_two_percent(self) -> None:
        baseline_times = [
            self._run_once(install_semaphores=False) for _ in range(self.NUM_TRIALS)
        ]
        flagged_times = [
            self._run_once(install_semaphores=True) for _ in range(self.NUM_TRIALS)
        ]

        baseline_p50 = statistics.median(baseline_times)
        flagged_p50 = statistics.median(flagged_times)
        # 5 trials → "p99" is effectively the max.
        baseline_p99 = max(baseline_times)
        flagged_p99 = max(flagged_times)

        p50_regression = (flagged_p50 - baseline_p50) / baseline_p50
        p99_regression = (flagged_p99 - baseline_p99) / baseline_p99

        self.assertLess(
            p50_regression,
            self.REGRESSION_THRESHOLD,
            f"p50 regression {p50_regression:.1%} exceeds "
            f"{self.REGRESSION_THRESHOLD:.0%} threshold "
            f"(baseline={baseline_p50:.3f}s, flagged={flagged_p50:.3f}s)",
        )
        self.assertLess(
            p99_regression,
            self.REGRESSION_THRESHOLD,
            f"p99 regression {p99_regression:.1%} exceeds "
            f"{self.REGRESSION_THRESHOLD:.0%} threshold "
            f"(baseline={baseline_p99:.3f}s, flagged={flagged_p99:.3f}s)",
        )


class AdmissionGateRegistryWiringTest(unittest.TestCase):
    """Smoke test: ``_install_semaphores_for_test=True`` populates the registry."""

    def test_registry_populated_for_each_pipe_stage(self) -> None:
        config = PipelineConfig(
            src=SourceConfig(iter(range(4))),
            pipes=[
                Pipe(_stage_1, concurrency=2, name="s1"),
                Pipe(_stage_2, concurrency=3, name="s2"),
            ],
            sink=SinkConfig(buffer_size=4),
        )
        pipeline = build_pipeline(
            config,
            num_threads=4,
            _install_semaphores_for_test=True,
        )
        impl = pipeline._impl
        # Both Pipe stages should be registered. The qualified name equals
        # ``StageInfo.stage_name`` for non-MultiPipe stages (V5.4).
        registry_keys = sorted(impl._semaphore_registry.keys())
        self.assertEqual(len(registry_keys), 2)
        self.assertIn("s1", registry_keys)
        self.assertIn("s2", registry_keys)
        # ``_dynamic_concurrency`` mirrors the build-time concurrency.
        self.assertEqual(impl._dynamic_concurrency["s1"], 2)
        self.assertEqual(impl._dynamic_concurrency["s2"], 3)
        # ``_stage_info_by_name`` mirrors the StageInfo for error messages.
        self.assertEqual(impl._stage_info_by_name["s1"].stage_name, "s1")
        self.assertEqual(impl._stage_info_by_name["s2"].stage_name, "s2")

    def test_registry_empty_when_flag_off(self) -> None:
        config = PipelineConfig(
            src=SourceConfig(iter(range(4))),
            pipes=[
                Pipe(_stage_1, concurrency=2, name="s1"),
            ],
            sink=SinkConfig(buffer_size=4),
        )
        pipeline = build_pipeline(config, num_threads=2)
        impl = pipeline._impl
        self.assertEqual(impl._semaphore_registry, {})
        self.assertEqual(impl._dynamic_concurrency, {})
        self.assertEqual(impl._stage_info_by_name, {})
        # Phase D: ``_output_queue_by_name`` mirrors the same key set,
        # so it is also empty when the flag is off.
        self.assertEqual(impl._output_queue_by_name, {})

    def test_output_queue_by_name_mirrors_semaphore_registry_keys(self) -> None:
        """Phase D: every entry in ``_semaphore_registry`` MUST have a
        corresponding entry in ``_output_queue_by_name`` under the same
        qualified name. The Diff 6 controller relies on this invariant
        when iterating ``_semaphore_registry`` and looking up the queue
        for each registered stage.
        """
        config = PipelineConfig(
            src=SourceConfig(iter(range(4))),
            pipes=[
                Pipe(_stage_1, concurrency=2, name="s1"),
                Pipe(_stage_2, concurrency=3, name="s2"),
                Pipe(_stage_3, concurrency=4, name="s3"),
            ],
            sink=SinkConfig(buffer_size=4),
        )
        pipeline = build_pipeline(
            config,
            num_threads=4,
            _install_semaphores_for_test=True,
        )
        impl = pipeline._impl
        sem_keys = sorted(impl._semaphore_registry.keys())
        queue_keys = sorted(impl._output_queue_by_name.keys())
        self.assertEqual(sem_keys, queue_keys)
        # Each value is an actual queue instance, not None.
        for qname in sem_keys:
            self.assertIsNotNone(impl._output_queue_by_name[qname])

    def test_output_queue_by_name_captures_actual_pipeline_queue(self) -> None:
        """Phase D: the queue stored in ``_output_queue_by_name[qname]``
        is the SAME instance that the pipeline build actually uses as
        the OUTPUT queue of stage ``qname``. We verify two structural
        properties:
        - Each stage's queue is an :py:class:`asyncio.Queue` subclass
          instance (i.e., a real pipeline queue, not a sentinel).
        - Each stage gets a UNIQUE queue instance (by id), so the
          controller can address each stage's queue independently.
        """
        import asyncio

        config = PipelineConfig(
            src=SourceConfig(iter(range(4))),
            pipes=[
                Pipe(_stage_1, concurrency=2, name="first"),
                Pipe(_stage_2, concurrency=2, name="middle"),
                Pipe(_stage_3, concurrency=2, name="last"),
            ],
            sink=SinkConfig(buffer_size=4),
        )
        pipeline = build_pipeline(
            config,
            num_threads=4,
            _install_semaphores_for_test=True,
        )
        impl = pipeline._impl
        queues = [
            impl._output_queue_by_name["first"],
            impl._output_queue_by_name["middle"],
            impl._output_queue_by_name["last"],
        ]
        # Real queues, not sentinels.
        for q in queues:
            self.assertIsInstance(q, asyncio.Queue)
        # Each stage has a unique queue instance — the controller can
        # address each stage's lap stats independently.
        queue_ids = {id(q) for q in queues}
        self.assertEqual(len(queue_ids), 3)


class AdmissionGateBranchSemanticsTest(unittest.TestCase):
    """V5.6: when a semaphore is registered, its value (not args.concurrency)
    governs the admission cap — the static gate is REPLACED, not augmented.

    We verify the wired path produces correct end-to-end output. Stronger
    in-flight-cap assertions live in the tests that import ``_pipe`` /
    ``_pipe_with_semaphore`` directly.
    """

    def test_pipeline_with_semaphores_produces_correct_output(self) -> None:
        config = PipelineConfig(
            src=SourceConfig(iter(range(20))),
            pipes=[
                Pipe(_stage_1, concurrency=2, name="add_one"),
                Pipe(_stage_2, concurrency=2, name="double"),
            ],
            sink=SinkConfig(buffer_size=8),
        )
        pipeline = build_pipeline(
            config, num_threads=4, _install_semaphores_for_test=True
        )
        with pipeline.auto_stop():
            results = sorted(pipeline.get_iterator())
        # f(x) = (x + 1) * 2 for x in range(20).
        expected = sorted((x + 1) * 2 for x in range(20))
        self.assertEqual(results, expected)

    def test_semaphore_value_caps_in_flight(self) -> None:
        """Sanity: in-flight count never exceeds the semaphore's value.

        Build a slow stage (sleeps) with concurrency=3, push more items
        than the cap, and observe the peak in-flight count from the work
        function itself.
        """
        in_flight: list[int] = [0]
        peak: list[int] = [0]
        # The slow op runs on worker threads; we need atomic
        # increments/decrements to observe the in-flight count
        # accurately. This is test-instrumentation only; pipeline code
        # itself uses no locks.
        import threading

        lock: threading.Lock = threading.Lock()

        def slow_op(x: int) -> int:
            with lock:
                in_flight[0] += 1
                if in_flight[0] > peak[0]:
                    peak[0] = in_flight[0]
            time.sleep(0.005)
            with lock:
                in_flight[0] -= 1
            return x

        config = PipelineConfig(
            src=SourceConfig(iter(range(50))),
            pipes=[
                Pipe(slow_op, concurrency=3, name="slow"),
            ],
            sink=SinkConfig(buffer_size=8),
        )
        pipeline = build_pipeline(
            config, num_threads=8, _install_semaphores_for_test=True
        )
        with pipeline.auto_stop():
            count = 0
            for _ in pipeline.get_iterator():
                count += 1
        self.assertEqual(count, 50)
        # The semaphore (set to args.concurrency=3) caps in-flight at 3.
        self.assertLessEqual(peak[0], 3)
        # Sanity: we did get parallelism (peak should reach 2 or 3).
        self.assertGreaterEqual(peak[0], 2)


class StatsQueueLastLapStatsCacheTest(unittest.TestCase):
    """Phase E: ``StatsQueue._log_interval_stats()`` caches the freshly
    computed :py:class:`QueuePerfStats` on ``self._last_lap_stats`` so
    non-callback readers (e.g., the LCA
    ``DomeVideoConcurrencyController``) can observe the latest interval
    without calling the destructive ``_get_lap_stats()``.

    These tests exercise the cache mechanism in isolation — they do NOT
    drive a pipeline. The end-to-end check (controller reading non-None
    ``queue_stats`` from a real pipeline) lives in
    ``test_concurrency_controller.py``.
    """

    def _make_queue(self) -> StatsQueue:
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="test")
        return StatsQueue(info, buffer_size=4)

    def test_last_lap_stats_starts_as_none(self) -> None:
        """Before the first interval fires, the cache MUST be ``None``
        (the controller's ``getattr(..., None)`` default would also
        return ``None``, but we want a real attribute so static
        analysers can see the type).
        """
        queue = self._make_queue()
        self.assertIsNone(queue._last_lap_stats)

    def test_log_interval_stats_populates_cache(self) -> None:
        """Calling ``_log_interval_stats()`` once writes the freshly
        computed ``QueuePerfStats`` to ``self._last_lap_stats``.
        """
        queue: StatsQueue = self._make_queue()
        # ``_get_lap_stats()`` reads ``self._lap_t0`` against
        # ``time.monotonic()``; seed it so ``elapsed`` is positive.
        queue._lap_t0 = time.monotonic() - 1.0

        async def _run() -> None:
            await queue._log_interval_stats()

        asyncio.run(_run())

        self.assertIsNotNone(queue._last_lap_stats)
        self.assertIsInstance(queue._last_lap_stats, QueuePerfStats)

    def test_log_interval_stats_overwrites_cache(self) -> None:
        """A second ``_log_interval_stats()`` call replaces the cached
        stats with the latest interval — verifies the cache is a
        single-slot snapshot, not an accumulator.
        """
        queue: StatsQueue = self._make_queue()
        queue._lap_t0 = time.monotonic() - 1.0

        async def _run_twice() -> tuple[QueuePerfStats, QueuePerfStats]:
            await queue._log_interval_stats()
            first = queue._last_lap_stats
            assert first is not None
            # Force a measurable elapsed delta on the second lap.
            queue._lap_t0 = time.monotonic() - 0.5
            await queue._log_interval_stats()
            second = queue._last_lap_stats
            assert second is not None
            return first, second

        first, second = asyncio.run(_run_twice())
        # Both cached values are real ``QueuePerfStats`` instances; the
        # second call replaced the first (different object identity).
        self.assertIsNot(first, second)
