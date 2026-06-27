# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from spdl.pipeline import PipelineBuilder


def _double(i: int) -> int:
    return i * 2


class TestProcessPoolOwnership(unittest.TestCase):
    """The pipeline recreates and owns a process-pool executor passed to a stage."""

    def test_process_pool_recreated_and_owned(self) -> None:
        """A ProcessPoolExecutor becomes an eagerly-spawned, pipeline-owned worker pool."""
        executor = ProcessPoolExecutor(max_workers=2)
        try:
            pipeline = (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(_double, executor=executor)
                .add_sink(100)
                .build(num_threads=1, mp_context="spawn")
            )
            # The user's executor is used only as a spec — its own workers never spawn.
            self.assertEqual(len(executor._processes), 0)
            # The pipeline owns an equivalent worker pool, spawned eagerly at build time.
            self.assertEqual(len(pipeline._impl._pools), 1)

            with pipeline.auto_stop():
                results = sorted(pipeline.get_iterator())
            self.assertEqual(results, [i * 2 for i in range(10)])

            # The user's executor stayed a pure spec: even after the pipeline ran to
            # completion, it never spawned a worker or had work submitted to it.
            self.assertEqual(len(executor._processes), 0)
            self.assertEqual(executor._queue_count, 0)

            # Stopping reaps the owned pool exactly once.
            self.assertEqual(len(pipeline._impl._pools), 0)
        finally:
            executor.shutdown()

    def test_used_process_pool_warns(self) -> None:
        """Passing an already-used ProcessPoolExecutor warns at build time."""
        executor = ProcessPoolExecutor(max_workers=2)
        try:
            # Submitting work spawns workers, marking the executor as used.
            executor.submit(_double, 1).result()

            builder = (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(_double, executor=executor)
                .add_sink(100)
            )
            with self.assertWarns(RuntimeWarning):
                pipeline = builder.build(num_threads=1, mp_context="spawn")
            # Clean up the owned pool that the build spawned.
            pipeline.stop()
        finally:
            executor.shutdown()

    def test_shared_process_pool_owned_once(self) -> None:
        """A process pool attached to multiple stages maps to a single owned pool."""
        executor = ProcessPoolExecutor(max_workers=2)
        try:
            pipeline = (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(_double, executor=executor)
                .pipe(_double, executor=executor)
                .add_sink(100)
                .build(num_threads=1, mp_context="spawn")
            )
            self.assertEqual(len(pipeline._impl._pools), 1)

            with pipeline.auto_stop():
                results = sorted(pipeline.get_iterator())
            self.assertEqual(results, [i * 4 for i in range(10)])
        finally:
            executor.shutdown()


class TestThreadPoolUnchanged(unittest.TestCase):
    """Thread-pool executors are not recreated or owned by this change."""

    def test_thread_pool_not_owned(self) -> None:
        """A ThreadPoolExecutor is used directly and not added to the owned pools."""
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            pipeline = (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(_double, executor=executor)
                .add_sink(100)
                .build(num_threads=1)
            )
            self.assertEqual(len(pipeline._impl._pools), 0)

            with pipeline.auto_stop():
                results = sorted(pipeline.get_iterator())
            self.assertEqual(results, [i * 2 for i in range(10)])
        finally:
            executor.shutdown()


class TestContinuousSourceOwnership(unittest.TestCase):
    """Pipeline ownership is what cleans up a process pool in continuous mode.

    With a continuous source the pipeline is started once and reused across epochs,
    so there is no ``auto_stop`` block scoping its lifetime. Ownership is what makes
    this safe: because the pipeline (not the caller) owns the recreated worker pool,
    stopping the pipeline reaps the workers even though the caller never wrapped
    iteration in ``auto_stop``.
    """

    def test_continuous_source_without_auto_stop(self) -> None:
        """A continuous-source process-pool pipeline reaps its owned pool on stop, no auto_stop."""
        executor = ProcessPoolExecutor(max_workers=2)
        try:
            pipeline = (
                PipelineBuilder()
                .add_source(range(5), continuous=True)
                .pipe(_double, executor=executor)
                .add_sink(100)
                .build(num_threads=1, mp_context="spawn")
            )
            # The pipeline owns an eagerly-spawned pool; the user executor is untouched.
            self.assertEqual(len(pipeline._impl._pools), 1)
            self.assertEqual(len(executor._processes), 0)

            # Iterate several epochs without auto_stop and without an explicit start: each
            # ``for`` pass consumes one epoch and the background thread auto-starts on the first.
            for _ in range(3):
                epoch = sorted(item for item in pipeline)
                self.assertEqual(epoch, [i * 2 for i in range(5)])

            # The owned pool's workers ran the whole time; the user executor never did.
            self.assertEqual(len(executor._processes), 0)
            self.assertEqual(executor._queue_count, 0)

            # Stopping the pipeline (the caller never opened an ``auto_stop`` block) reaps the
            # owned pool exactly once: ``_shutdown_pools`` clears the owned-pool list. This is a
            # deterministic check of the ownership contract, independent of GC/finalizer timing.
            pipeline.stop()
            self.assertEqual(len(pipeline._impl._pools), 0)
        finally:
            executor.shutdown()
