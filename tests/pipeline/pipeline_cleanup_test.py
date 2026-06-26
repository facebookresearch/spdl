# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gc
import unittest
import warnings

from spdl.pipeline import PipelineBuilder


class TestPipelineCleanup(unittest.TestCase):
    """Test class for Pipeline cleanup functionality."""

    def test_cleanup_called_on_garbage_collection(self) -> None:
        """Test that the pipeline background thread is automatically stopped
        on garbage collection without warnings."""

        # Setup: Create a pipeline and start it
        pipeline = (
            PipelineBuilder().add_source(range(100)).add_sink(1000).build(num_threads=1)
        )

        pipeline.start(timeout=30)

        # Verify the pipeline is running
        self.assertTrue(pipeline._impl._event_loop.is_started())

        # Keep a reference to the impl to verify it stopped
        impl = pipeline._impl

        # Execute: Delete the pipeline reference without calling stop()
        # The facade's finalizer should cleanly stop the pipeline
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Delete the pipeline and force garbage collection
            del pipeline
            gc.collect()

            # Assert: No warning should be issued — the facade handles cleanup
            cleanup_warnings = [
                warning
                for warning in w
                if "Pipeline is running in the background" in str(warning.message)
            ]
            self.assertEqual(len(cleanup_warnings), 0)

        # Verify the background thread actually stopped
        self.assertTrue(impl._event_loop.is_task_completed())

    def test_cleanup_not_called_when_explicitly_stopped(self) -> None:
        """Test that _cleanup_pipeline is not called when Pipeline is explicitly stopped."""

        # Setup: Create a pipeline and start it
        pipeline = (
            PipelineBuilder().add_source(range(100)).add_sink(1000).build(num_threads=1)
        )

        pipeline.start(timeout=30)

        # Execute: Explicitly stop the pipeline
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            pipeline.stop(timeout=30)

            # Delete the pipeline reference and force garbage collection
            del pipeline
            gc.collect()

            # Assert: Verify that no warning was issued
            # Since we explicitly stopped the pipeline, the finalizer should be detached
            # and no cleanup warning should be issued
            cleanup_warnings = [
                warning
                for warning in w
                if "Pipeline is running in the background" in str(warning.message)
            ]
            self.assertEqual(len(cleanup_warnings), 0)

    def test_cleanup_with_auto_stop_context_manager(self) -> None:
        """Test that cleanup is not called when using auto_stop context manager."""

        # Setup: Create a pipeline
        pipeline = (
            PipelineBuilder().add_source(range(10)).add_sink(1000).build(num_threads=1)
        )

        # Execute: Use the pipeline with auto_stop context manager
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with pipeline.auto_stop(timeout=30):
                # Get some items from the pipeline
                iterator = pipeline.get_iterator(timeout=30)
                for _ in range(5):
                    next(iterator)

            # Delete the pipeline reference and force garbage collection
            del pipeline
            gc.collect()

            # Assert: Verify that no cleanup warning was issued
            # The context manager should have stopped the pipeline properly
            cleanup_warnings = [
                warning
                for warning in w
                if "Pipeline is running in the background" in str(warning.message)
            ]
            self.assertEqual(len(cleanup_warnings), 0)

    def test_auto_start_and_cleanup_without_explicit_start_stop(self) -> None:
        """Test that iterating a pipeline without calling start/stop works:
        the background thread is started automatically on first iteration,
        and the finalizer cleans it up on garbage collection."""

        pipeline = (
            PipelineBuilder().add_source(range(10)).add_sink(1000).build(num_threads=1)
        )

        # Pipeline should not be started yet
        self.assertFalse(pipeline._impl._event_loop.is_started())

        # Iterate without explicit start — auto-start should kick in
        items = []
        for item in pipeline:
            items.append(item)

        # Verify auto-start happened
        self.assertTrue(pipeline._impl._event_loop.is_started())
        self.assertEqual(sorted(items), list(range(10)))

        # Keep a reference to the impl to verify cleanup
        impl = pipeline._impl

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            del pipeline
            gc.collect()

            cleanup_warnings = [
                warning
                for warning in w
                if "Pipeline is running in the background" in str(warning.message)
            ]
            self.assertEqual(len(cleanup_warnings), 0)

        # Verify the impl was stopped (stop was called by finalizer)
        self.assertTrue(impl._event_loop.is_task_completed())

    def test_auto_start_and_cleanup_continuous_source(self) -> None:
        """Test auto start/stop with a continuous source iterated multiple times.

        With continuous=True the source re-iterates, injecting epoch boundary
        sentinels. Each ``for ... in pipeline`` consumes one epoch. The pipeline
        should auto-start on the first epoch and remain running across epochs,
        then clean up on garbage collection."""

        pipeline = (
            PipelineBuilder()
            .add_source(range(5), continuous=True)
            .add_sink(1000)
            .build(num_threads=1)
        )

        self.assertFalse(pipeline._impl._event_loop.is_started())

        num_epochs = 3
        all_epoch_items = []
        for _ in range(num_epochs):
            epoch_items = []
            for item in pipeline:
                epoch_items.append(item)
            all_epoch_items.append(sorted(epoch_items))

        # Verify auto-start happened and each epoch produced the same items
        self.assertTrue(pipeline._impl._event_loop.is_started())
        for epoch_items in all_epoch_items:
            self.assertEqual(epoch_items, list(range(5)))

        # Verify cleanup on garbage collection
        impl = pipeline._impl

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            del pipeline
            gc.collect()

            cleanup_warnings = [
                warning
                for warning in w
                if "Pipeline is running in the background" in str(warning.message)
            ]
            self.assertEqual(len(cleanup_warnings), 0)

        self.assertTrue(impl._event_loop.is_task_completed())
