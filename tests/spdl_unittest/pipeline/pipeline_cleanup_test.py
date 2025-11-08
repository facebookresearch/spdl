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
        """Test that _cleanup_pipeline is called implicitly during garbage collection
        when Pipeline is not explicitly stopped."""

        # Setup: Create a pipeline and start it
        pipeline = (
            PipelineBuilder().add_source(range(100)).add_sink(1000).build(num_threads=1)
        )

        pipeline.start(timeout=3)

        # Verify the pipeline is running
        self.assertTrue(pipeline._event_loop.is_started())

        # Execute: Delete the pipeline reference without calling stop()
        # This should trigger the weakref.finalize cleanup
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Delete the pipeline and force garbage collection
            del pipeline
            gc.collect()

            # Assert: Verify that a warning was issued from the cleanup function
            # The cleanup function should warn when stopping an implicitly stopped pipeline
            cleanup_warnings = [
                warning
                for warning in w
                if "Pipeline is running in the background" in str(warning.message)
            ]
            self.assertGreaterEqual(len(cleanup_warnings), 1)
            self.assertIn(
                "Stopping the background thread",
                str(cleanup_warnings[0].message),
            )

    def test_cleanup_not_called_when_explicitly_stopped(self) -> None:
        """Test that _cleanup_pipeline is not called when Pipeline is explicitly stopped."""

        # Setup: Create a pipeline and start it
        pipeline = (
            PipelineBuilder().add_source(range(100)).add_sink(1000).build(num_threads=1)
        )

        pipeline.start(timeout=3)

        # Execute: Explicitly stop the pipeline
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            pipeline.stop(timeout=3)

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

            with pipeline.auto_stop(timeout=3):
                # Get some items from the pipeline
                iterator = pipeline.get_iterator(timeout=3)
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
