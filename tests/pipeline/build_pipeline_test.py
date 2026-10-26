# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import warnings
from unittest.mock import patch

from spdl.pipeline import AsyncQueue, build_pipeline
from spdl.pipeline._components import StageInfo
from spdl.pipeline._profile import _ProfilePipeline
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig


class _SizeRecordingQueue(AsyncQueue):
    sizes: list[int] = []

    def __init__(self, info: StageInfo, *, buffer_size: int = 1) -> None:
        super().__init__(info, buffer_size=buffer_size)
        _SizeRecordingQueue.sizes.append(buffer_size)


class TestBuildPipeline(unittest.TestCase):
    """Test class for build_pipeline functionality."""

    def test_stage_buffer_size_sizes_every_queue_but_the_sink(self) -> None:
        """``stage_buffer_size`` sizes the source and pipe queues; the sink keeps its own."""
        cfg = PipelineConfig(
            src=SourceConfig(range(5)),
            pipes=[Pipe(lambda i: i + 1), Pipe(lambda i: i * 2)],
            sink=SinkConfig(4),
        )
        builds = {
            "default": lambda: build_pipeline(
                cfg, num_threads=1, queue_class=_SizeRecordingQueue
            ),
            "1": lambda: build_pipeline(
                cfg, num_threads=1, queue_class=_SizeRecordingQueue, stage_buffer_size=1
            ),
        }
        expected = {"default": [2, 2, 2, 4], "1": [1, 1, 1, 4]}
        for name, build in builds.items():
            with self.subTest(stage_buffer_size=name):
                _SizeRecordingQueue.sizes = []
                pipeline = build()
                with pipeline.auto_stop():
                    self.assertEqual(list(pipeline), [2, 4, 6, 8, 10])
                self.assertEqual(_SizeRecordingQueue.sizes, expected[name])

    def test_rejects_non_positive_stage_buffer_size(self) -> None:
        """A zero ``stage_buffer_size`` is rejected rather than read as unbounded."""
        cfg = PipelineConfig(src=SourceConfig(range(5)), pipes=[], sink=SinkConfig(1))
        with self.assertRaisesRegex(ValueError, "stage_buffer_size"):
            build_pipeline(cfg, num_threads=1, stage_buffer_size=0)

    def test_build_pipeline_diagnostic_mode(self) -> None:
        """Test that when SPDL_PIPELINE_DIAGNOSTIC_MODE=1, build_pipeline
        calls _build_pipeline_diagnostic_mode and returns _ProfilePipeline.
        """

        def simple_op(i: int) -> int:
            return i * 2

        cfg = PipelineConfig(
            src=SourceConfig(range(5)),
            pipes=[
                Pipe(simple_op),
            ],
            sink=SinkConfig(1),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="coroutine .* was never awaited",
                category=RuntimeWarning,
            )
            with patch.dict("os.environ", {"SPDL_PIPELINE_DIAGNOSTIC_MODE": "1"}):
                pipeline = build_pipeline(cfg, num_threads=2)
                self.assertIsInstance(pipeline, _ProfilePipeline)

            with patch.dict("os.environ", {}, clear=False):
                os.environ.pop("SPDL_PIPELINE_DIAGNOSTIC_MODE", None)

                pipeline = build_pipeline(cfg, num_threads=2)
                self.assertNotIsInstance(pipeline, _ProfilePipeline)
