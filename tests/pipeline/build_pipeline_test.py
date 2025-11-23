# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest.mock import patch

from spdl.pipeline import build_pipeline
from spdl.pipeline._profile import _ProfilePipeline
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig

# pyre-strict


class TestBuildPipeline(unittest.TestCase):
    """Test class for build_pipeline functionality."""

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

        with patch.dict("os.environ", {"SPDL_PIPELINE_DIAGNOSTIC_MODE": "1"}):
            pipeline = build_pipeline(cfg, num_threads=2)
            self.assertIsInstance(pipeline, _ProfilePipeline)

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("SPDL_PIPELINE_DIAGNOSTIC_MODE", None)

            pipeline = build_pipeline(cfg, num_threads=2)
            self.assertNotIsInstance(pipeline, _ProfilePipeline)
