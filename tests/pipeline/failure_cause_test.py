# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import traceback
import unittest
from fractions import Fraction

from spdl.pipeline import build_pipeline, PipelineFailure
from spdl.pipeline.defs import (
    Pipe,
    PipelineConfig,
    PlacementConfig,
    ProcessPoolExecutorConfig,
    SinkConfig,
    SourceConfig,
)


class FailureCauseTest(unittest.TestCase):
    def _failure(
        self, config: PipelineConfig[object], *, max_failures: int | Fraction = 0
    ) -> PipelineFailure:
        pipeline = build_pipeline(config, num_threads=1, max_failures=max_failures)
        with self.assertRaises(PipelineFailure) as caught:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=10))
        return caught.exception

    def test_global_count_failure_keeps_original_exception(self) -> None:
        """The foreground reader can inspect the original task error and traceback."""
        config = PipelineConfig(
            SourceConfig(["bad sample"]), (Pipe(int),), SinkConfig(1)
        )
        failure = self._failure(config)
        cause = failure.exceptions[0].__cause__
        self.assertIsInstance(cause, ValueError)
        self.assertIn("bad sample", str(cause))
        self.assertIn("invalid literal", "".join(traceback.format_exception(failure)))

    def test_stage_override_keeps_original_exception(self) -> None:
        """A stage-specific failure cap reports its cause with unlimited global policy."""
        config = PipelineConfig(
            SourceConfig(["stage failure"]), (Pipe(int, max_failures=0),), SinkConfig(1)
        )
        failure = self._failure(config, max_failures=-1)
        self.assertIn("stage failure", str(failure.exceptions[0].__cause__))

    def test_count_limit_reports_the_failure_that_exceeds_it(self) -> None:
        """An earlier tolerated error does not replace the fatal task's cause."""
        config = PipelineConfig(
            SourceConfig(["tolerated", "fatal"]),
            (Pipe(int, concurrency=1, output_order="input"),),
            SinkConfig(1),
        )
        failure = self._failure(config, max_failures=1)
        self.assertIn("fatal", str(failure.exceptions[0].__cause__))
        self.assertNotIn("tolerated", str(failure.exceptions[0].__cause__))

    def test_rate_limit_keeps_cause_after_probation(self) -> None:
        """Strict rate comparison reports the error crossing the cap after warmup."""
        config = PipelineConfig(
            SourceConfig(["1"] * 50 + ["tolerated"] * 50 + ["fatal after probation"]),
            (Pipe(int, concurrency=1, output_order="input"),),
            SinkConfig(1),
        )
        failure = self._failure(config, max_failures=Fraction(1, 2))
        self.assertIn("fatal after probation", str(failure.exceptions[0].__cause__))

    def test_global_failure_cause_is_shared_across_stages(self) -> None:
        """A downstream global-limit failure identifies the failing upstream task."""
        config = PipelineConfig(
            SourceConfig(["upstream failure", "1", "2"]),
            (
                Pipe(int, concurrency=1, output_order="input", max_failures=-1),
                Pipe(abs, concurrency=1, output_order="input"),
            ),
            SinkConfig(1),
        )
        failure = self._failure(config)
        self.assertIn("upstream failure", str(failure.exceptions[0].__cause__))

    def test_process_region_preserves_worker_failure(self) -> None:
        """The parent traceback includes the cause from a normally spawned region."""
        config = PipelineConfig(
            SourceConfig(["process failure"]),
            (
                PlacementConfig(
                    ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn")
                ),
                Pipe(int, concurrency=1, output_order="input"),
            ),
            SinkConfig(1),
        )
        failure = self._failure(config)
        self.assertIn("process failure", "".join(traceback.format_exception(failure)))
