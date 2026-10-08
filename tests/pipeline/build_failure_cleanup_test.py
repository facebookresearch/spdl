# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

from spdl.pipeline import build_pipeline
from spdl.pipeline.defs import PipelineConfig, SinkConfig, SourceConfig


class BuildFailureCleanupTest(unittest.TestCase):
    def test_later_build_failure_shuts_down_eager_fused_pool(self) -> None:
        """A build failure releases a fused pool before ownership transfers."""
        config = PipelineConfig(
            src=SourceConfig(range(1)),
            pipes=[],
            sink=SinkConfig(1),
        )
        pool = MagicMock()

        with patch(
            "spdl.pipeline._build._fuse_marked_regions",
            return_value=(config, [pool]),
        ):
            # ThreadPoolExecutor rejects zero workers after region fusion has succeeded.
            with self.assertRaises(ValueError):
                build_pipeline(config, num_threads=0)

        pool.shutdown.assert_called_once_with()

    def test_constructor_failure_closes_all_unowned_resources(self) -> None:
        """Cleanup failures do not mask construction errors or skip resources."""
        config = PipelineConfig(
            src=SourceConfig(range(1)),
            pipes=[],
            sink=SinkConfig(1),
        )
        first_pool = MagicMock()
        second_pool = MagicMock()
        coro = MagicMock()
        output_queue = MagicMock()
        executor = MagicMock()
        coro.close.side_effect = RuntimeError("coroutine cleanup failed")
        executor.shutdown.side_effect = RuntimeError("executor cleanup failed")
        first_pool.shutdown.side_effect = RuntimeError("pool cleanup failed")

        with (
            patch(
                "spdl.pipeline._build._fuse_marked_regions",
                return_value=(config, [first_pool, second_pool]),
            ),
            patch(
                "spdl.pipeline._build.ThreadPoolExecutor",
                return_value=executor,
            ),
            patch(
                "spdl.pipeline._build._build_pipeline_coro",
                return_value=(coro, output_queue),
            ),
            patch(
                "spdl.pipeline._build.Pipeline",
                side_effect=RuntimeError("pipeline construction failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "pipeline construction failed"):
                build_pipeline(config, num_threads=1)

        coro.close.assert_called_once_with()
        executor.shutdown.assert_called_once_with()
        first_pool.shutdown.assert_called_once_with()
        second_pool.shutdown.assert_called_once_with()
