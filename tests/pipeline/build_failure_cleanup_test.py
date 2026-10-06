# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

from spdl.pipeline import build_pipeline, run_pipeline_in_subprocess
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

    def test_subprocess_setup_failure_shuts_down_all_eager_pools(self) -> None:
        """Subprocess setup rollback releases fused and hoisted worker pools."""
        config = PipelineConfig(
            src=SourceConfig(range(1)),
            pipes=[],
            sink=SinkConfig(1),
        )
        fused_pool = MagicMock()
        hoisted_pool = MagicMock()

        with (
            patch(
                "spdl.pipeline._build._fuse_marked_regions",
                return_value=(config, [fused_pool]),
            ),
            patch(
                "spdl.pipeline._build._hoist_process_pools",
                return_value=(config, [hoisted_pool]),
            ),
            patch(
                "spdl.pipeline._build._make_config_executors_picklable",
                return_value=config,
            ),
            patch("spdl.pipeline._build._get_initializer", return_value=[]),
            patch(
                "spdl.pipeline._build.iterate_in_subprocess",
                side_effect=RuntimeError("subprocess setup failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "subprocess setup failed"):
                run_pipeline_in_subprocess(config, num_threads=1)

        hoisted_pool.shutdown.assert_called_once_with()
        fused_pool.shutdown.assert_called_once_with()

    def test_monitor_start_failure_joins_subprocess_before_pool_shutdown(self) -> None:
        """Monitor startup rollback joins the subprocess before closing shared pools."""
        config = PipelineConfig(
            src=SourceConfig(range(1)),
            pipes=[],
            sink=SinkConfig(1),
        )
        rollback_order: list[str] = []
        iterable = MagicMock()
        iterable._finalizer.side_effect = lambda: rollback_order.append("subprocess")
        fused_pool = MagicMock()
        fused_pool.shutdown.side_effect = lambda: rollback_order.append("fused")
        hoisted_pool = MagicMock()
        hoisted_pool.shutdown.side_effect = lambda: rollback_order.append("hoisted")

        with (
            patch(
                "spdl.pipeline._build._fuse_marked_regions",
                return_value=(config, [fused_pool]),
            ),
            patch(
                "spdl.pipeline._build._hoist_process_pools",
                return_value=(config, [hoisted_pool]),
            ),
            patch(
                "spdl.pipeline._build._make_config_executors_picklable",
                return_value=config,
            ),
            patch("spdl.pipeline._build._get_initializer", return_value=[]),
            patch(
                "spdl.pipeline._build.iterate_in_subprocess",
                return_value=iterable,
            ),
            patch(
                "spdl.pipeline._build._start_pool_monitors",
                side_effect=RuntimeError("monitor startup failed"),
            ) as start_monitors,
        ):
            with self.assertRaisesRegex(RuntimeError, "monitor startup failed"):
                run_pipeline_in_subprocess(config, num_threads=1)

        start_monitors.assert_called_once_with([hoisted_pool])
        iterable._finalizer.assert_called_once_with()
        hoisted_pool.shutdown.assert_called_once_with()
        fused_pool.shutdown.assert_called_once_with()
        self.assertEqual(rollback_order, ["subprocess", "hoisted", "fused"])
