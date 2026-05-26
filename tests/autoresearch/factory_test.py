# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from spdl.autoresearch.pipeline_optimization import create_workflow

__all__: list[str] = []


def _full_argv() -> list[str]:
    return [
        "--pipeline-script",
        "/tmp/pipeline.py",
        "--source-dir",
        "/tmp/src",
        "--build-command",
        "make image",
        "--base-launch-command",
        "torchx run --image $IMAGE",
        "--notes",
        "smoke",
        "--max-iterations",
        "5",
        "--patience",
        "2",
        "--job-timeout",
        "300",
    ]


class _CreateWorkflowTest(unittest.TestCase):
    def test_returns_workflow_spec(self) -> None:
        """create_workflow returns an object exposing the WorkflowSpec surface."""
        spec = create_workflow(_full_argv(), None)
        for attr in (
            "engine_argv_tail",
            "description",
            "supervisor_known_config",
            "supervisor_missing_config",
            "setup",
            "build_workflow",
        ):
            self.assertTrue(callable(getattr(spec, attr)), attr)
        self.assertEqual(spec.max_concurrency, 3)

    def test_engine_argv_tail_round_trips_supplied_flags(self) -> None:
        """Every value passed in survives in engine_argv_tail in flag/value order."""
        spec = create_workflow(_full_argv(), None)
        tail = spec.engine_argv_tail()
        self.assertEqual(tail[tail.index("--pipeline-script") + 1], "/tmp/pipeline.py")
        self.assertEqual(tail[tail.index("--source-dir") + 1], "/tmp/src")
        self.assertEqual(tail[tail.index("--build-command") + 1], "make image")
        self.assertEqual(
            tail[tail.index("--base-launch-command") + 1],
            "torchx run --image $IMAGE",
        )
        self.assertEqual(tail[tail.index("--max-iterations") + 1], "5")
        self.assertEqual(tail[tail.index("--patience") + 1], "2")
        self.assertEqual(tail[tail.index("--max-concurrency") + 1], "3")
        self.assertEqual(tail[tail.index("--job-timeout") + 1], "300")
        self.assertEqual(tail[tail.index("--platform") + 1], "auto")

    def test_engine_argv_tail_omits_unset_options(self) -> None:
        """Unset optional flags are not emitted at all."""
        spec = create_workflow([], None)
        tail = spec.engine_argv_tail()
        self.assertNotIn("--pipeline-script", tail)
        self.assertNotIn("--build-command", tail)
        self.assertNotIn("--base-launch-command", tail)
        self.assertNotIn("--source-dir", tail)
        self.assertNotIn("--notes", tail)

    def test_engine_argv_tail_emits_boolean_flags(self) -> None:
        """Boolean flags appear by themselves with no value when set."""
        spec = create_workflow(
            [
                "--skip-instrument",
                "--dangerously-skip-permissions",
            ],
            None,
        )
        tail = spec.engine_argv_tail()
        self.assertIn("--skip-instrument", tail)
        self.assertIn("--dangerously-skip-permissions", tail)

    def test_supervisor_missing_config_lists_required_fields(self) -> None:
        """A bare invocation reports all four required fields as missing."""
        spec = create_workflow([], None)
        missing = spec.supervisor_missing_config()
        self.assertIn("pipeline script", missing)
        self.assertIn("source directory", missing)
        self.assertIn("build command", missing)
        self.assertIn("launch command template", missing)

    def test_supervisor_missing_config_empty_when_all_supplied(self) -> None:
        """A fully-configured invocation reports no missing fields."""
        spec = create_workflow(_full_argv(), None)
        self.assertEqual(spec.supervisor_missing_config(), [])

    def test_supervisor_known_config_reflects_argv(self) -> None:
        """supervisor_known_config exposes the parsed values for the supervisor prompt."""
        spec = create_workflow(_full_argv(), None)
        known = spec.supervisor_known_config()
        self.assertEqual(known["pipeline_script"], "/tmp/pipeline.py")
        self.assertEqual(known["build_command"], "make image")
        self.assertEqual(known["local_execution_mode"], "full")

    def test_max_concurrency_reflects_argv(self) -> None:
        """A non-default --max-concurrency is visible on spec.max_concurrency."""
        spec = create_workflow([*_full_argv(), "--max-concurrency", "7"], None)
        self.assertEqual(spec.max_concurrency, 7)

    def test_description_contains_supervisor_and_platform_content(self) -> None:
        """description() joins the supervisor and platform prompt directories."""
        spec = create_workflow(_full_argv(), None)
        description = spec.description()
        self.assertIsNotNone(description)
        assert description is not None  # for type checker
        self.assertIn("---", description)
        self.assertIn("Automated SPDL Pipeline Optimization", description)
