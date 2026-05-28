# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from spdl.autoresearch.pipeline_optimization._ops._store import _set_queued_priority
from spdl.autoresearch.pipeline_optimization._platform import (
    _MetricsEvidence,
    AutoresearchPlatform,
    create_platform,
)
from spdl.autoresearch.pipeline_optimization._platform._agents import (
    _MockAgent,
    _parse_agent_result,
)

__all__: list[str] = []


def _echo_marker_command(line: str) -> str:
    """Build a cross-platform shell one-liner that prints ``line`` to stdout.

    Tests previously used POSIX ``printf``; on Windows ``cmd.exe`` lacks it,
    so we shell out to the current Python interpreter instead.
    """
    py = (
        shlex.quote(sys.executable)
        if sys.platform != "win32"
        else f'"{sys.executable}"'
    )
    return f'{py} -c "print({line!r})"'


class _PlatformTest(unittest.TestCase):
    def test_default_platform_has_capability_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {"SPDL_AUTORESEARCH_PLATFORM_PROVIDERS": ""},
            ):
                platform = create_platform("auto", Path(tmp))

            self.assertIsInstance(platform, AutoresearchPlatform)
            self.assertTrue(hasattr(platform, "workspace"))
            self.assertTrue(hasattr(platform, "artifacts"))
            self.assertTrue(hasattr(platform, "execution"))
            self.assertTrue(hasattr(platform, "evidence"))
            self.assertTrue(hasattr(platform, "agent"))

    def test_local_platform_runs_subprocess_and_collects_log_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            platform = create_platform("local", workdir)

            job_id = platform.execution.launch(
                _echo_marker_command("[autoresearch] step=1"),
                workdir,
            )
            self.assertIsNotNone(job_id)
            assert job_id is not None

            for _ in range(50):
                status = platform.execution.status(job_id)
                if status == "SUCCEEDED":
                    break
                time.sleep(0.05)
            self.assertEqual("SUCCEEDED", platform.execution.status(job_id))
            self.assertEqual(
                "[autoresearch] step=1", platform.execution.progress(job_id)
            )

            metrics_dir = workdir / "runs" / "000_baseline" / "metrics"
            evidence = platform.evidence.collect(job_id, metrics_dir)

            self.assertIsInstance(evidence, _MetricsEvidence)
            self.assertIn("system metrics unavailable", evidence.system_metrics)
            self.assertIn("[autoresearch] step=1", evidence.pipeline_stats_log)

    def test_local_dry_run_completes_without_launching_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            platform = create_platform(
                {"platform": "local", "local_execution_mode": "dry_run"},
                workdir,
            )

            job_id = platform.execution.launch(
                _echo_marker_command("not executed"), workdir
            )
            self.assertIsNotNone(job_id)
            assert job_id is not None

            self.assertEqual("SUCCEEDED", platform.execution.status(job_id))
            self.assertIn("dry_run", platform.execution.progress(job_id) or "")

    def test_local_dataloader_only_uses_dataloader_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            platform = create_platform(
                {
                    "platform": "local",
                    "local_execution_mode": "dataloader_only",
                    "local_dataloader_command": _echo_marker_command(
                        "[autoresearch] dataloader"
                    ),
                },
                workdir,
            )

            job_id = platform.execution.launch(
                _echo_marker_command("training"), workdir
            )
            self.assertIsNotNone(job_id)
            assert job_id is not None

            for _ in range(50):
                status = platform.execution.status(job_id)
                if status == "SUCCEEDED":
                    break
                time.sleep(0.05)

            self.assertEqual("SUCCEEDED", platform.execution.status(job_id))
            self.assertEqual(
                "[autoresearch] dataloader", platform.execution.progress(job_id)
            )

    def test_mock_agent_is_selected_independently_of_platform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            platform = create_platform(
                {"platform": "local", "agent": "mock"},
                Path(tmp),
            )

            self.assertEqual("", platform.agent.run("prompt", Path(tmp), "phase"))

    def test_platform_config_validation_rejects_bad_local_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "local execution mode"):
                create_platform(
                    {"platform": "local", "local_execution_mode": "unknown"},
                    Path(tmp),
                )

    def test_unknown_remote_provider_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {"SPDL_AUTORESEARCH_PLATFORM_PROVIDERS": ""},
            ):
                with self.assertRaisesRegex(
                    ValueError, "Unknown autoresearch platform"
                ):
                    create_platform(
                        {"platform": "unknown_remote", "agent": "mock"},
                        Path(tmp),
                    )

    def test_agent_result_reports_parse_errors_without_llm(self) -> None:
        agent = _MockAgent()

        parsed = _parse_agent_result(agent, '```json\n{"action": "stop"}\n```')
        failed = _parse_agent_result(agent, "not json")

        self.assertEqual({"action": "stop"}, parsed.json)
        self.assertIsNone(parsed.parse_error)
        self.assertIsNone(failed.json)
        self.assertEqual("No JSON object found", failed.parse_error)

    def test_queue_command_updates_checkpoint_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            engine = workdir / "engine"
            engine.mkdir(parents=True)
            (engine / "checkpoint.json").write_text(
                '{"status": "interrupted", "queued": ['
                '{"id": "001_a", "priority": 10, "kind": "experiment", '
                '"payload": {"node": {"node_id": "001_a"}}}], "running": []}\n'
            )

            _set_queued_priority(workdir, "001_a", -5)

            text = (engine / "checkpoint.json").read_text()
            self.assertIn('"priority": -5.0', text)


if __name__ == "__main__":
    unittest.main()
