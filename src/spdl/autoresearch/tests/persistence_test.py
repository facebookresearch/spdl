# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from spdl.autoresearch.core import (
    load_or_init,
    read_engine_state,
    TaskSpec,
    write_engine_state,
)

__all__: list[str] = []


class _PersistenceTest(unittest.TestCase):
    def test_round_trip_preserves_spec_fields(self) -> None:
        """write_engine_state followed by read_engine_state preserves all TaskSpec fields."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            queued = [
                TaskSpec(
                    id="exp_001",
                    priority=-1.5,
                    kind="experiment",
                    payload={"node": {"node_id": "exp_001"}, "extra": [1, 2, 3]},
                ),
                TaskSpec(id="exp_002", priority=0.0, kind="default", payload={}),
            ]
            running = [TaskSpec(id="exp_003", priority=2.0, kind="experiment")]

            write_engine_state(
                workdir, queued=queued, running=running, status="running"
            )
            result = read_engine_state(workdir)

            self.assertIsNotNone(result)
            assert result is not None  # for type checker
            got_queued, got_running, status = result
            self.assertEqual(status, "running")
            self.assertEqual([spec.id for spec in got_queued], ["exp_001", "exp_002"])
            self.assertEqual(got_queued[0].priority, -1.5)
            self.assertEqual(got_queued[0].kind, "experiment")
            self.assertEqual(
                got_queued[0].payload,
                {"node": {"node_id": "exp_001"}, "extra": [1, 2, 3]},
            )
            self.assertEqual([spec.id for spec in got_running], ["exp_003"])

    def test_read_returns_none_when_missing(self) -> None:
        """read_engine_state on a fresh workdir returns None instead of raising."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_engine_state(Path(tmp)))

    def test_status_round_trips(self) -> None:
        """All three orchestrator status values survive persistence."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            for status in ("running", "stopped", "interrupted"):
                write_engine_state(workdir, queued=[], running=[], status=status)
                result = read_engine_state(workdir)
                assert result is not None
                self.assertEqual(result[2], status)

    def test_write_creates_workdir(self) -> None:
        """write_engine_state creates the workdir if it does not yet exist."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "nested" / "fresh"
            self.assertFalse(workdir.exists())
            write_engine_state(workdir, queued=[], running=[], status="running")
            self.assertTrue((workdir / "engine_state.json").exists())

    def test_load_or_init_uses_factory_on_fresh_run(self) -> None:
        """load_or_init calls the factory exactly once when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            calls: list[int] = []

            def factory() -> list[TaskSpec]:
                calls.append(1)
                return [TaskSpec(id="seed", priority=0.0)]

            specs = load_or_init(workdir, factory)

            self.assertEqual(len(calls), 1)
            self.assertEqual([spec.id for spec in specs], ["seed"])

    def test_load_or_init_resumes_from_checkpoint(self) -> None:
        """load_or_init returns queued+running and skips the factory on resume."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            queued = [TaskSpec(id="q1", priority=-1.0)]
            running = [TaskSpec(id="r1", priority=0.0)]
            write_engine_state(
                workdir, queued=queued, running=running, status="interrupted"
            )

            def factory() -> list[TaskSpec]:
                self.fail("factory must not be invoked when checkpoint exists")

            specs = load_or_init(workdir, factory)

            self.assertEqual([spec.id for spec in specs], ["q1", "r1"])

    def test_read_rejects_malformed_json_object(self) -> None:
        """A non-object JSON file raises ValueError instead of silently misparsing."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "engine_state.json").write_text("[]\n")
            with self.assertRaises(ValueError):
                read_engine_state(workdir)

    def test_read_rejects_non_list_field(self) -> None:
        """A non-list queued/running field raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "engine_state.json").write_text(
                json.dumps({"status": "running", "queued": "oops", "running": []})
            )
            with self.assertRaises(ValueError):
                read_engine_state(workdir)

    def test_write_does_not_truncate_existing_checkpoint_on_failure(self) -> None:
        """A failing write leaves the previous engine_state.json intact.

        Simulates a mid-write interruption by patching the temp file's
        ``write_text`` to raise after the previous checkpoint has been
        written successfully. The reader must still see the previous
        valid checkpoint, never a truncated or partial file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            write_engine_state(
                workdir,
                queued=[TaskSpec(id="prev", priority=0.0)],
                running=[],
                status="running",
            )

            original_write_text = Path.write_text

            def _fail_on_tmp(self: Path, *args: object, **kwargs: object) -> int:
                if ".tmp." in self.name:
                    raise OSError("simulated mid-write failure")
                return original_write_text(self, *args, **kwargs)  # type: ignore[arg-type]

            with unittest.mock.patch.object(Path, "write_text", _fail_on_tmp):
                with self.assertRaises(OSError):
                    write_engine_state(
                        workdir,
                        queued=[TaskSpec(id="new", priority=0.0)],
                        running=[],
                        status="running",
                    )

            result = read_engine_state(workdir)
            assert result is not None
            queued, _, _ = result
            self.assertEqual([spec.id for spec in queued], ["prev"])
