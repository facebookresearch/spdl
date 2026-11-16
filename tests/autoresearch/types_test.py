# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from spdl.autoresearch.core import (
    FailureKind,
    FailurePhase,
    FailureRecord,
    HypothesisNode,
    TaskSpec,
)


class TaskSpecTest(unittest.TestCase):
    def test_priority_deserialization_preserves_float_conversion(self) -> None:
        """Numeric values and numeric strings retain the existing priority conversion."""
        cases: list[tuple[object, float]] = [
            (3, 3.0),
            (2.5, 2.5),
            ("1.25", 1.25),
            (b"4.5", 4.5),
            (bytearray(b"6.5"), 6.5),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                spec = TaskSpec.from_dict({"id": "task", "priority": value})
                self.assertEqual(spec.priority, expected)
        self.assertEqual(TaskSpec.from_dict({"id": "task"}).priority, 0.0)

    def test_priority_deserialization_rejects_invalid_values(self) -> None:
        """The float conversion still rejects nonnumeric input instead of defaulting it."""
        with self.assertRaises(ValueError):
            TaskSpec.from_dict({"id": "task", "priority": "invalid"})
        for value in [None, object()]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                TaskSpec.from_dict({"id": "task", "priority": value})


class FailureRecordTest(unittest.TestCase):
    def test_round_trips_through_dict(self) -> None:
        """FailureRecord survives to_dict/from_dict serialization."""
        record = FailureRecord(
            kind=FailureKind.JOB_STARTUP_FAILED,
            phase=FailurePhase.JOB,
            message="MTP failed during initialization",
            details={"component": "mtp"},
            job_id="job123",
            created_at="2026-01-01T00:00:00",
        )

        loaded = FailureRecord.from_dict(record.to_dict())

        self.assertEqual(FailureKind.JOB_STARTUP_FAILED, loaded.kind)
        self.assertEqual(FailurePhase.JOB, loaded.phase)
        self.assertEqual("MTP failed during initialization", loaded.message)
        self.assertEqual({"component": "mtp"}, loaded.details)
        self.assertEqual("job123", loaded.job_id)


class HypothesisNodeTest(unittest.TestCase):
    def test_round_trips_through_dict(self) -> None:
        """HypothesisNode with a failure survives to_dict/from_dict."""
        node = HypothesisNode(
            node_id="001_bad_mtp",
            name="bad_mtp",
            status="failed",
            failure=FailureRecord(
                kind=FailureKind.JOB_STARTUP_FAILED,
                phase=FailurePhase.JOB,
                message="MTP failed during initialization",
                details={"component": "mtp"},
                job_id="job123",
                created_at="2026-01-01T00:00:00",
            ),
        )

        loaded = HypothesisNode.from_dict(node.to_dict())

        self.assertEqual("001_bad_mtp", loaded.node_id)
        self.assertEqual("failed", loaded.status)
        failure = loaded.failure
        assert failure is not None
        self.assertEqual(FailureKind.JOB_STARTUP_FAILED, failure.kind)
        self.assertEqual({"component": "mtp"}, failure.details)

    def test_round_trips_without_failure(self) -> None:
        """HypothesisNode without a failure round-trips cleanly."""
        node = HypothesisNode(
            node_id="000_baseline",
            name="baseline",
            status="completed",
            priority=-1000,
        )

        loaded = HypothesisNode.from_dict(node.to_dict())

        self.assertEqual("000_baseline", loaded.node_id)
        self.assertEqual("completed", loaded.status)
        self.assertIsNone(loaded.failure)
        self.assertEqual(-1000, loaded.priority)
