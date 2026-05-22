# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared autoresearch domain types."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum

TERMINAL_STATUSES = frozenset({"completed", "failed"})

__all__ = [
    "FailureKind",
    "FailurePhase",
    "FailureRecord",
    "AnalysisResult",
    "AutoresearchError",
    "HypothesisNode",
    "TERMINAL_STATUSES",
]


class FailureKind(str, Enum):
    """Stable workflow failure categories.

    These values are part of the autoresearch persistence contract. Future
    agents should add a new kind when a failure needs different retry,
    triage, or reporting behavior; do not collapse domain failures back into
    free-form strings in the runner.

    .. mermaid::

       flowchart LR
           Status["status = failed"]
           Record["FailureRecord"]
           Kind["FailureKind"]
           Reports["failure.json / history / engine_state"]

           Status --> Record
           Record --> Kind
           Record --> Reports
    """

    CODE_CHANGE_FAILED = "code_change_failed"
    SOURCE_RESTORE_FAILED = "source_restore_failed"
    BUILD_FAILED = "build_failed"
    LAUNCH_FAILED = "launch_failed"
    JOB_STARTUP_FAILED = "job_startup_failed"
    JOB_RUNTIME_FAILED = "job_runtime_failed"
    JOB_FAILED = "job_failed"
    JOB_STALLED = "job_stalled"
    METRICS_COLLECTION_FAILED = "metrics_collection_failed"
    ANALYSIS_FAILED = "analysis_failed"
    PLANNING_FAILED = "planning_failed"
    SETUP_INSTRUMENTATION_FAILED = "setup_instrumentation_failed"
    SETUP_SOURCE_FAILED = "setup_source_failed"
    UNEXPECTED_EXCEPTION = "unexpected_exception"
    INTERRUPTED = "interrupted"


class FailurePhase(str, Enum):
    """Workflow phase where a failure was detected."""

    PREPARE = "prepare"
    SOURCE = "source"
    CODE_CHANGE = "code_change"
    BUILD = "build"
    LAUNCH = "launch"
    JOB = "job"
    METRICS = "metrics"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    SETUP = "setup"
    INSTRUMENTATION = "instrumentation"
    RUNNER = "runner"


@dataclass(frozen=True)
class FailureRecord:
    """Machine-readable failure attached to a failed hypothesis node."""

    kind: FailureKind
    """Category of the failure."""

    phase: FailurePhase
    """Workflow phase where the failure occurred."""

    message: str
    """Human-readable description."""

    retryable: bool = False
    """Whether the engine should attempt a retry."""

    details: dict = field(default_factory=dict)
    """Arbitrary extra context for debugging."""

    exception_type: str | None = None
    """Fully qualified name of the original exception, if any."""

    job_id: str | None = None
    """Identifier of the failed job, if applicable."""

    created_at: str | None = None
    """ISO 8601 timestamp when the failure was recorded."""

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["kind"] = self.kind.value
        data["phase"] = self.phase.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> FailureRecord:
        return cls(
            kind=FailureKind(data["kind"]),
            phase=FailurePhase(data["phase"]),
            message=str(data.get("message", "")),
            retryable=bool(data.get("retryable", False)),
            details=dict(data.get("details", {})),
            exception_type=data.get("exception_type"),
            job_id=data.get("job_id"),
            created_at=data.get("created_at"),
        )


class AutoresearchError(RuntimeError):
    """Expected workflow failure carrying a structured failure record.

    Args:
        failure: The structured failure record describing what went wrong.
            Stored as the ``failure`` attribute and its ``message`` field
            is forwarded to ``RuntimeError``.
    """

    def __init__(self, failure: FailureRecord) -> None:
        super().__init__(failure.message)
        self.failure = failure


@dataclass
class HypothesisNode:
    """A node in the autoresearch hypothesis tree."""

    node_id: str
    """Unique identifier (e.g. ``"003_nvdec"``)."""

    name: str
    """Human-readable experiment name."""

    parent_id: str | None = None
    """ID of the parent node, or ``None`` for root nodes."""

    commit: str | None = None
    """Source-control commit hash for this experiment's code state."""

    spec: dict = field(default_factory=dict)
    """Free-form experiment specification dict."""

    status: str = "queued"
    """Current lifecycle status (``"queued"``, ``"running"``,
    ``"completed"``, ``"failed"``)."""

    job_id: str | None = None
    """External job identifier, if launched."""

    launched_at: float | None = None
    """Unix timestamp when the job was launched."""

    result: dict | None = None
    """Raw result dict from analysis."""

    children: list[str] = field(default_factory=list)
    """IDs of child nodes in the hypothesis tree."""

    priority: float = 0.0
    """Scheduling priority (lower runs first)."""

    duration: float | None = None
    """Job duration in seconds, if completed."""

    failure: FailureRecord | None = None
    """Structured failure record, if the experiment failed."""

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        if self.failure is not None:
            data["failure"] = self.failure.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> HypothesisNode:
        field_names = {field.name for field in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in field_names}
        if isinstance(values.get("failure"), dict):
            values["failure"] = FailureRecord.from_dict(values["failure"])
        return cls(**values)


@dataclass
class AnalysisResult:
    """Structured analysis returned after an experiment finishes."""

    structured: dict | None = None
    """Parsed JSON from the agent's analysis response."""

    duration: float | None = None
    """Job duration in seconds."""

    improved: bool = False
    """Whether the experiment improved on the current best."""

    failure: FailureRecord | None = None
    """Structured failure, if analysis detected a problem."""
