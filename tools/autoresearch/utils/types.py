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
from typing import NotRequired, TypedDict

_TERMINAL_STATUSES = frozenset({"completed", "failed"})

__all__ = [
    "FailureKind",
    "FailurePhase",
    "FailureRecord",
    "_AnalysisResult",
    "_AutoresearchConfig",
    "_AutoresearchError",
    "_AutoresearchState",
    "_HypothesisNode",
    "_StoppingCriteriaConfig",
    "_TERMINAL_STATUSES",
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
    phase: FailurePhase
    message: str
    retryable: bool = False
    details: dict = field(default_factory=dict)
    exception_type: str | None = None
    job_id: str | None = None
    created_at: str | None = None

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


class _AutoresearchError(RuntimeError):
    """Expected workflow failure carrying a structured failure record."""

    def __init__(self, failure: FailureRecord) -> None:
        super().__init__(failure.message)
        self.failure = failure


class _StoppingCriteriaConfig(TypedDict, total=False):
    max_iterations: int
    patience: int


class _AutoresearchConfig(TypedDict, total=False):
    schema_version: int
    pipeline_script: str | None
    source_dir: str
    scm: str
    build_command: str
    base_launch_command: str
    notes: str
    platform: str
    agent: str
    local_execution_mode: str
    stopping_criteria: _StoppingCriteriaConfig
    max_concurrency: int
    job_timeout_s: int
    poll_interval: int
    startup_failure_retries: int
    startup_retryable_experiments: list[str]
    claude_flags: list[str]


class _AutoresearchState(TypedDict):
    schema_version: int
    iteration: int
    status: str
    baseline_job: str | None
    current_best: str | None
    best_metric: float | None
    plateau_count: int
    best_practices_tried: list[str]
    anchor_commit: str
    history: list[dict]
    cached_image: NotRequired[str]


@dataclass
class _HypothesisNode:
    """A node in the autoresearch hypothesis tree."""

    node_id: str
    name: str
    parent_id: str | None = None
    commit: str | None = None
    spec: dict = field(default_factory=dict)
    status: str = "queued"
    job_id: str | None = None
    launched_at: float | None = None
    result: dict | None = None
    children: list[str] = field(default_factory=list)
    priority: float = 0.0
    duration: float | None = None
    failure: FailureRecord | None = None

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        if self.failure is not None:
            data["failure"] = self.failure.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> _HypothesisNode:
        field_names = {field.name for field in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in field_names}
        if isinstance(values.get("failure"), dict):
            values["failure"] = FailureRecord.from_dict(values["failure"])
        return cls(**values)


@dataclass
class _AnalysisResult:
    """Structured analysis returned after an experiment finishes."""

    structured: dict | None = None
    duration: float | None = None
    improved: bool = False
    failure: FailureRecord | None = None
