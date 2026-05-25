# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Failure construction and classification for autoresearch workflow code.

The runner intentionally treats coroutine errors generically. Autoresearch
needs richer semantics because a failed source edit, a failed build, a job that
dies before training starts, and a job that crashes after producing metrics all
need different triage. Keep those domain decisions here and persist the
resulting ``FailureRecord`` on the node.

.. mermaid::

   flowchart LR
       Operation["workflow operation"]
       Error["AutoresearchError"]
       Adapter["PipelineOptimizationWorkflow"]
       Node["HypothesisNode.failure"]
       Store["failure.json / history"]

       Operation -->|"expected failure"| Error
       Error --> Adapter
       Adapter --> Node
       Node --> Store
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from spdl.autoresearch.core import (
    AutoresearchError,
    FailureKind,
    FailurePhase,
    FailureRecord,
)

from .._platform import _MetricsEvidence

__all__ = [
    "_FAILURE_POLICIES",
    "_FailurePolicy",
    "_classify_terminal_job_failure",
    "_failure_note",
    "_failure_summary",
    "_make_failure",
    "_raise_failure",
    "_read_failures",
    "_unexpected_failure",
]


@dataclass(frozen=True)
class _FailurePolicy:
    phase: FailurePhase
    retryable: bool
    retry_strategy: str
    action: str


_FAILURE_POLICIES = {
    FailureKind.CODE_CHANGE_FAILED: _FailurePolicy(
        FailurePhase.CODE_CHANGE,
        False,
        "none",
        "Inspect the coding-agent output and source patch.",
    ),
    FailureKind.SOURCE_RESTORE_FAILED: _FailurePolicy(
        FailurePhase.SOURCE,
        False,
        "none",
        "Inspect source-control state before retrying.",
    ),
    FailureKind.BUILD_FAILED: _FailurePolicy(
        FailurePhase.BUILD,
        False,
        "none",
        "Inspect the build log and generated source patch.",
    ),
    FailureKind.LAUNCH_FAILED: _FailurePolicy(
        FailurePhase.LAUNCH,
        False,
        "none",
        "Inspect the launch command and platform configuration.",
    ),
    FailureKind.JOB_STARTUP_FAILED: _FailurePolicy(
        FailurePhase.JOB,
        True,
        "repair_source_then_retry",
        "Repair initialization, import, or pickling issues before measuring.",
    ),
    FailureKind.JOB_RUNTIME_FAILED: _FailurePolicy(
        FailurePhase.JOB,
        False,
        "none",
        "Inspect runtime logs and metrics from the failed workload.",
    ),
    FailureKind.JOB_FAILED: _FailurePolicy(
        FailurePhase.JOB,
        False,
        "none",
        "Inspect job logs to classify the failure more specifically.",
    ),
    FailureKind.JOB_STALLED: _FailurePolicy(
        FailurePhase.JOB,
        False,
        "none",
        "Inspect progress logs and timeout settings.",
    ),
    FailureKind.METRICS_COLLECTION_FAILED: _FailurePolicy(
        FailurePhase.METRICS,
        False,
        "none",
        "Inspect evidence collection logs and platform metrics availability.",
    ),
    FailureKind.ANALYSIS_FAILED: _FailurePolicy(
        FailurePhase.ANALYSIS,
        False,
        "none",
        "Inspect coding-agent analysis output.",
    ),
    FailureKind.PLANNING_FAILED: _FailurePolicy(
        FailurePhase.PLANNING,
        False,
        "none",
        "Inspect coding-agent planning output.",
    ),
    FailureKind.SETUP_INSTRUMENTATION_FAILED: _FailurePolicy(
        FailurePhase.INSTRUMENTATION,
        False,
        "none",
        "Inspect instrumentation prompt output.",
    ),
    FailureKind.SETUP_SOURCE_FAILED: _FailurePolicy(
        FailurePhase.SETUP,
        False,
        "none",
        "Inspect source-control access and source directory.",
    ),
    FailureKind.UNEXPECTED_EXCEPTION: _FailurePolicy(
        FailurePhase.RUNNER,
        False,
        "none",
        "Inspect the exception and add a structured failure kind if needed.",
    ),
    FailureKind.INTERRUPTED: _FailurePolicy(
        FailurePhase.RUNNER,
        True,
        "resume",
        "Resume from the engine checkpoint.",
    ),
}

_STARTUP_PATTERNS = (
    "can't pickle",
    "cannot pickle",
    "pickle",
    "modulenotfounderror",
    "importerror",
    "syntaxerror",
    "configuration",
    "config error",
    "initialization",
    "initialize",
    "startup",
    "mtp",
)

_RUNTIME_PATTERNS = (
    "cuda out of memory",
    "outofmemory",
    "oom",
    "dataloader",
    "data loader",
    "training step",
    "iteration",
    "epoch",
)

_METRIC_MARKERS = (
    "step_time",
    "steady_step_time",
    "ttfb",
    "data_readiness",
    "sm_util",
    "[autoresearch]",
)


def _make_failure(
    kind: FailureKind,
    phase: FailurePhase,
    message: str,
    *,
    retryable: bool = False,
    details: Mapping[str, object] | None = None,
    exception: BaseException | None = None,
    job_id: str | None = None,
) -> FailureRecord:
    policy = _FAILURE_POLICIES[kind]
    return FailureRecord(
        kind=kind,
        phase=phase,
        message=message,
        retryable=retryable or policy.retryable,
        details=dict(details or {}),
        exception_type=type(exception).__name__ if exception is not None else None,
        job_id=job_id,
        created_at=_timestamp(),
    )


def _raise_failure(
    kind: FailureKind,
    phase: FailurePhase,
    message: str,
    *,
    retryable: bool = False,
    details: Mapping[str, object] | None = None,
    exception: BaseException | None = None,
    job_id: str | None = None,
) -> NoReturn:
    raise AutoresearchError(
        _make_failure(
            kind,
            phase,
            message,
            retryable=retryable,
            details=details,
            exception=exception,
            job_id=job_id,
        )
    )


def _unexpected_failure(
    error: BaseException, *, job_id: str | None = None
) -> FailureRecord:
    return _make_failure(
        FailureKind.UNEXPECTED_EXCEPTION,
        FailurePhase.RUNNER,
        str(error) or type(error).__name__,
        exception=error,
        job_id=job_id,
    )


def _failure_note(failure: FailureRecord) -> str:
    policy = _FAILURE_POLICIES.get(failure.kind)
    action = f" Action: {policy.action}" if policy is not None else ""
    return f"{failure.kind.value}: {failure.message}{action}"


def _classify_terminal_job_failure(
    evidence: _MetricsEvidence,
    *,
    job_id: str,
    progress_seen: bool,
) -> FailureRecord:
    """Classify a terminal failed job from logs and metric evidence.

    Platform status only says that a job failed. The workflow owns this
    classifier because it can combine evidence with SPDL conventions:
    no metrics plus pickle/import/init errors means startup; metrics or
    progress before the crash means runtime; otherwise keep the conservative
    ``JOB_FAILED`` fallback.
    """

    text = "\n".join(
        (
            evidence.system_metrics,
            evidence.pipeline_stats_log,
            evidence.metrics_summary,
            evidence.error_summary or "",
        )
    ).lower()
    has_metrics = _has_metric_evidence(text)
    progress_observed = progress_seen or bool(evidence.progress_seen)
    details = {
        "progress_seen": progress_observed,
        "has_metric_evidence": has_metrics,
        "exit_code": evidence.exit_code,
        "error_summary": evidence.error_summary,
        "log_paths": evidence.log_paths,
    }

    if not has_metrics and any(pattern in text for pattern in _STARTUP_PATTERNS):
        return _make_failure(
            FailureKind.JOB_STARTUP_FAILED,
            FailurePhase.JOB,
            "Job failed during startup before measured workload began",
            details=details,
            job_id=job_id,
        )

    if (
        progress_observed
        or has_metrics
        or any(pattern in text for pattern in _RUNTIME_PATTERNS)
    ):
        return _make_failure(
            FailureKind.JOB_RUNTIME_FAILED,
            FailurePhase.JOB,
            "Job failed after the workload started",
            details=details,
            job_id=job_id,
        )

    return _make_failure(
        FailureKind.JOB_FAILED,
        FailurePhase.JOB,
        "Job reached terminal failed status",
        details=details,
        job_id=job_id,
    )


def _has_metric_evidence(text: str) -> bool:
    if "unavailable" in text and not any(marker in text for marker in _METRIC_MARKERS):
        return False
    return any(marker in text for marker in _METRIC_MARKERS)


def _read_failures(workdir: Path) -> str:
    """Render persisted node and setup failures as a markdown bullet list.

    Reads ``engine/tree.json`` and ``engine/setup_failures.json`` from
    ``workdir`` and produces a string like::

        - 001_a: build_failed - command exited 1
        - setup: source_setup_failed - missing source dir

    Returns ``"(none)"`` when no failures are recorded.
    """
    lines = []
    tree = workdir / "engine" / "tree.json"
    if tree.exists():
        for raw in json.loads(tree.read_text()):
            failure = raw.get("failure")
            if failure:
                lines.append(
                    "- "
                    f"{raw.get('node_id', '')}: {failure.get('kind', '')} - "
                    f"{failure.get('message', '')}"
                )
    setup = workdir / "engine" / "setup_failures.json"
    if setup.exists():
        for failure in json.loads(setup.read_text()):
            lines.append(
                f"- setup: {failure.get('kind', '')} - {failure.get('message', '')}"
            )
    return "\n".join(lines) if lines else "(none)"


def _failure_summary(workdir: Path) -> str:
    """Render persisted failures as a tab-separated table for status output.

    Each line lists ``node_id``, ``job_id``, failure ``kind``, and a
    descriptive message. Returns ``"Failures  : none"`` when no failures
    are recorded.
    """
    tree = workdir / "engine" / "tree.json"
    failed = []
    if tree.exists():
        for raw in json.loads(tree.read_text()):
            failure = raw.get("failure")
            if not failure:
                continue
            failed.append(
                "  "
                + "\t".join(
                    [
                        str(raw.get("node_id", "")),
                        str(raw.get("job_id") or ""),
                        str(failure.get("kind", "")),
                        _node_message(raw, failure),
                    ]
                )
            )
    setup = _setup_failure_lines(workdir)
    if not failed and not setup:
        return "Failures  : none"
    lines = ["Failures  :"]
    lines.extend(failed)
    lines.extend(setup)
    return "\n".join(lines)


def _setup_failure_lines(workdir: Path) -> list[str]:
    path = workdir / "engine" / "setup_failures.json"
    if not path.exists():
        return []
    failures = json.loads(path.read_text())
    return [
        "  "
        + "\t".join(
            [
                "setup",
                "",
                str(failure.get("kind", "")),
                str(failure.get("message", "")),
            ]
        )
        for failure in failures
    ]


def _node_message(raw: dict, failure: dict) -> str:
    spec = raw.get("spec") or {}
    parts = []
    retry_of = spec.get("_startup_retry_of")
    retry_attempt = spec.get("_startup_retry_attempt")
    if retry_of:
        parts.append(f"retry {retry_attempt} of {retry_of}")
    parts.append(str(failure.get("message", "")))
    return "; ".join(part for part in parts if part)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
