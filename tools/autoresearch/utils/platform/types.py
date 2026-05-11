# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autoresearch platform capability boundary.

Design note for future agents
=============================

The runner is deliberately simple: it runs coroutine work and persists queued
or running specs on cancellation. Autoresearch-specific behavior belongs in the
workflow and in this platform boundary. Keep platform operations grouped by
capability instead of adding callback arguments, hidden imports, or scheduler
logic to ``runner.py``.

``AutoresearchPlatform`` is intentionally small and boring. The workflow can
swap local, remote, Claude, Codex, or test implementations by replacing these
capability objects without learning any implementation-specific commands.

.. mermaid::

   flowchart LR
       Workflow["AutoresearchAdapter"]
       Platform["AutoresearchPlatform"]
       _Workspace["_Workspace"]
       _Artifacts["_Artifacts"]
       _Execution["_Execution"]
       _Evidence["_Evidence"]
       Agent["_CodingAgent"]
       Local["local implementations"]
       Remote["optional fb implementations"]

       Workflow --> Platform
       Platform --> _Workspace
       Platform --> _Artifacts
       Platform --> _Execution
       Platform --> _Evidence
       Platform --> Agent
       _Workspace --> Local
       _Artifacts --> Local
       _Execution --> Local
       _Evidence --> Local
       _Workspace --> Remote
       _Artifacts --> Remote
       _Execution --> Remote
       _Evidence --> Remote
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = [
    "AutoresearchPlatform",
    "_AgentResult",
    "_Artifacts",
    "_CodingAgent",
    "_Evidence",
    "_Execution",
    "_MetricsEvidence",
    "_PlatformError",
    "_Workspace",
]


@dataclass(frozen=True)
class _AgentResult:
    """Parsed coding-agent response used by workflow code."""

    raw_text: str
    json: dict | None = None
    parse_error: str | None = None


@dataclass(frozen=True)
class _MetricsEvidence:
    """Performance evidence collected for analysis.

    Local execution may still produce meaningful metrics. Treat missing
    ``system_metrics`` as an evidence limitation, not as proof that local mode
    is unguided.
    """

    system_metrics: str
    pipeline_stats_log: str
    metrics_summary: str
    log_paths: list[str] = field(default_factory=list)
    exit_code: int | None = None
    error_summary: str | None = None
    progress_seen: bool | None = None


@dataclass(frozen=True)
class _PlatformError(RuntimeError):
    """Structured platform failure before workflow classification."""

    capability: str
    operation: str
    message: str
    command: str | None = None
    returncode: int | None = None
    stderr_tail: str | None = None
    log_path: str | None = None
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.capability}.{self.operation}: {self.message}"


class _Workspace(Protocol):
    """Source-control and source-tree operations for workflow code.

    .. mermaid::

       flowchart LR
           Workflow["workflow/source_ops.py"]
           _Workspace["_Workspace"]
           SourceTree["source tree"]
           SCM["SCM"]

           Workflow --> _Workspace
           _Workspace --> SourceTree
           _Workspace --> SCM
    """

    def detect(self, source_dir: str) -> str: ...

    def current(self, scm_type: str, source_dir: str) -> str: ...

    def commit(self, scm_type: str, source_dir: str, message: str) -> str: ...

    def goto(
        self, scm_type: str, source_dir: str, target: str, anchor: str
    ) -> None: ...

    def has_changes(self, scm_type: str, source_dir: str) -> bool: ...

    def apply_lint(self, source_dir: str) -> None: ...


class _Artifacts(Protocol):
    """Build or resolve artifacts consumed by experiment launches."""

    def build(self, build_command: str, workdir: Path, cwd: str = "") -> str | None: ...


class _Execution(Protocol):
    """Launch, poll, and cancel experiment jobs."""

    def launch(self, command: str, workdir: Path) -> str | None: ...

    def status(self, job_id: str) -> str: ...

    def progress(self, job_id: str) -> str | None: ...

    def cancel(self, job_id: str) -> bool: ...

    def duration(self, job_id: str) -> int | None: ...


class _Evidence(Protocol):
    """Collect metrics, logs, and local evidence for completed jobs."""

    def collect(self, job_id: str, metrics_dir: Path) -> _MetricsEvidence: ...


class _CodingAgent(Protocol):
    """Prompt rendering and stateless coding-agent execution.

    Workflow code should depend on this protocol, not on a specific CLI. This
    keeps Claude, Codex, and deterministic test agents swappable.
    """

    def _load_prompt(self, name: str, **kwargs: str) -> str: ...

    def _load_knowledge(self) -> str: ...

    def run(self, prompt: str, workdir: Path, phase: str) -> str: ...

    def _extract_json_block(self, text: str) -> dict | None: ...


@dataclass(frozen=True)
class AutoresearchPlatform:
    """Capability aggregate used by the autoresearch workflow.

    The aggregate is not a service locator. It has exactly the capability
    groups that the workflow needs, so new implementation-specific behavior has
    a clear home and does not leak into runner or workflow scheduling logic.

    .. mermaid::

       flowchart TB
           Adapter["AutoresearchAdapter"]
           Platform["AutoresearchPlatform"]
           _Workspace["workspace"]
           _Artifacts["artifacts"]
           _Execution["execution"]
           _Evidence["evidence"]
           Agent["agent"]

           Adapter --> Platform
           Platform --> _Workspace
           Platform --> _Artifacts
           Platform --> _Execution
           Platform --> _Evidence
           Platform --> Agent
    """

    workspace: _Workspace
    artifacts: _Artifacts
    execution: _Execution
    evidence: _Evidence
    agent: _CodingAgent
