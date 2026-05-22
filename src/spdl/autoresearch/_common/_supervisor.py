# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Interactive supervisor-agent launchers for autoresearch.

The public autoresearch CLI always starts a supervisor agent. That agent owns
the interactive operator loop: asking for missing configuration, running the
engine, inspecting workdir files, and explaining failures. This module keeps
that role separate from ``_CodingAgent`` in ``utils.platform``; workflow agents
are stateless calls made by the engine, while supervisor agents are long-lived
interactive processes.

.. mermaid::

   flowchart TD
      CLI["spdl autoresearch"]
      Supervisor["_SupervisorAgent"]
      Claude["_ClaudeSupervisor"]
      Codex["_CodexSupervisor"]
      Engine["run.py engine command"]
      Workdir["workdir diagnostics"]

      CLI --> Supervisor
      Supervisor --> Claude
      Supervisor --> Codex
      Claude --> Engine
      Codex --> Engine
      Claude --> Workdir
      Codex --> Workdir
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "_ClaudeSupervisor",
    "_CodexSupervisor",
    "_SupervisorAvailability",
    "_SupervisorAgent",
    "_create_supervisor_agent",
    "_resolve_supervisor_agent",
]


@dataclass(frozen=True)
class _SupervisorAvailability:
    """Availability result for an interactive supervisor agent."""

    available: bool
    reason: str = ""


class _SupervisorAgent(Protocol):
    """Interactive agent process that supervises the autoresearch engine."""

    name: str

    def is_available(self) -> _SupervisorAvailability: ...

    def command(self, system_prompt: str, user_request: str) -> list[str]: ...


class _ClaudeSupervisor:
    """Claude Code interactive supervisor.

    This is intentionally a process launcher only. It should not know how the
    engine works beyond receiving the rendered operator prompt from the CLI.
    """

    name = "claude"

    def is_available(self) -> _SupervisorAvailability:
        if shutil.which("claude"):
            return _SupervisorAvailability(True)
        return _SupervisorAvailability(False, "claude command not found")

    def command(self, system_prompt: str, user_request: str) -> list[str]:
        command = ["claude", "--system-prompt", system_prompt]
        if user_request:
            command.append(user_request)
        return command


class _CodexSupervisor:
    """Codex interactive supervisor.

    The command shape is deliberately narrow and testable. If the Codex CLI
    changes, update this boundary instead of spreading command construction
    through launch scripts or workflow code.
    """

    name = "codex"

    def is_available(self) -> _SupervisorAvailability:
        if shutil.which("codex"):
            return _SupervisorAvailability(True)
        return _SupervisorAvailability(False, "codex command not found")

    def command(self, system_prompt: str, user_request: str) -> list[str]:
        prompt = system_prompt
        if user_request:
            prompt = f"{system_prompt}\n\n## User Request\n\n{user_request}"
        return ["codex", prompt]


def _create_supervisor_agent(kind: str) -> _SupervisorAgent:
    if kind == "claude":
        return _ClaudeSupervisor()
    if kind == "codex":
        return _CodexSupervisor()
    raise ValueError(f"Unknown autoresearch supervisor agent: {kind}")


def _resolve_supervisor_agent(kind: str) -> _SupervisorAgent:
    """Resolve ``auto`` or an explicit supervisor agent name."""

    if kind != "auto":
        agent = _create_supervisor_agent(kind)
        availability = agent.is_available()
        if not availability.available:
            raise RuntimeError(
                f"Supervisor agent '{kind}' is unavailable: {availability.reason}"
            )
        return agent

    unavailable = []
    for candidate in ("claude", "codex"):
        agent = _create_supervisor_agent(candidate)
        availability = agent.is_available()
        if availability.available:
            return agent
        unavailable.append(f"{candidate}: {availability.reason}")
    raise RuntimeError(
        "No autoresearch supervisor agent is available. " + "; ".join(unavailable)
    )
