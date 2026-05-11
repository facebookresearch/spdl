# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coding-agent implementations for autoresearch."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import claude, prompts
from .types import _AgentResult

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_ClaudeAgent",
    "_CodexAgent",
    "_MockAgent",
    "_parse_agent_result",
    "_create_agent",
    "_extract_json_block",
]


class _ClaudeAgent:
    """Stateless Claude-backed coding agent."""

    def _load_prompt(self, name: str, **kwargs: str) -> str:
        return prompts._load_prompt(name, **kwargs)

    def _load_knowledge(self) -> str:
        return prompts._load_knowledge()

    def run(self, prompt: str, workdir: Path, phase: str) -> str:
        return claude._run_claude(prompt, workdir, phase)

    def _extract_json_block(self, text: str) -> dict | None:
        return _extract_json_block(text)


class _CodexAgent:
    """Stateless Codex-backed coding agent.

    The interface mirrors ``_ClaudeAgent`` so the workflow can swap agents
    without changing experiment logic. The command is intentionally configured
    outside the workflow because different environments expose Codex through
    different CLIs.
    """

    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command or ["codex", "exec", "-"]

    def _load_prompt(self, name: str, **kwargs: str) -> str:
        return prompts._load_prompt(name, **kwargs)

    def _load_knowledge(self) -> str:
        return prompts._load_knowledge()

    def run(self, prompt: str, workdir: Path, phase: str) -> str:
        log_dir = workdir / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_file = log_dir / f"{timestamp}_{phase}_prompt.md"
        output_file = log_dir / f"{timestamp}_{phase}_output.md"
        raw_file = log_dir / f"{timestamp}_{phase}_raw.txt"
        prompt_file.write_text(prompt)

        _LG.info("Running codex [phase=%s] cmd=%s", phase, self._command)
        result = subprocess.run(
            self._command,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=str(workdir),
            timeout=900,
        )
        raw_file.write_text(f"[STDOUT]\n{result.stdout}\n[STDERR]\n{result.stderr}")
        if result.returncode != 0:
            raise RuntimeError(
                f"Codex exited with code {result.returncode} during '{phase}'. "
                f"See {raw_file} for details.\n"
                f"stderr: {result.stderr[:500]}"
            )
        output_file.write_text(result.stdout)
        return result.stdout

    def _extract_json_block(self, text: str) -> dict | None:
        return _extract_json_block(text)


class _MockAgent:
    """Deterministic test agent backed by phase/prompt-name responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    def _load_prompt(self, name: str, **kwargs: str) -> str:
        template = self.responses.get(f"prompt:{name}", f"PROMPT:{name}")
        for key, value in kwargs.items():
            template = template.replace(f"__{key.upper()}__", str(value))
        return template

    def _load_knowledge(self) -> str:
        return self.responses.get("knowledge", "")

    def run(self, prompt: str, workdir: Path, phase: str) -> str:
        self.calls.append((phase, prompt))
        return self.responses.get(phase, self.responses.get("*", ""))

    def _extract_json_block(self, text: str) -> dict | None:
        return _extract_json_block(text)


def _create_agent(kind: str, *, command: list[str] | None = None) -> object:
    if kind == "claude":
        return _ClaudeAgent()
    if kind == "codex":
        return _CodexAgent(command)
    if kind == "mock":
        return _MockAgent()
    raise ValueError(f"Unknown autoresearch agent: {kind}")


def _parse_agent_result(agent: Any, raw_text: str) -> _AgentResult:
    try:
        parsed = agent._extract_json_block(raw_text)
    except Exception as error:
        return _AgentResult(raw_text=raw_text, parse_error=str(error))
    if parsed is None:
        return _AgentResult(raw_text=raw_text, parse_error="No JSON object found")
    return _AgentResult(raw_text=raw_text, json=parsed)


def _extract_json_block(text: str) -> dict | None:
    """Extract a `````json`` block from an agent response."""

    matches = list(re.finditer(r"```json\s*\n(.*?)\n```", text, re.DOTALL))
    if not matches:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    for match in reversed(matches):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    return None
