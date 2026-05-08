# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Claude invocation and prompt template loading."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .state import read_config

_LG: logging.Logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"


def load_prompt(name: str, **kwargs: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        print(f"Error: prompt template not found: {path}", file=sys.stderr)
        sys.exit(1)
    template = path.read_text()
    for key, value in kwargs.items():
        template = template.replace(f"__{key.upper()}__", str(value))
    return template


def load_knowledge() -> str:
    """Load shared skill files and autoresearch-specific knowledge.

    Reads symlinked skill files from prompts/ for shared pipeline
    optimization knowledge, then appends autoresearch-specific
    knowledge from knowledge.md and fb/knowledge.md.
    """
    parts: list[str] = []
    for name in ["optimization_strategies.md", "how_to_interpret_pipeline_stats.md"]:
        path = PROMPTS_DIR / name
        if path.exists():
            parts.append(path.read_text())
    parts.append(load_prompt("knowledge"))
    for name in [
        "fetching_pipeline_stats.md",
        "mast_job_lifecycle.md",
        "knowledge.md",
    ]:
        path = PROMPTS_DIR / "fb" / name
        if path.exists():
            parts.append(path.read_text())
    return "\n\n".join(parts)


def run_claude(prompt: str, workdir: Path, phase: str) -> str:
    """Run a stateless Claude session and return the text response.

    Uses --output-format json to get structured output, then extracts
    the ``result`` field. Saves both the raw JSON response and the
    extracted text to the logs directory.
    """
    log_dir = workdir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file = log_dir / f"{timestamp}_{phase}_prompt.md"
    output_file = log_dir / f"{timestamp}_{phase}_output.md"
    raw_file = log_dir / f"{timestamp}_{phase}_raw.json"
    prompt_file.write_text(prompt)

    config = read_config(workdir)
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--no-session-persistence",
    ]
    cmd.extend(config.get("claude_flags", []))

    _LG.info("Running claude [phase=%s] cmd=%s prompt_len=%d", phase, cmd, len(prompt))
    _LG.debug("Prompt saved to %s", prompt_file)

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        cwd=str(workdir),
    )

    _LG.info(
        "Claude finished [phase=%s] returncode=%d stdout_len=%d stderr_len=%d",
        phase,
        result.returncode,
        len(result.stdout),
        len(result.stderr),
    )

    if result.returncode != 0:
        raw_file.write_text(f"[STDOUT]\n{result.stdout}\n[STDERR]\n{result.stderr}")
        msg = (
            f"Claude exited with code {result.returncode} during '{phase}'. "
            f"See {raw_file} for details.\n"
            f"stderr: {result.stderr[:500]}"
        )
        _LG.error(msg)
        raise RuntimeError(msg)

    raw_file.write_text(result.stdout)

    text = _extract_result_text(result.stdout, phase)
    output_file.write_text(text)

    _LG.info(
        "Claude [phase=%s] cost=$%.4f duration=%sms",
        phase,
        _safe_get(result.stdout, "total_cost_usd", 0),
        _safe_get(result.stdout, "duration_ms", "?"),
    )
    _LG.debug("Output saved to %s", output_file)
    return text


def _extract_result_text(raw_stdout: str, phase: str) -> str:
    """Extract the result text from Claude's JSON output.

    The JSON line may be preceded by stderr-like banner lines on stdout
    (e.g. "Claude Code at Meta ..."). We find the JSON object by looking
    for the last line that starts with '{'.
    """
    json_line = ""
    for line in raw_stdout.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            json_line = stripped

    if not json_line:
        _LG.warning(
            "No JSON found in Claude output for phase '%s', returning raw", phase
        )
        return raw_stdout

    try:
        data = json.loads(json_line)
    except json.JSONDecodeError:
        _LG.warning("Failed to parse Claude JSON for phase '%s', returning raw", phase)
        return raw_stdout

    if data.get("is_error"):
        msg = f"Claude returned error for '{phase}': {data.get('result', '')}"
        _LG.error(msg)
        raise RuntimeError(msg)

    return data.get("result", raw_stdout)


def _safe_get(raw_stdout: str, key: str, default: object = None) -> object:
    """Extract a field from the JSON output, returning default on failure."""
    for line in raw_stdout.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped).get(key, default)
            except json.JSONDecodeError:
                pass
    return default


def extract_json_block(text: str) -> dict | None:
    """Extract a ```json ... ``` code block from Claude's text response."""
    matches = list(re.finditer(r"```json\s*\n(.*?)\n```", text, re.DOTALL))
    if not matches:
        return None
    for match in reversed(matches):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    return None
