# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Codex invocation for autoresearch."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_run_codex",
]


def _run_codex(
    prompt: str,
    workdir: Path,
    phase: str,
    command: list[str] | None = None,
) -> str:
    """Run a stateless Codex session and return the text response.

    Invokes the Codex CLI with the prompt on stdin, saves logs, and
    returns stdout on success.

    Args:
        prompt: The full prompt to send to Codex.
        workdir: Working directory for the subprocess and log output.
        phase: Label for this invocation (e.g. ``"apply_001"``), used
            in log filenames.
        command: CLI command to invoke. Defaults to
            ``["codex", "exec", "-"]``.

    Returns:
        The raw stdout from the Codex process.

    Raises:
        RuntimeError: If Codex exits with a non-zero return code.
    """
    cmd = command or ["codex", "exec", "-"]

    log_dir = workdir / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file = log_dir / f"{timestamp}_{phase}_prompt.md"
    output_file = log_dir / f"{timestamp}_{phase}_output.md"
    raw_file = log_dir / f"{timestamp}_{phase}_raw.txt"
    prompt_file.write_text(prompt)

    _LG.info("Running codex [phase=%s] cmd=%s", phase, cmd)
    result = subprocess.run(
        cmd,
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
