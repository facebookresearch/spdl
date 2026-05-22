# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Job launch, monitoring, and metrics collection.

Generic stubs are defined here. Platform implementations can wrap these helpers
or replace them behind ``utils/platform`` capabilities.

Design note
===========

This module is an infrastructure helper, not part of the generic runner. Keep
scheduler-neutral job helpers here and keep platform-specific details behind
``utils/platform`` capabilities. Do not add job-platform behavior to
``runner.py``.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_apply_lint",
    "_build_image",
    "_cancel_job",
    "_check_job_progress",
    "_check_job_status",
    "_collect_metrics_summary",
    "_fetch_pipeline_stats",
    "_fetch_system_metrics",
    "_get_job_duration",
    "_launch_job",
]


def _fetch_system_metrics(job_id: str) -> str:
    """Fetch system metrics (GPU/CPU) for a training job.

    Override this through a platform _Evidence implementation.
    """
    return "(system metrics not available — implement _fetch_system_metrics)"


def _fetch_pipeline_stats(job_id: str, output_dir: str) -> str:
    """Fetch SPDL pipeline performance stats for a training job.

    Override this through a platform _Evidence implementation.
    """
    return "(pipeline stats not available — implement _fetch_pipeline_stats)"


def _check_job_status(job_id: str) -> str:
    """Check training job status. Override for your infrastructure."""
    return "UNKNOWN"


def _check_job_progress(job_id: str) -> str | None:
    """Return a progress indicator for a running job.

    Should return a string that changes when the job makes progress,
    or None if progress checking is not available.
    Override this through a platform _Execution implementation.
    """
    return None


def _get_job_duration(job_id: str) -> int | None:
    """Get the actual runtime of a job in seconds (excludes queue time).
    Override for your infrastructure."""
    return None


def _cancel_job(job_id: str) -> bool:
    """Cancel a running job. Returns True on success.
    Override for your infrastructure."""
    return False


def _apply_lint(source_dir: str) -> None:
    """Run linter/formatter and fix build deps after code changes.
    Override for your infrastructure."""


def _launch_job(command: str, workdir: Path) -> str | None:
    """Launch a training job and return its ID."""
    _LG.info("Launching job: %s", command)
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(workdir),
    )
    _LG.debug(
        "_launch_job returncode=%d stdout=%s stderr=%s",
        result.returncode,
        result.stdout[:500],
        result.stderr[:500],
    )
    combined = result.stdout + result.stderr
    for pattern in [
        r"Launched app:\s*\S+/(\S+)",
        r"Job ID:\s*(\S+)",
        r"job[_-]?id[=: ]+(\S+)",
    ]:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            job_id = match.group(1)
            _LG.info("Extracted job ID: %s", job_id)
            return job_id
    _LG.error("Could not extract job ID from output:\n%s", combined[:500])
    return None


def _build_image(build_command: str, workdir: Path, cwd: str = "") -> str | None:
    if not build_command:
        return None
    run_cwd = cwd or str(workdir)
    _LG.info("Building image: %s (cwd=%s)", build_command, run_cwd)
    result = subprocess.run(
        build_command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=run_cwd,
    )
    _LG.debug("_build_image returncode=%d", result.returncode)
    if result.returncode != 0:
        _LG.error(
            "Build failed (rc=%d):\n[STDOUT]\n%s\n[STDERR]\n%s",
            result.returncode,
            result.stdout[-2000:] if result.stdout else "(empty)",
            result.stderr[-2000:] if result.stderr else "(empty)",
        )
        build_log = workdir / "logs" / "last_build_error.txt"
        build_log.parent.mkdir(exist_ok=True)
        build_log.write_text(
            f"[COMMAND]\n{build_command}\n\n"
            f"[STDOUT]\n{result.stdout}\n\n"
            f"[STDERR]\n{result.stderr}\n"
        )
        print(f"  Build error details saved to {build_log}")
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        _LG.error("Build produced no output")
        return None
    image = lines[-1]
    _LG.info("Built image: %s", image)
    return image


def _collect_metrics_summary(metrics_dir: Path) -> str:
    parts = []
    if not metrics_dir.exists():
        return "(no metrics files found)"
    for f in sorted(metrics_dir.iterdir()):
        if f.suffix == ".tsv":
            content = f.read_text()
            lines = content.strip().splitlines()
            if len(lines) > 52:
                content = "\n".join(lines[:2] + ["..."] + lines[-50:])
            parts.append(f"\n### {f.name}\n```\n{content}\n```")
    return "\n".join(parts) if parts else "(no .tsv files found)"
