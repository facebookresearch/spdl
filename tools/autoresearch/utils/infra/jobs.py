# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Job launch, monitoring, and metrics collection.

Generic stubs are defined here. Platform implementations can wrap these helpers
or replace them behind ``utils/platform`` capabilities.

Design note for future agents
=============================

This module is an infrastructure helper, not part of the generic runner. Keep
scheduler-neutral job helpers here and keep platform-specific details behind
``utils/platform`` capabilities. Do not add job-platform behavior to
``runner.py``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
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
    "_wait_for_jobs",
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


def _get_check_job_status():
    """Get the active _check_job_status implementation.

    Platform implementations should call their status capability directly.
    This lookup remains for older helper-only callers.
    """
    pkg = sys.modules.get(__package__)
    if pkg is not None:
        fn = getattr(pkg, "_check_job_status", None)
        if fn is not None:
            return fn
    return _check_job_status


def _get_check_job_progress():
    """Get the active _check_job_progress implementation."""
    pkg = sys.modules.get(__package__)
    if pkg is not None:
        fn = getattr(pkg, "_check_job_progress", None)
        if fn is not None:
            return fn
    return _check_job_progress


def _get_cancel_job():
    """Get the active _cancel_job implementation."""
    pkg = sys.modules.get(__package__)
    if pkg is not None:
        fn = getattr(pkg, "_cancel_job", None)
        if fn is not None:
            return fn
    return _cancel_job


def _wait_for_jobs(
    job_ids: list[str],
    poll_interval: int = 120,
    max_unknown_polls: int = 5,
    max_stall_polls: int = 15,
) -> dict[str, dict[str, object]]:
    """Wait for jobs to complete. Returns {job_id: {"status": str, "elapsed_s": int}}.

    Kills stuck jobs (no progress for max_stall_polls consecutive polls).
    """
    results: dict[str, dict[str, object]] = {}
    pending = set(job_ids)
    start_time = time.monotonic()
    job_start: dict[str, float] = {j: start_time for j in job_ids}
    unknown_counts: dict[str, int] = dict.fromkeys(job_ids, int())
    stall_counts: dict[str, int] = dict.fromkeys(job_ids, int())
    last_progress: dict[str, str | None] = dict.fromkeys(job_ids, None)
    _check = _get_check_job_status()
    _progress = _get_check_job_progress()
    _cancel = _get_cancel_job()

    def _finish(job_id: str, status: str) -> None:
        elapsed_s = int(time.monotonic() - job_start[job_id])
        results[job_id] = {"status": status, "elapsed_s": elapsed_s}
        pending.discard(job_id)

    while pending:
        for job_id in list(pending):
            status = _check(job_id)
            if status in ("SUCCEEDED", "COMPLETE"):
                _finish(job_id, "completed")
                print(f"  [{job_id}] COMPLETED")
            elif status == "FAILED":
                _finish(job_id, "failed")
                print(f"  [{job_id}] FAILED")
            elif status == "UNKNOWN":
                unknown_counts[job_id] = unknown_counts.get(job_id, 0) + 1
                if unknown_counts[job_id] >= max_unknown_polls:
                    _cancel(job_id)
                    _finish(job_id, "failed")
                    print(
                        f"  [{job_id}] UNKNOWN after {max_unknown_polls} polls, killed"
                    )
                else:
                    print(
                        f"  [{job_id}] UNKNOWN "
                        f"({unknown_counts[job_id]}/{max_unknown_polls})"
                    )
            else:
                unknown_counts[job_id] = 0
                progress = _progress(job_id)
                if progress is not None:
                    prev = last_progress.get(job_id)
                    if prev is not None and progress == prev:
                        stall_counts[job_id] = stall_counts.get(job_id, 0) + 1
                        if stall_counts[job_id] >= max_stall_polls:
                            _cancel(job_id)
                            _finish(job_id, "failed")
                            print(
                                f"  [{job_id}] STUCK — no progress for "
                                f"{max_stall_polls} polls, killed"
                            )
                            continue
                        print(
                            f"  [{job_id}] {status} "
                            f"(no progress {stall_counts[job_id]}/{max_stall_polls})"
                        )
                    else:
                        stall_counts[job_id] = 0
                        last_progress[job_id] = progress
                        print(f"  [{job_id}] {status}")
                else:
                    print(f"  [{job_id}] {status}")

        if pending:
            print(f"  Waiting {poll_interval}s... ({len(pending)} job(s) pending)")
            time.sleep(poll_interval)

    return results


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
