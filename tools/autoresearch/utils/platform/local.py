# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local and generic platform capabilities."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from spdl.autoresearch._common import _jobs as jobs, _scm as scm

from .types import _MetricsEvidence

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_DefaultArtifacts",
    "_DefaultEvidence",
    "_DefaultExecution",
    "_DefaultWorkspace",
    "_LocalEvidence",
    "_LocalExecution",
    "_LocalWorkspace",
    "_extract_traceback",
    "_summarize_error",
]


class _DefaultWorkspace:
    def detect(self, source_dir: str) -> str:
        return scm._detect_scm(source_dir)

    def current(self, scm_type: str, source_dir: str) -> str:
        return scm._current_commit(scm_type, source_dir)

    def commit(self, scm_type: str, source_dir: str, message: str) -> str:
        return scm._commit(scm_type, source_dir, message)

    def goto(self, scm_type: str, source_dir: str, target: str, anchor: str) -> None:
        scm._goto(scm_type, source_dir, target, anchor)

    def has_changes(self, scm_type: str, source_dir: str) -> bool:
        return scm._has_pending_changes(scm_type, source_dir)

    def apply_lint(self, source_dir: str) -> None:
        jobs._apply_lint(source_dir)


class _LocalWorkspace(_DefaultWorkspace):
    def apply_lint(self, source_dir: str) -> None:
        _LG.info("Skipping infrastructure lint for local platform: %s", source_dir)


class _DefaultArtifacts:
    def build(self, build_command: str, workdir: Path, cwd: str = "") -> str | None:
        return jobs._build_image(build_command, workdir, cwd)


class _DefaultExecution:
    def launch(self, command: str, workdir: Path) -> str | None:
        return jobs._launch_job(command, workdir)

    def status(self, job_id: str) -> str:
        return jobs._check_job_status(job_id)

    def progress(self, job_id: str) -> str | None:
        return jobs._check_job_progress(job_id)

    def cancel(self, job_id: str) -> bool:
        return jobs._cancel_job(job_id)

    def duration(self, job_id: str) -> int | None:
        return jobs._get_job_duration(job_id)


class _DefaultEvidence:
    def collect(self, job_id: str, metrics_dir: Path) -> _MetricsEvidence:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        system_metrics = jobs._fetch_system_metrics(job_id)
        pipeline_stats_log = jobs._fetch_pipeline_stats(job_id, str(metrics_dir))
        metrics_summary = jobs._collect_metrics_summary(metrics_dir)
        return _MetricsEvidence(
            system_metrics=system_metrics,
            pipeline_stats_log=pipeline_stats_log,
            metrics_summary=metrics_summary,
            error_summary=_summarize_error(
                "\n".join((system_metrics, pipeline_stats_log, metrics_summary))
            ),
        )


class _LocalExecution:
    """Run experiment jobs as local subprocesses."""

    def __init__(
        self,
        mode: str = "full",
        dataloader_command: str | None = None,
    ) -> None:
        self._root: Path | None = None
        self._mode = mode
        self._dataloader_command = dataloader_command

    def bind_workdir(self, workdir: Path) -> _LocalExecution:
        self._root = workdir / "local_jobs"
        return self

    def launch(self, command: str, workdir: Path) -> str | None:
        self.bind_workdir(workdir)
        job_id = f"local_{int(time.time() * 1000)}"
        local_dir = workdir / "local_jobs" / job_id
        local_dir.mkdir(parents=True, exist_ok=True)

        if self._mode == "dry_run":
            _write_json(
                local_dir / "job.json",
                {
                    "job_id": job_id,
                    "command": command,
                    "started_at": time.time(),
                    "ended_at": time.time(),
                    "returncode": 0,
                    "mode": self._mode,
                },
            )
            (local_dir / "stdout.log").write_text(
                f"[autoresearch] dry_run command={command}\n"
            )
            (local_dir / "stderr.log").write_text("")
            (local_dir / "returncode.txt").write_text("0\n")
            return job_id

        run_command = (
            self._dataloader_command
            if self._mode == "dataloader_only" and self._dataloader_command
            else command
        )
        stdout = (local_dir / "stdout.log").open("w")
        stderr = (local_dir / "stderr.log").open("w")
        rc_file = local_dir / "returncode.txt"
        wrapped_command = (
            f"{run_command}; "
            f"rc=$?; "
            f"printf '%s\\n' \"$rc\" > {shlex.quote(str(rc_file))}; "
            "exit $rc"
        )
        process = subprocess.Popen(
            wrapped_command,
            shell=True,
            cwd=str(workdir),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        _write_json(
            local_dir / "job.json",
            {
                "job_id": job_id,
                "command": run_command,
                "requested_command": command,
                "pid": process.pid,
                "started_at": time.time(),
                "returncode": None,
                "mode": self._mode,
            },
        )
        return job_id

    def status(self, job_id: str) -> str:
        data = self._load(job_id)
        if not data:
            return "UNKNOWN"
        rc_file = self._job_path(job_id).parent / "returncode.txt"
        if rc_file.exists():
            try:
                data["returncode"] = int(rc_file.read_text().strip())
                data.setdefault("ended_at", time.time())
                _write_json(self._job_path(job_id), data)
            except ValueError:
                _LG.warning("Invalid local return code in %s", rc_file)
        returncode = data.get("returncode")
        if isinstance(returncode, int):
            return "SUCCEEDED" if returncode == 0 else "FAILED"
        pid = int(data["pid"])
        if _process_alive(pid):
            return "RUNNING"
        data["returncode"] = 1
        data["ended_at"] = time.time()
        _write_json(self._job_path(job_id), data)
        return "FAILED"

    def progress(self, job_id: str) -> str | None:
        for name in ("stdout.log", "stderr.log"):
            path = self._job_path(job_id).parent / name
            if not path.exists():
                continue
            for line in reversed(path.read_text(errors="replace").splitlines()):
                if "[autoresearch]" in line:
                    return line.strip()
        return None

    def cancel(self, job_id: str) -> bool:
        data = self._load(job_id)
        if not data:
            return False
        if isinstance(data.get("returncode"), int):
            return True
        pid = int(data["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            if _process_alive(pid):
                os.kill(pid, signal.SIGKILL)
            data["returncode"] = data.get("returncode", -signal.SIGTERM)
            data["ended_at"] = time.time()
            _write_json(self._job_path(job_id), data)
            return True
        except ProcessLookupError:
            return True

    def duration(self, job_id: str) -> int | None:
        data = self._load(job_id)
        if not data:
            return None
        started = data.get("started_at")
        ended = data.get("ended_at", time.time())
        if not isinstance(started, (int, float)) or not isinstance(ended, (int, float)):
            return None
        return int(float(ended) - float(started))

    def _job_path(self, job_id: str) -> Path:
        if self._root is None:
            raise RuntimeError("_LocalExecution has no workdir root")
        return self._root / job_id / "job.json"

    def _load(self, job_id: str) -> dict | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())


class _LocalEvidence:
    def collect(self, job_id: str, metrics_dir: Path) -> _MetricsEvidence:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        local_dir = metrics_dir.parents[2] / "local_jobs" / job_id
        logs = []
        log_paths = []
        for name in ("stdout.log", "stderr.log"):
            path = local_dir / name
            if path.exists():
                log_paths.append(str(path))
                logs.append(f"## {name}\n{path.read_text(errors='replace')[-4000:]}")
        local_log = "\n\n".join(logs) if logs else "(no local job logs found)"
        metrics_summary = jobs._collect_metrics_summary(metrics_dir)
        exit_code = _read_local_exit_code(local_dir)
        return _MetricsEvidence(
            system_metrics="(system metrics unavailable for this local platform)",
            pipeline_stats_log=local_log,
            metrics_summary=metrics_summary,
            log_paths=log_paths,
            exit_code=exit_code,
            error_summary=_summarize_error(local_log),
            progress_seen="[autoresearch]" in local_log,
        )


def _summarize_error(text: str) -> str | None:
    traceback = _extract_traceback(text)
    if traceback:
        return traceback
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(
            marker in lower
            for marker in (
                "error",
                "exception",
                "traceback",
                "failed",
                "cannot pickle",
                "can't pickle",
                "out of memory",
            )
        ):
            return stripped[-1000:]
    return None


def _extract_traceback(text: str) -> str | None:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            start = index
    if start is None:
        return None
    block = []
    for line in lines[start:]:
        block.append(line)
        if len(block) >= 40:
            break
        stripped = line.strip()
        if block and ":" in stripped and not line.startswith((" ", "\t")):
            if not stripped.startswith("Traceback"):
                break
    return "\n".join(block)[-2000:]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _read_local_exit_code(local_dir: Path) -> int | None:
    path = local_dir / "returncode.txt"
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
