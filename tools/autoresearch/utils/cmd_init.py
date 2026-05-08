# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from .log import setup_logging
from .scm import current_commit, detect_scm
from .state import MASTER_TABLE_HEADERS, write_state

_LG: logging.Logger = logging.getLogger(__name__)


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a new autoresearch experiment workdir.",
    )
    parser.add_argument("workdir", help="Working directory for the experiment")
    parser.add_argument("--pipeline-script", help="Pipeline script to optimize")
    parser.add_argument("--build-command", help="Command to build the job image")
    parser.add_argument(
        "--base-launch-command",
        help="Base torchx launch command (use $IMAGE for image placeholder)",
    )
    parser.add_argument(
        "--notes",
        help="Free-form notes about the experiment (included in prompts)",
    )
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after N consecutive iterations with no improvement (default: 3)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between job status polls (default: 120)",
    )
    parser.add_argument(
        "--source-dir",
        help="Source directory containing the pipeline code to modify in-place",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of concurrent training jobs (default: 3)",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=1800,
        help="Wall-clock timeout per job in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Pass --dangerously-skip-permissions to claude invocations",
    )
    return parser.parse_args(args)


def run(args: list[str]) -> None:
    ns = _parse_args(args)
    workdir = Path(ns.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "runs").mkdir(exist_ok=True)
    (workdir / "logs").mkdir(exist_ok=True)
    setup_logging(workdir)
    _LG.info("Initializing experiment at %s", workdir)

    source_dir = os.path.abspath(ns.source_dir) if ns.source_dir else ""
    scm_type = ""
    anchor_commit = ""
    if source_dir:
        scm_type = detect_scm(source_dir)
        anchor_commit = current_commit(scm_type, source_dir)
        _LG.info("SCM: %s, anchor commit: %s", scm_type, anchor_commit[:12])

    config = {
        "created_at": datetime.now().isoformat(),
        "pipeline_script": (
            os.path.abspath(ns.pipeline_script) if ns.pipeline_script else None
        ),
        "source_dir": source_dir,
        "scm": scm_type,
        "build_command": ns.build_command or "",
        "base_launch_command": ns.base_launch_command or "",
        "notes": ns.notes or "",
        "stopping_criteria": {
            "max_iterations": ns.max_iterations,
            "patience": ns.patience,
        },
        "max_concurrency": ns.max_concurrency,
        "job_timeout_s": ns.job_timeout,
        "poll_interval": ns.poll_interval,
        "claude_flags": (
            ["--dangerously-skip-permissions"]
            if ns.dangerously_skip_permissions
            else []
        ),
    }

    (workdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    state = {
        "iteration": 0,
        "status": "initialized",
        "baseline_job": None,
        "current_best": None,
        "best_metric": None,
        "plateau_count": 0,
        "best_practices_tried": [],
        "cached_image": None,
        "anchor_commit": anchor_commit,
        "history": [],
    }
    write_state(workdir, state)

    (workdir / "master_table.tsv").write_text("\t".join(MASTER_TABLE_HEADERS) + "\n")

    if ns.pipeline_script and os.path.exists(ns.pipeline_script):
        shutil.copy2(ns.pipeline_script, workdir / "pipeline.py")

    print(f"Initialized experiment at {workdir}")
    print(f"Next: python launch.py assess {workdir} --baseline-job <JOB_NAME>")
