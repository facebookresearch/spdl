#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone autoresearch engine runner.

Single entry point for the full autoresearch workflow: initializes the
workdir if needed, instruments the pipeline, then runs the async engine.
Designed to be invoked by Claude Code or directly from the command line.

First run::

    fbpython run.py <workdir> \\
      --pipeline-script path/to/pipeline.py \\
      --source-dir path/to/source \\
      --build-command "docker build -t my_image ." \\
      --base-launch-command "torchx run ... --image \\$IMAGE ..." \\
      --platform auto \\
      --agent claude

Resume after Ctrl+C::

    fbpython run.py <workdir>

All config is persisted in ``<workdir>/config.json`` after the first run,
so resuming only needs the workdir path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from spdl.tools.autoresearch.utils.log import setup_logging
from spdl.tools.autoresearch.utils.platform import (
    AutoresearchPlatform,
    create_platform,
)
from spdl.tools.autoresearch.utils.runner import AsyncWorkEngine
from spdl.tools.autoresearch.utils.state import (
    MASTER_TABLE_HEADERS,
    read_config,
    read_state,
    SCHEMA_VERSION,
    write_state,
)
from spdl.tools.autoresearch.utils.types import FailureKind, FailurePhase, FailureRecord
from spdl.tools.autoresearch.utils.workflow import AutoresearchAdapter
from spdl.tools.autoresearch.utils.workflow.failures import _make_failure

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = ["main"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("workdir", help="Working directory for experiment state")

    parser.add_argument("--pipeline-script", help="Pipeline script to optimize")
    parser.add_argument("--build-command", help="Command to build the job image")
    parser.add_argument(
        "--base-launch-command",
        help="Base torchx launch command (use $IMAGE for image placeholder)",
    )
    parser.add_argument(
        "--source-dir",
        help="Source directory containing the pipeline code to modify in-place",
    )
    parser.add_argument("--notes", help="Free-form notes about the experiment")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--job-timeout", type=int, default=1800)
    parser.add_argument(
        "--platform",
        default="auto",
        help=(
            "Job execution platform provider. 'auto' uses the best available "
            "provider discovered in this environment."
        ),
    )
    parser.add_argument(
        "--agent",
        choices=("claude", "codex"),
        default="claude",
        help="Coding agent used for source changes, analysis, and planning.",
    )
    parser.add_argument(
        "--local-execution-mode",
        choices=("full", "dataloader_only", "dry_run"),
        default="full",
        help="How local platform launches experiment commands.",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Pass --dangerously-skip-permissions to claude invocations",
    )
    parser.add_argument(
        "--skip-instrument",
        action="store_true",
        help="Skip automatic TTFB/step-time instrumentation",
    )
    return parser.parse_args()


def _record_setup_failure(
    workdir: Path,
    state: dict,
    failure: FailureRecord,
) -> None:
    state.setdefault("_setup_failures", []).append(failure.to_dict())
    engine_dir = workdir / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    setup_failures = engine_dir / "setup_failures.json"
    setup_failures.write_text(json.dumps(state["_setup_failures"], indent=2) + "\n")
    write_state(workdir, state)


def _init_workdir(
    ns: argparse.Namespace,
    workdir: Path,
    platform: AutoresearchPlatform,
) -> dict:
    """Initialize the workdir with config and state. Returns config dict."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "runs").mkdir(exist_ok=True)
    (workdir / "logs").mkdir(exist_ok=True)

    source_dir = os.path.abspath(ns.source_dir) if ns.source_dir else ""
    scm_type = ""
    anchor_commit_hash = ""
    setup_failures = []
    if source_dir:
        try:
            scm_type = platform.workspace.detect(source_dir)
            anchor_commit_hash = platform.workspace.current(scm_type, source_dir)
            _LG.info("SCM: %s, anchor commit: %s", scm_type, anchor_commit_hash[:12])
        except Exception as error:
            failure = _make_failure(
                FailureKind.SETUP_SOURCE_FAILED,
                FailurePhase.SETUP,
                "Failed to inspect source control state during setup",
                exception=error,
            )
            setup_failures.append(failure.to_dict())
            _LG.warning("Continuing without source control anchor: %s", error)

    config = {
        "schema_version": SCHEMA_VERSION,
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
        "startup_failure_retries": 2,
        "startup_retryable_experiments": ["mtp"],
        "platform": ns.platform,
        "agent": ns.agent,
        "local_execution_mode": ns.local_execution_mode,
        "claude_flags": (
            ["--dangerously-skip-permissions"]
            if ns.dangerously_skip_permissions
            else []
        ),
    }

    (workdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    state = {
        "schema_version": SCHEMA_VERSION,
        "iteration": 0,
        "status": "initialized",
        "baseline_job": None,
        "current_best": None,
        "best_metric": None,
        "plateau_count": 0,
        "best_practices_tried": [],
        "anchor_commit": anchor_commit_hash,
        "history": [],
        "_setup_failures": setup_failures,
    }
    write_state(workdir, state)
    if setup_failures:
        engine_dir = workdir / "engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        (engine_dir / "setup_failures.json").write_text(
            json.dumps(setup_failures, indent=2) + "\n"
        )

    (workdir / "master_table.tsv").write_text("\t".join(MASTER_TABLE_HEADERS) + "\n")

    if ns.pipeline_script and os.path.exists(ns.pipeline_script):
        shutil.copy2(ns.pipeline_script, workdir / "pipeline.py")

    print(f"Initialized experiment at {workdir}")
    return config


def _instrument_pipeline(
    workdir: Path,
    config: dict,
    platform: AutoresearchPlatform,
) -> None:
    """Add TTFB/step-time instrumentation to the pipeline script."""
    pipeline_script = config.get("pipeline_script", "")
    if not pipeline_script or not Path(pipeline_script).exists():
        return

    pipeline_code = Path(pipeline_script).read_text()
    _LG.info("Instrumenting %s with TTFB/step-time logging", pipeline_script)
    print("Instrumenting pipeline with TTFB/step-time logging...")

    prompt = platform.agent._load_prompt(
        "instrument",
        PIPELINE_SCRIPT=pipeline_script,
        PIPELINE_CODE=pipeline_code,
    )

    try:
        output = platform.agent.run(prompt, workdir, "instrument")
    except Exception as error:
        _record_setup_failure(
            workdir,
            read_state(workdir),
            _make_failure(
                FailureKind.SETUP_INSTRUMENTATION_FAILED,
                FailurePhase.INSTRUMENTATION,
                "Coding agent failed while instrumenting the pipeline",
                exception=error,
            ),
        )
        return

    matches = list(re.finditer(r"```python\s*\n(.*?)\n```", output, re.DOTALL))
    modified_code = matches[-1].group(1) if matches else None

    if not modified_code:
        _LG.warning("First instrument attempt returned no code block, retrying")
        retry_prompt = (
            prompt + "\n\n**You MUST output the ENTIRE modified file in a single "
            "```python code block. Do NOT describe changes in prose. "
            "Output ONLY code.**\n"
        )
        try:
            output = platform.agent.run(retry_prompt, workdir, "instrument_retry")
        except Exception as error:
            _record_setup_failure(
                workdir,
                read_state(workdir),
                _make_failure(
                    FailureKind.SETUP_INSTRUMENTATION_FAILED,
                    FailurePhase.INSTRUMENTATION,
                    "Coding agent failed while retrying pipeline instrumentation",
                    exception=error,
                ),
            )
            return
        matches = list(re.finditer(r"```python\s*\n(.*?)\n```", output, re.DOTALL))
        modified_code = matches[-1].group(1) if matches else None

    if not modified_code:
        _LG.warning("Instrumentation failed — no code block in response")
        print("  Warning: instrumentation failed")
        _record_setup_failure(
            workdir,
            read_state(workdir),
            _make_failure(
                FailureKind.SETUP_INSTRUMENTATION_FAILED,
                FailurePhase.INSTRUMENTATION,
                "Coding agent did not return a Python code block for instrumentation",
            ),
        )
        return

    Path(pipeline_script).write_text(modified_code)
    _LG.info("Wrote instrumented code to %s", pipeline_script)
    print(f"  Wrote instrumented code to {pipeline_script}")

    scm = config.get("scm", "")
    source_dir = config.get("source_dir", "")
    if scm and source_dir and platform.workspace.has_changes(scm, source_dir):
        commit_hash = platform.workspace.commit(
            scm, source_dir, "[autoresearch] Add TTFB/step-time instrumentation"
        )
        _LG.info("Committed instrumentation: %s", commit_hash[:12])
        print(f"  Committed: {commit_hash[:12]}")


def main() -> None:
    ns = _parse_args()
    workdir = Path(ns.workdir).resolve()

    config_file = workdir / "config.json"
    is_fresh = not config_file.exists()

    platform_kind = ns.platform
    source_dir = ns.source_dir or ""
    if not is_fresh:
        existing_config = read_config(workdir)
        platform_kind = str(existing_config.get("platform", "auto"))
        ns.agent = str(existing_config.get("agent", ns.agent))
        ns.local_execution_mode = str(
            existing_config.get("local_execution_mode", ns.local_execution_mode)
        )
        source_dir = str(existing_config.get("source_dir", source_dir))
    platform_config = {
        "platform": platform_kind,
        "agent": ns.agent,
        "local_execution_mode": ns.local_execution_mode,
        "source_dir": source_dir,
    }
    platform = create_platform(platform_config, workdir)

    if is_fresh:
        if not ns.base_launch_command:
            print(
                "Error: first run requires --base-launch-command",
                file=sys.stderr,
            )
            sys.exit(1)
        config = _init_workdir(ns, workdir, platform)
    else:
        config = read_config(workdir)

    setup_logging(workdir)
    _LG.info("Starting autoresearch engine at %s", workdir)

    state = read_state(workdir)

    if is_fresh and not ns.skip_instrument:
        _instrument_pipeline(workdir, config, platform)
        scm_type = config.get("scm", "")
        source_dir = config.get("source_dir", "")
        if scm_type and source_dir:
            state["anchor_commit"] = platform.workspace.current(scm_type, source_dir)
            write_state(workdir, state)

    if state["status"] == "initialized":
        state["status"] = "assessed"
        write_state(workdir, state)

    state["status"] = "looping"
    write_state(workdir, state)

    adapter = AutoresearchAdapter(
        workdir=workdir,
        config=config,
        state=state,
        platform=platform,
    )
    engine = AsyncWorkEngine(
        adapter=adapter,
        max_concurrency=config.get("max_concurrency", 3),
    )

    checkpoint = workdir / "engine" / "checkpoint.json"
    if checkpoint.exists():
        _LG.info("Resuming from persisted engine state")
        print("Resuming from previous engine state...")
    else:
        print("Starting engine...")
    asyncio.run(engine.run())

    print(f"\nEngine finished. Results in {workdir}")


if __name__ == "__main__":
    main()
