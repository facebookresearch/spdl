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
      --build-command "fbpkg build ..." \\
      --base-launch-command "torchx run ... --image \\$IMAGE ..."

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

from spdl.tools.autoresearch.utils.callbacks import (
    analyze_job,
    build_initial_nodes,
    get_plan,
    launch_node,
    prepare_node,
    should_stop,
    update_on_complete,
    update_summary_and_plot,
)
from spdl.tools.autoresearch.utils.claude import load_prompt, run_claude
from spdl.tools.autoresearch.utils.engine import ExperimentEngine
from spdl.tools.autoresearch.utils.jobs import (
    _get_cancel_job,
    _get_check_job_progress,
    _get_check_job_status,
)
from spdl.tools.autoresearch.utils.log import setup_logging
from spdl.tools.autoresearch.utils.scm import (
    commit as scm_commit,
    current_commit,
    detect_scm,
    has_pending_changes,
)
from spdl.tools.autoresearch.utils.state import (
    MASTER_TABLE_HEADERS,
    read_config,
    read_state,
    write_state,
)

_LG: logging.Logger = logging.getLogger(__name__)


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


def _init_workdir(ns: argparse.Namespace, workdir: Path) -> dict:
    """Initialize the workdir with config and state. Returns config dict."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "runs").mkdir(exist_ok=True)
    (workdir / "logs").mkdir(exist_ok=True)

    source_dir = os.path.abspath(ns.source_dir) if ns.source_dir else ""
    scm_type = ""
    anchor_commit_hash = ""
    if source_dir:
        scm_type = detect_scm(source_dir)
        anchor_commit_hash = current_commit(scm_type, source_dir)
        _LG.info("SCM: %s, anchor commit: %s", scm_type, anchor_commit_hash[:12])

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
        "anchor_commit": anchor_commit_hash,
        "history": [],
    }
    write_state(workdir, state)

    (workdir / "master_table.tsv").write_text("\t".join(MASTER_TABLE_HEADERS) + "\n")

    if ns.pipeline_script and os.path.exists(ns.pipeline_script):
        shutil.copy2(ns.pipeline_script, workdir / "pipeline.py")

    print(f"Initialized experiment at {workdir}")
    return config


def _instrument_pipeline(workdir: Path, config: dict) -> None:
    """Add TTFB/step-time instrumentation to the pipeline script."""
    pipeline_script = config.get("pipeline_script", "")
    if not pipeline_script or not Path(pipeline_script).exists():
        return

    pipeline_code = Path(pipeline_script).read_text()
    _LG.info("Instrumenting %s with TTFB/step-time logging", pipeline_script)
    print("Instrumenting pipeline with TTFB/step-time logging...")

    prompt = load_prompt(
        "instrument",
        PIPELINE_SCRIPT=pipeline_script,
        PIPELINE_CODE=pipeline_code,
    )

    output = run_claude(prompt, workdir, "instrument")

    matches = list(re.finditer(r"```python\s*\n(.*?)\n```", output, re.DOTALL))
    modified_code = matches[-1].group(1) if matches else None

    if not modified_code:
        _LG.warning("First instrument attempt returned no code block, retrying")
        retry_prompt = (
            prompt + "\n\n**You MUST output the ENTIRE modified file in a single "
            "```python code block. Do NOT describe changes in prose. "
            "Output ONLY code.**\n"
        )
        output = run_claude(retry_prompt, workdir, "instrument_retry")
        matches = list(re.finditer(r"```python\s*\n(.*?)\n```", output, re.DOTALL))
        modified_code = matches[-1].group(1) if matches else None

    if not modified_code:
        _LG.warning("Instrumentation failed — no code block in response")
        print("  Warning: instrumentation failed")
        return

    Path(pipeline_script).write_text(modified_code)
    _LG.info("Wrote instrumented code to %s", pipeline_script)
    print(f"  Wrote instrumented code to {pipeline_script}")

    scm = config.get("scm", "")
    source_dir = config.get("source_dir", "")
    if scm and source_dir and has_pending_changes(scm, source_dir):
        commit_hash = scm_commit(
            scm, source_dir, "[autoresearch] Add TTFB/step-time instrumentation"
        )
        _LG.info("Committed instrumentation: %s", commit_hash[:12])
        print(f"  Committed: {commit_hash[:12]}")


def main() -> None:
    ns = _parse_args()
    workdir = Path(ns.workdir).resolve()

    config_file = workdir / "config.json"
    is_fresh = not config_file.exists()

    if is_fresh:
        if not ns.base_launch_command:
            print(
                "Error: first run requires --base-launch-command",
                file=sys.stderr,
            )
            sys.exit(1)
        config = _init_workdir(ns, workdir)
    else:
        config = read_config(workdir)

    setup_logging(workdir)
    _LG.info("Starting autoresearch engine at %s", workdir)

    state = read_state(workdir)

    if is_fresh and not ns.skip_instrument:
        _instrument_pipeline(workdir, config)

    if state["status"] == "initialized":
        state["status"] = "assessed"
        write_state(workdir, state)

    state["status"] = "looping"
    write_state(workdir, state)

    check_fn_sync = _get_check_job_status()
    cancel_fn_sync = _get_cancel_job()
    progress_fn_sync = _get_check_job_progress()

    async def launch(node):
        return await asyncio.to_thread(launch_node, config, state, node, workdir)

    async def check(job_id):
        raw = await asyncio.to_thread(check_fn_sync, job_id)
        if raw in ("SUCCEEDED", "COMPLETE"):
            return "completed"
        if raw == "FAILED":
            return "failed"
        return "running"

    async def cancel(job_id):
        return await asyncio.to_thread(cancel_fn_sync, job_id)

    async def analyze(node, status):
        return await asyncio.to_thread(
            analyze_job, workdir, config, state, node, status
        )

    async def plan(parent, result, tree):
        return await asyncio.to_thread(
            get_plan, workdir, config, state, parent, result, tree
        )

    async def prepare(node, parent):
        return await asyncio.to_thread(
            prepare_node, workdir, config, state, node, parent
        )

    async def on_complete(node, result):
        await asyncio.to_thread(
            update_on_complete, workdir, config, state, node, result
        )
        await asyncio.to_thread(update_summary_and_plot, workdir, state)

    async def stop(tree):
        return should_stop(config, state, tree)

    async def progress(job_id):
        return await asyncio.to_thread(progress_fn_sync, job_id)

    initial_nodes = build_initial_nodes(workdir, config, state)

    engine = ExperimentEngine(
        work_dir=str(workdir),
        max_concurrency=config.get("max_concurrency", 3),
        poll_interval=config.get("poll_interval", 120),
        job_timeout_s=config.get("job_timeout_s", 1800),
        launch_fn=launch,
        check_fn=check,
        cancel_fn=cancel,
        analyze_fn=analyze,
        plan_fn=plan,
        prepare_fn=prepare,
        should_stop_fn=stop,
        on_node_complete=on_complete,
        progress_fn=progress,
    )

    engine_tree = workdir / "engine" / "tree.json"
    if engine_tree.exists():
        _LG.info("Resuming from persisted engine state")
        print("Resuming from previous engine state...")
        asyncio.run(_resume_and_run(engine))
    else:
        print(f"Starting engine with {len(initial_nodes)} initial node(s)...")
        asyncio.run(engine.run(initial_nodes))

    print(f"\nEngine finished. Results in {workdir}")


async def _resume_and_run(engine: ExperimentEngine) -> None:
    """Load persisted state and resume the engine."""
    await engine.load_state()
    await engine.run()


if __name__ == "__main__":
    main()
