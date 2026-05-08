# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from . import fetch_pipeline_stats, fetch_system_metrics  # noqa: E402
from .claude import extract_json_block, load_knowledge, load_prompt, run_claude
from .jobs import collect_metrics_summary
from .log import setup_logging
from .scm import commit as scm_commit, has_pending_changes
from .state import append_master_row, read_config, read_state, write_state

_LG: logging.Logger = logging.getLogger(__name__)


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess baseline job performance.",
    )
    parser.add_argument("workdir")
    parser.add_argument(
        "--baseline-job", required=True, help="Job name for baseline run"
    )
    parser.add_argument(
        "--skip-instrument",
        action="store_true",
        help="Skip automatic instrumentation of the pipeline script",
    )
    return parser.parse_args(args)


def _extract_code_block(text: str) -> str | None:
    """Extract the last Python code block from Claude's response."""
    matches = list(re.finditer(r"```python\s*\n(.*?)\n```", text, re.DOTALL))
    if matches:
        return matches[-1].group(1)
    return None


def _instrument_pipeline(
    workdir: Path, config: dict, pipeline_script: str, pipeline_code: str
) -> None:
    """Ask Claude to add TTFB/step-time instrumentation to the pipeline script."""
    _LG.info("Instrumenting %s with TTFB/step-time logging", pipeline_script)
    print("Instrumenting pipeline with TTFB/step-time logging...")

    prompt = load_prompt(
        "instrument",
        PIPELINE_SCRIPT=pipeline_script,
        PIPELINE_CODE=pipeline_code,
    )

    output = run_claude(prompt, workdir, "instrument")
    modified_code = _extract_code_block(output)

    if not modified_code:
        _LG.warning("Could not extract instrumented code from Claude output")
        print("  Warning: instrumentation failed — no code block in response")
        return

    target = Path(pipeline_script)
    target.write_text(modified_code)
    _LG.info("Wrote instrumented code to %s", target)
    print(f"  Wrote instrumented code to {target}")

    scm = config.get("scm", "")
    source_dir = config.get("source_dir", "")
    if scm and source_dir and has_pending_changes(scm, source_dir):
        commit_hash = scm_commit(
            scm, source_dir, "[autoresearch] Add TTFB/step-time instrumentation"
        )
        _LG.info("Committed instrumentation: %s", commit_hash[:12])
        print(f"  Committed: {commit_hash[:12]}")


def run(args: list[str]) -> None:
    ns = _parse_args(args)
    workdir = Path(ns.workdir).resolve()
    setup_logging(workdir)
    _LG.info("Assessing baseline job: %s", ns.baseline_job)

    config = read_config(workdir)
    state = read_state(workdir)
    job_id = ns.baseline_job

    print(f"Fetching metrics for baseline job: {job_id}")

    run_dir = workdir / "runs" / "000_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    system_metrics = fetch_system_metrics(job_id)
    (run_dir / "system_metrics.txt").write_text(system_metrics)

    pipeline_stats_log = fetch_pipeline_stats(job_id, str(metrics_dir))
    (run_dir / "pipeline_stats_log.txt").write_text(pipeline_stats_log)

    pipeline_script = config.get("pipeline_script", "")
    pipeline_code = ""
    if pipeline_script and Path(pipeline_script).exists():
        pipeline_code = Path(pipeline_script).read_text()
    elif (workdir / "pipeline.py").exists():
        pipeline_code = (workdir / "pipeline.py").read_text()

    metrics_content = collect_metrics_summary(metrics_dir)

    knowledge = load_knowledge()
    prompt = load_prompt(
        "assess",
        KNOWLEDGE=knowledge,
        JOB_ID=job_id,
        SYSTEM_METRICS=system_metrics,
        PIPELINE_STATS=metrics_content,
        PIPELINE_CODE=pipeline_code or "(no pipeline code provided)",
        BASE_LAUNCH_COMMAND=config.get("base_launch_command", ""),
        NOTES=config.get("notes", ""),
    )

    print("Running Claude assessment...")
    output = run_claude(prompt, workdir, "assess")

    (run_dir / "analysis.md").write_text(output)
    structured = extract_json_block(output)

    row: dict[str, str | float] = {
        "run_id": "000",
        "name": "baseline",
        "job_id": job_id,
        "status": "completed",
        "changes": "baseline",
    }
    if structured and "metrics" in structured:
        m = structured["metrics"]
        row["step_time_ms"] = m.get("step_time_ms", "")
        row["ttfb_s"] = m.get("ttfb_s", "")
        row["sm_util_pct"] = m.get("sm_utilization_pct", "")
        row["data_readiness_pct"] = m.get("data_readiness_pct", "")
    append_master_row(workdir, row)

    state["baseline_job"] = job_id
    state["status"] = "assessed"
    state["current_best"] = "000_baseline"
    state["history"].append(
        {
            "iteration": 0,
            "run_id": "000",
            "name": "baseline",
            "job_id": job_id,
            "structured": structured,
        }
    )
    write_state(workdir, state)

    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)
    print(output)

    if not ns.skip_instrument and pipeline_script and pipeline_code:
        _instrument_pipeline(workdir, config, pipeline_script, pipeline_code)

    print(f"\nResults saved to {run_dir}")
    print(f"Next: python launch.py loop {workdir}")
