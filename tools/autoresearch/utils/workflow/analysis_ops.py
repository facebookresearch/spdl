# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metrics analysis and result recording operations for autoresearch."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from spdl.tools.autoresearch.plot_progress import (
    _load_tsv,
    _plot_hypothesis_tree,
    _plot_progress,
)

from ..platform import AutoresearchPlatform
from ..platform.agents import _parse_agent_result
from ..state import _append_master_row, _read_master_table, write_state
from ..types import (
    _AnalysisResult,
    _AutoresearchError,
    _HypothesisNode,
    FailureKind,
    FailurePhase,
)
from .common import (
    _compare_value,
    _current_best_metric,
    _is_headspace_entry,
    _read_pipeline_code,
)
from .failures import _classify_terminal_job_failure, _failure_note, _make_failure
from .policy import _change_summary_for_spec

_LG: logging.Logger = logging.getLogger(__name__)

_STRUCTURAL_PRACTICES = {"mtp"}
_STRUCTURAL_ATTEMPT_THRESHOLD = 3


def _analyze_job(
    workdir: Path,
    config: dict,
    state: dict,
    platform: AutoresearchPlatform,
    node: object,
    status: str,
    progress_seen: bool = False,
) -> object:
    """Analyze a completed job. Returns _AnalysisResult."""
    assert isinstance(node, _HypothesisNode)

    knowledge = platform.agent._load_knowledge()
    pipeline_code = _read_pipeline_code(config, workdir)
    exp = node.spec
    run_id = node.node_id

    run_dir = workdir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    job_id = node.job_id or ""
    print(f"\nCollecting metrics for {node.name} ({job_id})...")

    try:
        evidence = platform.evidence.collect(job_id, metrics_dir)
    except Exception as error:
        raise _AutoresearchError(
            _make_failure(
                FailureKind.METRICS_COLLECTION_FAILED,
                FailurePhase.METRICS,
                "Failed to collect job metrics",
                exception=error,
                job_id=job_id,
            )
        ) from error
    (run_dir / "system_metrics.txt").write_text(evidence.system_metrics)
    (run_dir / "pipeline_stats_log.txt").write_text(evidence.pipeline_stats_log)

    prompt = platform.agent._load_prompt(
        "analyze",
        KNOWLEDGE=knowledge,
        JOB_ID=job_id,
        RUN_NAME=node.name,
        CHANGES=exp.get("description", ""),
        HYPOTHESIS=exp.get("hypothesis", ""),
        SYSTEM_METRICS=evidence.system_metrics,
        PIPELINE_STATS=evidence.metrics_summary,
        MASTER_TABLE=_read_master_table(workdir),
        PIPELINE_CODE=pipeline_code or "(not provided)",
    )

    print(f"Running agent analysis for {node.name}...")
    try:
        analysis = platform.agent.run(prompt, workdir, f"analyze_{run_id}")
    except Exception as error:
        raise _AutoresearchError(
            _make_failure(
                FailureKind.ANALYSIS_FAILED,
                FailurePhase.ANALYSIS,
                "Coding-agent analysis failed",
                exception=error,
                job_id=job_id,
            )
        ) from error
    (run_dir / "analysis.md").write_text(analysis)

    structured = _parse_agent_result(platform.agent, analysis).json
    duration = platform.execution.duration(job_id) if job_id else None
    if duration is None:
        duration = node.duration

    terminal_failure = (
        _classify_terminal_job_failure(
            evidence, job_id=job_id, progress_seen=progress_seen
        )
        if status == "failed"
        else None
    )
    _append_master_result_row(
        workdir, node, status, duration, structured, terminal_failure
    )

    cur_type, cur_val = _compare_value(
        structured.get("metrics", {}) if structured else {}
    )
    best_type, best_val = _current_best_metric(state)
    improved = cur_type != "none" and (
        best_type == "none" or (cur_type == best_type and cur_val < best_val)
    )

    return _AnalysisResult(
        structured=structured,
        duration=float(duration)
        if isinstance(duration, (int, float)) and duration > 0
        else None,
        improved=improved,
        failure=terminal_failure,
    )


def _append_master_result_row(
    workdir: Path,
    node: _HypothesisNode,
    status: str,
    duration: object,
    structured: dict | None,
    terminal_failure,
) -> None:
    row: dict[str, str | float] = {
        "run_id": node.node_id,
        "name": node.name,
        "job_id": node.job_id or "",
        "status": status,
        "changes": node.spec.get("description", ""),
        "change_summary": _change_summary_for_spec(node.spec),
        "duration_s": duration or "",
    }
    if structured and "metrics" in structured:
        metrics = structured["metrics"]
        row["step_time_ms"] = metrics.get("step_time_ms", "")
        row["steady_step_time_ms"] = metrics.get("steady_step_time_ms", "")
        row["ttfb_s"] = metrics.get("ttfb_s", "")
        row["sm_util_pct"] = metrics.get("sm_utilization_pct", "")
        row["steady_sm_util_pct"] = metrics.get("steady_sm_utilization_pct", "")
        row["data_readiness_pct"] = metrics.get("data_readiness_pct", "")
        row["notes"] = metrics.get("notes", "")
    if terminal_failure is not None and not row.get("notes"):
        row["notes"] = _failure_note(terminal_failure)
    _append_master_row(workdir, row)


def _update_on_complete(
    workdir: Path,
    config: dict,
    state: dict,
    node: object,
    result: object,
) -> None:
    """Update autoresearch state after a node completes."""
    assert isinstance(node, _HypothesisNode)
    assert isinstance(result, _AnalysisResult)

    exp = node.spec
    best_type, best_val = _current_best_metric(state)
    structured = result.structured or {}
    failure = result.failure
    if failure is not None:
        structured = dict(structured)
        structured["failure"] = failure.to_dict()

    state["history"].append(
        {
            "iteration": state.get("iteration", 0),
            "run_id": node.node_id,
            "name": node.name,
            "job_id": node.job_id or "",
            "commit": node.commit or "",
            "completed_at": datetime.now().isoformat(),
            "structured": structured,
        }
    )

    cur_type, cur_val = _compare_value(
        result.structured.get("metrics", {}) if result.structured else {}
    )
    if (
        not _is_headspace_entry({"name": node.name})
        and cur_type != "none"
        and (best_type == "none" or (cur_type == best_type and cur_val < best_val))
    ):
        state["current_best"] = node.node_id

    _record_successful_structural_practices(state, [exp], result.structured)
    _append_findings(workdir, node.name, result.structured)
    write_state(workdir, state)


def _record_successful_structural_practices(
    state: dict,
    experiments: list[dict],
    result_structured: dict | None,
) -> None:
    tried = set(state.get("best_practices_tried", []))
    attempts: dict[str, int] = state.get("_structural_attempts", {})

    for exp in experiments:
        structural_tags = [
            tag
            for tag in exp.get("best_practices_tags", [])
            if tag in _STRUCTURAL_PRACTICES
        ]
        if not structural_tags:
            continue

        metrics = (result_structured or {}).get("metrics", {})
        dur = metrics.get("duration_s")
        sm = metrics.get("sm_utilization_pct")
        succeeded = (isinstance(dur, (int, float)) and dur > 0) or (
            isinstance(sm, (int, float)) and sm > 0
        )
        for tag in structural_tags:
            attempts[tag] = attempts.get(tag, 0) + 1
            if succeeded or attempts[tag] >= _STRUCTURAL_ATTEMPT_THRESHOLD:
                tried.add(tag)

    state["_structural_attempts"] = attempts
    state["best_practices_tried"] = sorted(tried)


def _append_findings(workdir: Path, name: str, structured: dict | None) -> None:
    if not structured:
        return
    findings = structured.get("findings", [])
    if not findings:
        return

    findings_file = workdir / "findings.md"
    lines = []
    if not findings_file.exists():
        lines.append("# Accumulated Findings\n\n")
    for fact in findings:
        lines.append(f"- [{name}] {fact}\n")
    with open(findings_file, "a") as f:
        f.writelines(lines)


def _update_summary_and_plot(workdir: Path, state: dict) -> None:
    """Regenerate summary.md, progress.png, and hypothesis_tree.png."""
    history = state.get("history", [])
    best = state.get("current_best", "N/A")
    best_metric = state.get("best_metric")
    plateau = state.get("plateau_count", 0)

    best_str = f"{best_metric:.0f}ms" if best_metric else "N/A"
    lines = [
        "# Autoresearch Progress\n",
        f"**Total experiments**: {len(history)}",
        f"**Current best**: {best}",
        f"**Best steady step time**: {best_str}",
        f"**Plateau count**: {plateau}\n",
        "## Recent Results\n",
    ]
    for entry in history[-5:]:
        metrics = (entry.get("structured") or {}).get("metrics", {})
        step = metrics.get("steady_step_time_ms") or metrics.get("step_time_ms")
        dur = metrics.get("duration_s")
        metric_parts = []
        if step is not None:
            metric_parts.append(f"step={step}ms")
        if dur is not None:
            metric_parts.append(f"dur={dur:.0f}s")
        name = entry.get("name", entry.get("run_id", "?"))
        metric_str = ", ".join(metric_parts) if metric_parts else "N/A"
        lines.append(f"- **{name}**: {metric_str}")
    (workdir / "summary.md").write_text("\n".join(lines) + "\n")

    try:
        tsv_path = workdir / "master_table.tsv"
        if tsv_path.exists():
            experiments = _load_tsv(tsv_path)
            if experiments:
                _plot_progress(experiments, str(workdir / "progress.png"))
    except Exception as error:
        _LG.warning("Failed to regenerate progress plot: %s", error)

    try:
        tree_file = workdir / "engine" / "tree.json"
        if tree_file.exists():
            _plot_hypothesis_tree(
                json.loads(tree_file.read_text()),
                str(workdir / "hypothesis_tree.png"),
            )
    except Exception as error:
        _LG.warning("Failed to regenerate hypothesis tree plot: %s", error)


__all__ = ["_analyze_job", "_update_on_complete", "_update_summary_and_plot"]
