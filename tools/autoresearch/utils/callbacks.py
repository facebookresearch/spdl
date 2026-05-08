# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autoresearch callback implementations for the ExperimentEngine.

These functions are wired as callbacks by run.py to provide
autoresearch-specific behavior (Claude analysis, code changes,
job launching, etc.) to the generic engine.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from spdl.tools.autoresearch.plot_progress import (
    load_tsv,
    plot_hypothesis_tree,
    plot_progress,
)

from . import (
    apply_lint,
    fetch_pipeline_stats,
    fetch_system_metrics,
    get_job_duration,
    launch_job,
)
from .claude import (
    extract_json_block,
    load_knowledge,
    load_prompt,
    run_claude,
)
from .engine import AnalysisResult, HypothesisNode
from .jobs import build_image, collect_metrics_summary
from .scm import (
    commit as scm_commit,
    goto as scm_goto,
    has_pending_changes,
)
from .state import (
    append_master_row,
    read_master_table,
    write_state,
)

_LG: logging.Logger = logging.getLogger(__name__)

BEST_PRACTICES = [
    "subprocess_mtp",
    "batch_size_tuning",
    "concurrency_tuning",
]

STRUCTURAL_PRACTICES = {"subprocess_mtp"}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _read_pipeline_code(config: dict, workdir: Path) -> str:
    pipeline_script = config.get("pipeline_script", "")
    if pipeline_script and Path(pipeline_script).exists():
        return Path(pipeline_script).read_text()
    if (workdir / "pipeline.py").exists():
        return (workdir / "pipeline.py").read_text()
    return ""


def _get_compare_val(metrics: dict) -> tuple[str, float]:
    """Extract the comparison value from metrics. Returns (metric_type, value).

    Only compares values of the same type to avoid scale mixing
    (ms vs seconds).
    """
    step = metrics.get("steady_step_time_ms")
    if isinstance(step, (int, float)) and step > 0:
        return ("step_ms", float(step))
    dur = metrics.get("duration_s")
    if isinstance(dur, (int, float)) and dur > 0:
        return ("duration_s", float(dur))
    return ("none", float("inf"))


def _current_best_metric(state: dict) -> tuple[str, float]:
    """Return the best (lowest) metric observed so far.

    Returns (metric_type, value). Only compares values of the same
    type — step_ms values are compared with step_ms, duration_s with
    duration_s.
    """
    best_step = float("inf")
    best_dur = float("inf")
    for entry in state.get("history", []):
        s = entry.get("structured") or {}
        m = s.get("metrics", {})
        step = m.get("steady_step_time_ms")
        if isinstance(step, (int, float)) and step > 0:
            best_step = min(best_step, float(step))
        dur = m.get("duration_s")
        if isinstance(dur, (int, float)) and dur > 0:
            best_dur = min(best_dur, float(dur))

    if best_step < float("inf"):
        return ("step_ms", best_step)
    if best_dur < float("inf"):
        return ("duration_s", best_dur)
    return ("none", float("inf"))


def _untried_best_practices(state: dict) -> list[str]:
    tried = set(state.get("best_practices_tried", []))
    return [bp for bp in BEST_PRACTICES if bp not in tried]


def _update_plateau(state: dict, iteration_best_duration: float) -> None:
    prev_best = state.get("best_metric") or float("inf")
    if (
        isinstance(iteration_best_duration, (int, float))
        and iteration_best_duration < prev_best
    ):
        state["best_metric"] = iteration_best_duration
        state["plateau_count"] = 0
    else:
        state["plateau_count"] = state.get("plateau_count", 0) + 1


def _record_best_practices(state: dict, plan: dict) -> None:
    tried = set(state.get("best_practices_tried", []))
    for exp in plan.get("experiments", []):
        for tag in exp.get("best_practices_tags", []):
            if tag not in STRUCTURAL_PRACTICES:
                tried.add(tag)
    state["best_practices_tried"] = sorted(tried)


def _record_successful_structural_practices(
    state: dict,
    experiments: list[dict],
    results: dict[str, dict],
) -> None:
    tried = set(state.get("best_practices_tried", []))
    for exp in experiments:
        tags = exp.get("best_practices_tags", [])
        structural_tags = [t for t in tags if t in STRUCTURAL_PRACTICES]
        if not structural_tags:
            continue
        job_id = exp.get("_job_id", "")
        result = results.get(job_id, {})
        structured = result.get("structured") or {}
        metrics = structured.get("metrics", {})
        dur = metrics.get("duration_s")
        sm = metrics.get("sm_utilization_pct")
        succeeded = (isinstance(dur, (int, float)) and dur > 0) or (
            isinstance(sm, (int, float)) and sm > 0
        )
        if succeeded:
            for tag in structural_tags:
                tried.add(tag)
    state["best_practices_tried"] = sorted(tried)


def _extract_code_block(text: str) -> str | None:
    for pattern in [
        r"```python\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
    ]:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            candidate = matches[-1].group(1)
            if "import " in candidate or "def " in candidate or "class " in candidate:
                return candidate
    return None


def _apply_code_changes(
    workdir: Path,
    config: dict,
    state: dict,
    exp: dict,
    run_id: str,
    knowledge: str,
    pipeline_code: str,
) -> bool:
    pipeline_script = config.get("pipeline_script", "")
    if not pipeline_script or not pipeline_code:
        _LG.warning("No pipeline script configured; cannot apply code changes")
        return False

    _LG.info(
        "Applying code changes for %s: %s",
        run_id,
        exp.get("description", ""),
    )
    print(f"  Applying code changes for {exp['name']}...")

    prompt = load_prompt(
        "apply_changes",
        KNOWLEDGE=knowledge,
        EXPERIMENT_NAME=exp.get("name", run_id),
        EXPERIMENT_DESCRIPTION=exp.get("description", ""),
        EXPERIMENT_HYPOTHESIS=exp.get("hypothesis", ""),
        PIPELINE_SCRIPT=pipeline_script,
        PIPELINE_CODE=pipeline_code,
    )

    output = run_claude(prompt, workdir, f"apply_{run_id}")
    modified_code = _extract_code_block(output)

    if not modified_code:
        _LG.warning(
            "First apply attempt returned no code block for %s, retrying",
            run_id,
        )
        retry_prompt = (
            prompt + "\n\n**You MUST output the ENTIRE modified file in a single "
            "```python code block. Do NOT describe changes in prose.**\n"
        )
        output = run_claude(retry_prompt, workdir, f"apply_{run_id}_retry")
        modified_code = _extract_code_block(output)

    if not modified_code:
        _LG.warning(
            "Could not extract modified code from Claude output for %s",
            run_id,
        )
        _LG.debug("Claude output (first 2000 chars): %s", output[:2000])
        print("  Warning: code change failed — no code block in response")
        failed_log = workdir / "logs" / f"apply_{run_id}_failed.md"
        failed_log.write_text(output)
        print(f"  Full output saved to {failed_log}")
        return False

    target = Path(pipeline_script)
    target.write_text(modified_code)
    _LG.info("Wrote modified code to %s", target)
    print(f"  Wrote modified code to {target}")

    source_dir = config.get("source_dir", "")
    if source_dir:
        apply_lint(source_dir)

    scm = config.get("scm", "")
    if scm and source_dir and has_pending_changes(scm, source_dir):
        msg = f"[autoresearch] {run_id}: {exp.get('description', exp['name'])}"
        commit_hash = scm_commit(scm, source_dir, msg)
        exp["commit"] = commit_hash
        run_dir = workdir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "commit.txt").write_text(commit_hash + "\n")
        _LG.info("Committed code changes: %s", commit_hash[:12])
        print(f"  Committed: {commit_hash[:12]}")
        state["cached_image"] = None

    return True


def _get_plan(
    workdir: Path,
    config: dict,
    state: dict,
    knowledge: str,
    pipeline_code: str,
    iteration: int,
) -> dict | None:
    max_iter = config["stopping_criteria"]["max_iterations"]
    patience = config["stopping_criteria"].get("patience", 3)
    untried = _untried_best_practices(state)
    plateau_count = state.get("plateau_count", 0)

    last_analysis = ""
    if state["history"]:
        last = state["history"][-1]
        for d in [
            workdir / "runs" / last["run_id"],
            workdir / "runs" / f"{last['run_id']}_{last['name']}",
            workdir / "runs" / "000_baseline",
        ]:
            afile = d / "analysis.md"
            if afile.exists():
                last_analysis = afile.read_text()
                break

    prompt = load_prompt(
        "plan_next",
        KNOWLEDGE=knowledge,
        MASTER_TABLE=read_master_table(workdir),
        LAST_ANALYSIS=last_analysis[:8000],
        PIPELINE_CODE=pipeline_code or "(not provided)",
        ITERATION=str(iteration),
        MAX_ITERATIONS=str(max_iter),
        PATIENCE=str(patience),
        PLATEAU_COUNT=str(plateau_count),
        BEST_METRIC=str(state.get("best_metric") or "N/A"),
        BEST_PRACTICES_TRIED=(
            ", ".join(state.get("best_practices_tried", [])) or "none"
        ),
        BEST_PRACTICES_REMAINING=", ".join(untried) or "none",
        BASE_LAUNCH_COMMAND=config.get("base_launch_command", ""),
        BUILD_COMMAND=config.get("build_command", ""),
        CACHED_IMAGE=(state.get("cached_image") or "(none — build first)"),
        HISTORY_JSON=json.dumps(state["history"], indent=2)[:6000],
        NOTES=config.get("notes", ""),
    )

    print("Asking Claude to plan next experiments...")
    plan_output = run_claude(prompt, workdir, f"plan_{iteration:03d}")
    plan = extract_json_block(plan_output)

    if not plan:
        print("Could not parse plan from Claude output.")
        print(plan_output[:2000])
        return None

    if plan.get("action") != "stop":
        return plan

    can_stop = plateau_count >= patience and not untried
    if can_stop:
        return plan

    reasons = []
    if untried:
        reasons.append(f"best practices not yet tried: {', '.join(untried)}")
    if plateau_count < patience:
        reasons.append(f"plateau count {plateau_count}/{patience} not reached")
    _LG.info(
        "Claude suggested stop at iteration %d, overriding: %s",
        iteration,
        "; ".join(reasons),
    )
    print(f"Claude suggested stopping, but: {'; '.join(reasons)}")
    print("Re-planning with instruction to continue...")

    override = "\n\n**OVERRIDE: You MUST propose experiments. Do NOT stop.**\n"
    if untried:
        override += (
            "The following best practices have NOT been tried yet: "
            f"{', '.join(untried)}. Prioritize these.\n"
        )
    if plateau_count < patience:
        override += (
            f"Only {plateau_count} consecutive non-improving iterations "
            f"(need {patience} to stop). Keep exploring.\n"
        )
    plan_output = run_claude(prompt + override, workdir, f"plan_{iteration:03d}_retry")
    return extract_json_block(plan_output)


# ------------------------------------------------------------------
# Public API — engine callbacks
# ------------------------------------------------------------------


def analyze_job(
    workdir: Path,
    config: dict,
    state: dict,
    node: object,
    status: str,
) -> object:
    """Analyze a completed job. Returns AnalysisResult."""
    assert isinstance(node, HypothesisNode)

    knowledge = load_knowledge()
    pipeline_code = _read_pipeline_code(config, workdir)
    exp = node.spec
    run_id = node.node_id

    run_dir = workdir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    job_id = node.job_id or ""
    print(f"\nCollecting metrics for {node.name} ({job_id})...")

    system_metrics = fetch_system_metrics(job_id)
    (run_dir / "system_metrics.txt").write_text(system_metrics)

    ps_log = fetch_pipeline_stats(job_id, str(metrics_dir))
    (run_dir / "pipeline_stats_log.txt").write_text(ps_log)

    metrics_content = collect_metrics_summary(metrics_dir)

    prompt = load_prompt(
        "analyze",
        KNOWLEDGE=knowledge,
        JOB_ID=job_id,
        RUN_NAME=node.name,
        CHANGES=exp.get("description", ""),
        HYPOTHESIS=exp.get("hypothesis", ""),
        SYSTEM_METRICS=system_metrics,
        PIPELINE_STATS=metrics_content,
        MASTER_TABLE=read_master_table(workdir),
        PIPELINE_CODE=pipeline_code or "(not provided)",
    )

    print(f"Running Claude analysis for {node.name}...")
    analysis = run_claude(prompt, workdir, f"analyze_{run_id}")
    (run_dir / "analysis.md").write_text(analysis)

    structured = extract_json_block(analysis)

    duration = get_job_duration(job_id) if job_id else None
    if duration is None:
        duration = node.duration

    row: dict[str, str | float] = {
        "run_id": run_id,
        "name": node.name,
        "job_id": job_id,
        "status": status,
        "changes": exp.get("description", ""),
        "duration_s": duration or "",
    }
    if structured and "metrics" in structured:
        m = structured["metrics"]
        row["step_time_ms"] = m.get("step_time_ms", "")
        row["steady_step_time_ms"] = m.get("steady_step_time_ms", "")
        row["ttfb_s"] = m.get("ttfb_s", "")
        row["sm_util_pct"] = m.get("sm_utilization_pct", "")
        row["steady_sm_util_pct"] = m.get("steady_sm_utilization_pct", "")
        row["data_readiness_pct"] = m.get("data_readiness_pct", "")
        row["notes"] = m.get("notes", "")
    append_master_row(workdir, row)

    cur_type, cur_val = _get_compare_val(
        structured.get("metrics", {}) if structured else {}
    )
    best_type, best_val = _current_best_metric(state)
    improved = cur_type != "none" and (
        best_type == "none" or (cur_type == best_type and cur_val < best_val)
    )

    return AnalysisResult(
        structured=structured,
        duration=(
            float(duration)
            if isinstance(duration, (int, float)) and duration > 0
            else None
        ),
        improved=improved,
    )


def get_plan(
    workdir: Path,
    config: dict,
    state: dict,
    parent_node: object,
    result: object,
    tree: dict,
) -> list[dict]:
    """Plan follow-up experiments. Returns experiment spec dicts."""
    assert isinstance(parent_node, HypothesisNode)
    assert isinstance(result, AnalysisResult)

    knowledge = load_knowledge()
    pipeline_code = _read_pipeline_code(config, workdir)
    iteration = state.get("iteration", 0) + 1

    plan = _get_plan(workdir, config, state, knowledge, pipeline_code, iteration)
    if not plan:
        return []

    if plan.get("action") in ("stop", "manual"):
        return []

    experiments = plan.get("experiments", [])
    _record_best_practices(state, plan)

    state["iteration"] = iteration
    write_state(workdir, state)

    return experiments


def prepare_node(
    workdir: Path,
    config: dict,
    state: dict,
    node: object,
    parent: object | None,
) -> bool:
    """Prepare a node: go to parent commit, apply code changes, build."""
    assert isinstance(node, HypothesisNode)

    exp = node.spec
    run_id = node.node_id
    knowledge = load_knowledge()
    pipeline_code = _read_pipeline_code(config, workdir)

    if parent is not None and isinstance(parent, HypothesisNode) and parent.commit:
        scm = config.get("scm", "")
        source_dir = config.get("source_dir", "")
        anchor = state.get("anchor_commit", "")
        if scm and source_dir:
            scm_goto(scm, source_dir, parent.commit, anchor)

    if exp.get("rebuild"):
        success = _apply_code_changes(
            workdir,
            config,
            state,
            exp,
            run_id,
            knowledge,
            pipeline_code,
        )
        if not success:
            return False
        node.commit = exp.get("commit")
        pipeline_code = _read_pipeline_code(config, workdir)
    else:
        if parent is not None and isinstance(parent, HypothesisNode):
            node.commit = parent.commit

    source_dir = config.get("source_dir", "")
    scm = config.get("scm", "")
    if scm and source_dir and has_pending_changes(scm, source_dir):
        msg = f"[autoresearch] {run_id} changes"
        scm_commit(scm, source_dir, msg)
        state["cached_image"] = None

    if exp.get("rebuild") or not state.get("cached_image"):
        build_cmd = config.get("build_command", "")
        if build_cmd:
            image = build_image(build_cmd, workdir, source_dir)
            if image:
                state["cached_image"] = image
                write_state(workdir, state)
            else:
                _LG.warning("Build failed for %s", run_id)
                return False

    if state.get("cached_image"):
        exp["_image"] = state["cached_image"]

    run_dir = workdir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(exp, indent=2) + "\n")

    return True


def launch_node(
    config: dict,
    state: dict,
    node: object,
    workdir: Path,
) -> str | None:
    """Launch a job for a node. Returns job_id or None."""
    assert isinstance(node, HypothesisNode)

    exp = node.spec
    launch_cmd = exp.get("launch_command", "")
    if not launch_cmd:
        launch_cmd = config.get("base_launch_command", "")

    if not launch_cmd:
        print(f"  Skipping {node.name}: no launch_command")
        return None

    image = exp.get("_image") or state.get("cached_image")
    if "$IMAGE" in launch_cmd and image:
        launch_cmd = launch_cmd.replace("$IMAGE", image)

    job_id = launch_job(launch_cmd, workdir)
    run_dir = workdir / "runs" / node.node_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if job_id:
        print(f"  Job ID: {job_id}")
        (run_dir / "job_id.txt").write_text(job_id + "\n")
    else:
        (run_dir / "job_id.txt").write_text("LAUNCH_FAILED\n")

    return job_id


def should_stop(config: dict, state: dict, tree: dict) -> bool:
    """Check if the engine should stop."""
    max_iter = config["stopping_criteria"]["max_iterations"]
    patience = config["stopping_criteria"].get("patience", 3)
    plateau_count = state.get("plateau_count", 0)
    untried = _untried_best_practices(state)
    iteration = state.get("iteration", 0)

    if iteration >= max_iter:
        return True

    return plateau_count >= patience and not untried


def update_on_complete(
    workdir: Path,
    config: dict,
    state: dict,
    node: object,
    result: object,
) -> None:
    """Update autoresearch state after a node completes."""
    assert isinstance(node, HypothesisNode)
    assert isinstance(result, AnalysisResult)

    exp = node.spec

    # Compute best BEFORE appending, so the new entry's own value
    # doesn't prevent it from being recognized as a new best.
    best_type, best_val = _current_best_metric(state)

    state["history"].append(
        {
            "iteration": state.get("iteration", 0),
            "run_id": node.node_id,
            "name": node.name,
            "job_id": node.job_id or "",
            "commit": node.commit or "",
            "completed_at": datetime.now().isoformat(),
            "structured": result.structured,
        }
    )

    cur_type, cur_val = _get_compare_val(
        result.structured.get("metrics", {}) if result.structured else {}
    )

    if cur_type != "none" and (
        best_type == "none" or (cur_type == best_type and cur_val <= best_val)
    ):
        state["current_best"] = node.node_id

    _record_successful_structural_practices(
        state,
        [exp],
        ({node.job_id: {"structured": result.structured}} if node.job_id else {}),
    )

    _update_plateau(state, cur_val)

    write_state(workdir, state)


def update_summary_and_plot(workdir: Path, state: dict) -> None:
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
        s = entry.get("structured") or {}
        m = s.get("metrics", {})
        step = m.get("steady_step_time_ms") or m.get("step_time_ms")
        dur = m.get("duration_s")
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
            experiments = load_tsv(tsv_path)
            if experiments:
                plot_progress(experiments, str(workdir / "progress.png"))
    except Exception as e:
        _LG.warning("Failed to regenerate progress plot: %s", e)

    try:
        tree_file = workdir / "engine" / "tree.json"
        if tree_file.exists():
            plot_hypothesis_tree(
                json.loads(tree_file.read_text()),
                str(workdir / "hypothesis_tree.png"),
            )
    except Exception as e:
        _LG.warning("Failed to regenerate hypothesis tree plot: %s", e)


def build_initial_nodes(
    workdir: Path,
    config: dict,
    state: dict,
) -> list:
    """Create initial HypothesisNode objects for fixed attempts.

    Returns nodes for: baseline, headspace, and MTP.
    Skips any that have already been completed.
    """
    nodes: list[HypothesisNode] = []
    history_names = {e.get("name") for e in state.get("history", [])}
    tried_practices = set(state.get("best_practices_tried", []))

    base_launch = config.get("base_launch_command", "")

    if "baseline" not in history_names:
        nodes.append(
            HypothesisNode(
                node_id="000_baseline",
                name="baseline",
                spec={
                    "name": "baseline",
                    "description": (
                        "Run the unchanged pipeline to establish baseline metrics"
                    ),
                    "hypothesis": "Baseline measurement",
                    "rebuild": False,
                    "launch_command": base_launch,
                    "best_practices_tags": [],
                },
                priority=-1000,
            )
        )

    if not state.get("headspace_done") and "headspace_cache" not in history_names:
        nodes.append(
            HypothesisNode(
                node_id="000_headspace",
                name="headspace_cache",
                spec={
                    "name": "headspace_cache",
                    "description": ("Wrap with CacheDataLoader for headspace analysis"),
                    "hypothesis": (
                        "Measures upper bound of improvement "
                        "from data loading optimization"
                    ),
                    "rebuild": True,
                    "launch_command": base_launch,
                    "best_practices_tags": [],
                    "_is_headspace": True,
                },
                priority=-999,
            )
        )

    if "subprocess_mtp" not in tried_practices:
        nodes.append(
            HypothesisNode(
                node_id="001_subprocess_mtp",
                name="subprocess_mtp",
                spec={
                    "name": "subprocess_mtp",
                    "description": (
                        "Run pipeline in subprocess to avoid GIL contention"
                    ),
                    "hypothesis": (
                        "Subprocess isolation eliminates "
                        "GIL contention and improves throughput"
                    ),
                    "rebuild": True,
                    "launch_command": base_launch,
                    "best_practices_tags": ["subprocess_mtp"],
                },
                priority=-998,
            )
        )

    return nodes
