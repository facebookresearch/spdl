# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Initial experiment and follow-up planning operations for autoresearch."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from spdl.autoresearch._common._state import _read_master_table, write_state
from spdl.autoresearch.core import AnalysisResult, HypothesisNode

from .._platform import AutoresearchPlatform
from .._platform._agents import _parse_agent_result
from ._common import _current_best_metric, _read_pipeline_code
from ._policy import (
    _change_summary_for_spec,
    _compare_metric_value,
    _extract_default_executor_concurrency,
    _extract_total_threads,
    _format_best_metric,
    _validate_thread_budget as _policy_validate_thread_budget,
)

_LG: logging.Logger = logging.getLogger(__name__)

_BEST_PRACTICES = [
    "mtp",
    "batch_size_tuning",
    "concurrency_tuning",
]
_STRUCTURAL_PRACTICES = {"mtp"}
_MAX_THREADS_PER_RANK_DEFAULT = 16
_MAX_THREADS_PER_RANK_EXTENDED = 32
_HISTORY_JSON_MAX_CHARS = 12000


def _truncate_history_json(history: list[dict]) -> str:
    for start in range(len(history)):
        text = json.dumps(history[start:], indent=2)
        if len(text) <= _HISTORY_JSON_MAX_CHARS:
            if start > 0:
                return f"[... {start} earlier entries omitted ...]\n{text}"
            return text
    return "[]"


def _untried_best_practices(state: dict) -> list[str]:
    tried = set(state.get("best_practices_tried", []))
    return [bp for bp in _BEST_PRACTICES if bp not in tried]


def _thread_cap_for_experiment(state: dict, config: dict) -> int:
    base_cmd = config.get("base_launch_command", "")
    match = re.search(r"-j\s+\d+x(\d+)", base_cmd)
    num_gpus = int(match.group(1)) if match else 8
    if num_gpus <= 1:
        return _MAX_THREADS_PER_RANK_EXTENDED

    results_at_cap = []
    results_below_cap = []
    for entry in state.get("history", []):
        metrics = (entry.get("structured") or {}).get("metrics", {})
        metric_type, metric_val = _compare_metric_value(metrics)
        if metric_type == "none":
            continue
        run_id = entry.get("run_id", "")
        name = entry.get("name", "")
        if run_id.startswith("000") or "headspace" in name or "cache" in name:
            continue

        spec = (entry.get("structured") or {}).get("experiment", {})
        command = spec.get("launch_command", "") or config.get(
            "base_launch_command", ""
        )
        total = _extract_total_threads(command)
        if total is None:
            continue
        if total >= _MAX_THREADS_PER_RANK_DEFAULT:
            results_at_cap.append(metric_val)
        else:
            results_below_cap.append(metric_val)

    if not results_at_cap:
        return _MAX_THREADS_PER_RANK_DEFAULT
    best_at_cap = max(results_at_cap)  # higher is better
    if results_below_cap and best_at_cap > max(results_below_cap):
        return _MAX_THREADS_PER_RANK_EXTENDED
    return _MAX_THREADS_PER_RANK_DEFAULT


def _validate_thread_budget(experiments: list[dict], cap: int) -> list[dict]:
    valid = _policy_validate_thread_budget(experiments, cap)
    valid_ids = {id(exp) for exp in valid}
    for exp in experiments:
        if id(exp) not in valid_ids:
            command = exp.get("launch_command", "")
            total = _extract_total_threads(command)
            concurrency = _extract_default_executor_concurrency(command)
            _LG.warning(
                "Rejecting %s: num_threads=%s concurrency=%s cap=%d",
                exp.get("name", "?"),
                total,
                concurrency,
                cap,
            )
    return valid


def _record_best_practices(state: dict, plan: dict) -> None:
    tried = set(state.get("best_practices_tried", []))
    for exp in plan.get("experiments", []):
        for tag in exp.get("best_practices_tags", []):
            if tag not in _STRUCTURAL_PRACTICES:
                tried.add(tag)
    state["best_practices_tried"] = sorted(tried)


def _get_plan(
    workdir: Path,
    config: dict,
    state: dict,
    platform: AutoresearchPlatform,
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
        for directory in [
            workdir / "runs" / last["run_id"],
            workdir / "runs" / f"{last['run_id']}_{last['name']}",
            workdir / "runs" / "000_baseline",
        ]:
            analysis_file = directory / "analysis.md"
            if analysis_file.exists():
                last_analysis = analysis_file.read_text()
                break

    findings_file = workdir / "findings.md"
    findings = findings_file.read_text() if findings_file.exists() else "(none yet)"
    cap = _thread_cap_for_experiment(state, config)
    base_cmd = config.get("base_launch_command", "")
    base_threads = _extract_total_threads(base_cmd)
    thread_note = (
        f"\n\n**CPU THREAD BUDGET (HARD LIMIT):** Total threads per rank "
        f"must not exceed {cap}. Use the CPU cores assigned per rank when "
        f"instrumentation reports it; otherwise use roughly 16-20 as the "
        f"maximum practical value. This is enforced via `--num_threads`. "
        f"For stages using the default executor, `num_threads` must be at "
        f"least the maximum stage concurrency, not the sum of all stage "
        f"concurrencies."
    )
    if base_threads is not None:
        thread_note += (
            f" The base launch command currently uses {base_threads} threads."
        )
    else:
        thread_note += (
            " The base launch command does not set explicit `num_threads`, so "
            "thread budget validation is skipped for commands that also omit "
            "it. If you add `num_threads`, keep it within the cap and at least "
            "as large as the max default-executor stage concurrency."
        )
    notes = config.get("notes", "") + thread_note

    prompt = platform.agent._load_prompt(
        "plan_next",
        KNOWLEDGE=knowledge,
        MASTER_TABLE=_read_master_table(workdir),
        LAST_ANALYSIS=last_analysis[:8000],
        PIPELINE_CODE=pipeline_code or "(not provided)",
        ITERATION=str(iteration),
        MAX_ITERATIONS=str(max_iter),
        PATIENCE=str(patience),
        PLATEAU_COUNT=str(plateau_count),
        BEST_METRIC=_format_best_metric(*_current_best_metric(state)),
        BEST_PRACTICES_TRIED=", ".join(state.get("best_practices_tried", [])) or "none",
        BEST_PRACTICES_REMAINING=", ".join(untried) or "none",
        BASE_LAUNCH_COMMAND=config.get("base_launch_command", ""),
        BUILD_COMMAND=config.get("build_command", ""),
        CACHED_IMAGE="(always rebuilt)",
        HISTORY_JSON=_truncate_history_json(state["history"]),
        FINDINGS=findings[:4000],
        NOTES=notes,
    )

    print("Asking coding agent to plan next experiments...")
    plan_output = platform.agent.run(prompt, workdir, f"plan_{iteration:03d}")
    result = _parse_agent_result(platform.agent, plan_output)
    plan = result.json
    if not plan:
        print("Could not parse plan from agent output.")
        print(plan_output[:2000])
        return None
    if plan.get("action") != "stop":
        return plan

    can_stop = plateau_count >= patience and not untried
    if can_stop:
        return plan

    overrides_attempted = state.get("_stop_overrides", 0)
    if overrides_attempted >= 2:
        _LG.warning(
            "Coding agent insisted on stopping %d times at iteration %d; respecting it",
            overrides_attempted + 1,
            iteration,
        )
        return plan

    reasons = []
    if untried:
        reasons.append(f"best practices not yet tried: {', '.join(untried)}")
    if plateau_count < patience:
        reasons.append(f"plateau count {plateau_count}/{patience} not reached")
    _LG.info(
        "Coding agent suggested stop at iteration %d, overriding: %s",
        iteration,
        "; ".join(reasons),
    )
    print(f"Coding agent suggested stopping, but: {'; '.join(reasons)}")
    print("Re-planning with instruction to continue...")
    state["_stop_overrides"] = overrides_attempted + 1

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
    plan_output = platform.agent.run(
        prompt + override, workdir, f"plan_{iteration:03d}_retry"
    )
    return _parse_agent_result(platform.agent, plan_output).json


def _plan_followups(
    workdir: Path,
    config: dict,
    state: dict,
    platform: AutoresearchPlatform,
    parent_node: object,
    result: object,
    tree: dict,
) -> list[dict]:
    """Plan follow-up experiments. Returns experiment spec dicts."""
    assert isinstance(parent_node, HypothesisNode)
    assert isinstance(result, AnalysisResult)

    if parent_node.spec.get("_is_headspace"):
        return []

    knowledge = platform.agent._load_knowledge()
    pipeline_code = _read_pipeline_code(config, workdir)
    iteration = state.get("iteration", 0) + 1

    _, cur_best = _current_best_metric(state)
    prev_best_at_last_plan = state.get("_best_at_last_plan", float("-inf"))
    if cur_best > prev_best_at_last_plan:
        state["plateau_count"] = 0
        state["best_metric"] = cur_best
    else:
        state["plateau_count"] = state.get("plateau_count", 0) + 1
    state["_best_at_last_plan"] = cur_best

    plan = _get_plan(
        workdir, config, state, platform, knowledge, pipeline_code, iteration
    )
    if not plan or plan.get("action") in ("stop", "manual"):
        return []

    experiments = _validate_thread_budget(
        plan.get("experiments", []),
        _thread_cap_for_experiment(state, config),
    )
    plan_goto = plan.get("goto")
    for experiment in experiments:
        if "changes" not in experiment:
            experiment["changes"] = []
        # Propagate plan-level goto to experiments that don't specify their own.
        if "goto" not in experiment:
            experiment["goto"] = plan_goto
        experiment["change_summary"] = _change_summary_for_spec(experiment)
    _record_best_practices(state, {"experiments": experiments})
    state["_stop_overrides"] = 0
    state["iteration"] = iteration
    write_state(workdir, state)
    return experiments


def _build_initial_nodes(workdir: Path, config: dict, state: dict) -> list:
    """Create initial HypothesisNode objects for baseline, headspace, and MTP."""
    nodes: list[HypothesisNode] = []
    history_names = {entry.get("name") for entry in state.get("history", [])}
    tried_practices = set(state.get("best_practices_tried", []))
    base_launch = config.get("base_launch_command", "")

    if "baseline" not in history_names:
        nodes.append(
            HypothesisNode(
                node_id="000_baseline",
                name="baseline",
                spec={
                    "name": "baseline",
                    "changes": [],
                    "description": "Run the unchanged pipeline to establish baseline metrics",
                    "change_summary": "baseline",
                    "hypothesis": "Baseline measurement",
                    "launch_command": base_launch,
                    "best_practices_tags": [],
                },
                priority=-1000,
            )
        )

    headspace_succeeded = any(
        entry.get("name") == "headspace_cache"
        and (entry.get("structured") or {})
        .get("metrics", {})
        .get("steady_step_time_ms")
        for entry in state.get("history", [])
    )
    if not headspace_succeeded:
        nodes.append(
            HypothesisNode(
                node_id="000_headspace",
                name="headspace_cache",
                spec={
                    "name": "headspace_cache",
                    "changes": ["cache_dataloader"],
                    "description": "Wrap with CacheDataLoader for headspace analysis",
                    "change_summary": "headspace",
                    "hypothesis": (
                        "Measures upper bound of improvement "
                        "from data loading optimization"
                    ),
                    "launch_command": base_launch,
                    "best_practices_tags": [],
                    "_is_headspace": True,
                },
                priority=-999,
            )
        )

    if "mtp" not in tried_practices:
        nodes.append(
            HypothesisNode(
                node_id="001_mtp",
                name="mtp",
                spec={
                    "name": "mtp",
                    "changes": ["mtp"],
                    "description": (
                        "Run pipeline in subprocess to avoid background data "
                        "loading threads interfering with CUDA kernel launches"
                    ),
                    "change_summary": "subprocess MTP",
                    "hypothesis": (
                        "Subprocess isolation keeps CPU data loading work out "
                        "of the training process, reducing interference with "
                        "CUDA kernel launch scheduling"
                    ),
                    "launch_command": base_launch,
                    "best_practices_tags": ["mtp"],
                },
                priority=-998,
            )
        )

    return nodes


__all__ = ["_build_initial_nodes", "_plan_followups"]
