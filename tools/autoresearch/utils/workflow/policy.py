# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic autoresearch workflow policies.

This module intentionally contains no coding-agent calls, subprocess execution,
or filesystem writes. Keep predictable decisions here so they can be unit tested
without infrastructure.

Design note for future agents
=============================

If a behavior is deterministic and can be expressed from specs, nodes, state,
or metrics already in memory, prefer adding it here instead of burying it in
workflow orchestration or agent prompts. This keeps the workflow simple and
makes core autoresearch behavior testable without source control, subprocesses,
or LLM calls.

.. mermaid::

   flowchart LR
       Inputs["_WorkSpec / _HypothesisNode / state"]
       Policy["pure policy helpers"]
       Decisions["workflow decisions"]
       Tests["unit tests"]

       Inputs --> Policy
       Policy -->|"planning gate"| Decisions
       Policy -->|"duplicate filter"| Decisions
       Policy -->|"stall policy"| Decisions
       Policy -->|"spec/node conversion"| Decisions
       Tests --> Policy
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from ..runner import _WorkSpec
from ..types import _HypothesisNode, _TERMINAL_STATUSES, FailureKind

__all__ = [
    "_change_summary_for_spec",
    "_compare_metric_value",
    "_extract_counter",
    "_extract_total_threads",
    "_initial_experiments_finished",
    "_is_duplicate_spec",
    "_is_headspace_node",
    "_is_startup_failed_retry",
    "_node_from_spec",
    "_normalize_status",
    "_record_failed_best_practice_attempt",
    "_retry_policy_for_failure",
    "_select_planning_node",
    "_should_cancel_for_stall",
    "_spec_from_node",
    "_startup_retry_spec",
    "_update_spec_from_node",
    "_validate_thread_budget",
]

_KIND_EXPERIMENT = "experiment"
_REQUIRED_INITIAL_EXPERIMENTS = frozenset(
    {"baseline", "headspace_cache", "subprocess_mtp"}
)
_STRUCTURAL_ATTEMPT_THRESHOLD = 3
_MAX_THREADS_PER_RANK_DEFAULT = 16
_MAX_THREADS_PER_RANK_EXTENDED = 32
_STARTUP_FAILURE_RETRIES_DEFAULT = 2
_STARTUP_RETRYABLE_EXPERIMENTS_DEFAULT = ("subprocess_mtp",)
_NON_RETRYABLE_FAILURES = frozenset(
    {
        FailureKind.JOB_RUNTIME_FAILED,
        FailureKind.JOB_FAILED,
        FailureKind.BUILD_FAILED,
        FailureKind.LAUNCH_FAILED,
    }
)
_SUMMARY_LIMIT = 34


def _normalize_status(status: str) -> str:
    if status in ("SUCCEEDED", "COMPLETE", "completed"):
        return "completed"
    if status in ("FAILED", "failed"):
        return "failed"
    return "running"


def _is_headspace_node(node: _HypothesisNode) -> bool:
    return bool(node.spec.get("_is_headspace")) or "headspace" in node.name


def _initial_experiments_finished(
    tree: Mapping[str, _HypothesisNode],
    required_names: frozenset[str] = _REQUIRED_INITIAL_EXPERIMENTS,
) -> bool:
    by_name = {node.name: node for node in tree.values()}
    return all(
        (node := by_name.get(name)) is not None and node.status in _TERMINAL_STATUSES
        for name in required_names
    )


def _select_planning_node(
    completed_node: _HypothesisNode,
    tree: Mapping[str, _HypothesisNode],
) -> _HypothesisNode | None:
    """Pick the non-headspace completed node that should trigger planning."""
    if not _initial_experiments_finished(tree):
        return None
    if not _is_headspace_node(completed_node):
        if not _is_startup_failed_retry(completed_node):
            return completed_node

    candidates = [
        node
        for node in tree.values()
        if node.status in _TERMINAL_STATUSES and not _is_headspace_node(node)
    ]
    if not candidates:
        return None
    preferred = [node for node in candidates if not _is_startup_failed_retry(node)]
    return max(preferred or candidates, key=lambda node: _extract_counter(node.node_id))


def _is_duplicate_spec(spec: dict, nodes: Iterable[_HypothesisNode]) -> bool:
    name = spec.get("name", "")
    launch_cmd = spec.get("launch_command", "")
    for node in nodes:
        if node.status == "failed":
            continue
        if name and node.name == name:
            return True
        existing = node.spec
        if launch_cmd and existing.get("launch_command") == launch_cmd:
            return True
    return False


def _should_cancel_for_stall(
    *,
    now: float,
    launched_at: float,
    stall_start: float,
    ever_progressed: bool,
    timeout_s: float,
) -> bool:
    elapsed = now - launched_at
    stall_duration = now - stall_start
    if not ever_progressed:
        return elapsed > timeout_s * 3
    return stall_duration > timeout_s


def _compare_metric_value(metrics: Mapping[str, object]) -> tuple[str, float]:
    step = metrics.get("steady_step_time_ms")
    if isinstance(step, (int, float)) and step > 0:
        return ("step_ms", float(step))
    duration = metrics.get("duration_s")
    if isinstance(duration, (int, float)) and duration > 0:
        return ("duration_s", float(duration))
    return ("none", float("inf"))


def _extract_total_threads(launch_cmd: str) -> int | None:
    fetch = 8
    decode = 16
    match = re.search(r"--num[_-]fetch[_-]threads?\s+(\d+)", launch_cmd)
    if match:
        fetch = int(match.group(1))
    match = re.search(r"--num[_-]decode[_-]threads?\s+(\d+)", launch_cmd)
    if match:
        decode = int(match.group(1))
    match = re.search(r"--num[_-]threads?\s+(\d+)", launch_cmd)
    if match:
        return int(match.group(1))
    return fetch + decode


def _validate_thread_budget(experiments: list[dict], cap: int) -> list[dict]:
    valid = []
    for experiment in experiments:
        command = experiment.get("launch_command", "")
        total = _extract_total_threads(command) if command else None
        if total is not None and total > cap:
            continue
        valid.append(experiment)
    return valid


def _change_summary_for_spec(spec: dict) -> str:
    """Return a concise, stable label for plots and tables."""
    explicit = str(spec.get("change_summary", "")).strip()
    if explicit:
        return _short_change_label(explicit)

    retry_attempt = spec.get("_startup_retry_attempt")
    if retry_attempt:
        return f"startup repair {retry_attempt}"

    tags = spec.get("best_practices_tags", [])
    if isinstance(tags, list) and tags:
        return _short_change_label(str(tags[0]).replace("_", " "))

    command = str(spec.get("launch_command", ""))
    flag = _first_interesting_flag(command)
    if flag:
        return _short_change_label(flag)

    description = str(spec.get("description", "")).strip()
    if description:
        return _short_change_label(description)

    return _short_change_label(str(spec.get("name", "experiment")))


def _short_change_label(text: str) -> str:
    words = [word.strip(".,;:()[]{}") for word in text.replace("_", " ").split()]
    words = [word for word in words if word]
    if not words:
        return "experiment"
    label = " ".join(words[:5])
    if len(label) <= _SUMMARY_LIMIT:
        return label
    return label[: _SUMMARY_LIMIT - 3].rstrip() + "..."


def _first_interesting_flag(command: str) -> str | None:
    tokens = command.split()
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        name = token.lstrip("-").replace("_", " ").replace("-", " ")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            return f"{name} {tokens[index + 1]}"
        return name
    return None


def _record_failed_best_practice_attempt(state: dict, node: _HypothesisNode) -> None:
    tags = node.spec.get("best_practices_tags", [])
    if not tags:
        return
    tried = set(state.get("best_practices_tried", []))
    attempts: dict[str, int] = state.get("_structural_attempts", {})
    for tag in tags:
        if tag == "subprocess_mtp":
            attempts[tag] = attempts.get(tag, 0) + 1
            if attempts[tag] >= _STRUCTURAL_ATTEMPT_THRESHOLD:
                tried.add(tag)
        else:
            tried.add(tag)
    state["_structural_attempts"] = attempts
    state["best_practices_tried"] = sorted(tried)


def _retry_policy_for_failure(node: _HypothesisNode, config: dict) -> dict | None:
    failure = node.failure
    if failure is None:
        return None
    if failure.kind in _NON_RETRYABLE_FAILURES:
        return None
    if failure.kind != FailureKind.JOB_STARTUP_FAILED:
        return None

    retryable = set(
        config.get(
            "startup_retryable_experiments",
            list(_STARTUP_RETRYABLE_EXPERIMENTS_DEFAULT),
        )
    )
    tags = set(node.spec.get("best_practices_tags", []))
    if node.name not in retryable and not tags.intersection(retryable):
        return None

    return {
        "max_attempts": int(
            config.get("startup_failure_retries", _STARTUP_FAILURE_RETRIES_DEFAULT)
        ),
        "reason": "startup_failure",
    }


def _is_startup_failed_retry(node: _HypothesisNode) -> bool:
    failure = node.failure
    return bool(
        node.spec.get("_startup_retry_of")
        and failure is not None
        and failure.kind == FailureKind.JOB_STARTUP_FAILED
    )


def _startup_retry_spec(node: _HypothesisNode, config: dict) -> dict | None:
    """Return a repair experiment for retryable startup failures.

    Failed must-run experiments still count as terminal once attempts are
    exhausted. Startup failures are special because they often indicate
    pickling/import/init problems that the code-changing path can repair.
    """

    failure = node.failure
    policy = _retry_policy_for_failure(node, config)
    if failure is None or policy is None:
        return None

    limit = int(policy["max_attempts"])
    attempt = int(node.spec.get("_startup_retry_attempt", 0))
    if attempt >= limit:
        return None

    next_attempt = attempt + 1
    retry = dict(node.spec)
    retry["name"] = f"{node.name}_startup_retry_{next_attempt}"
    retry["_startup_retry_attempt"] = next_attempt
    retry["_startup_retry_of"] = node.spec.get("_startup_retry_of", node.node_id)
    retry["_startup_failure"] = failure.to_dict()
    retry["description"] = (
        f"Repair startup failure from {node.name}: {failure.message}. "
        f"{node.spec.get('description', '')}"
    ).strip()
    retry["change_summary"] = f"startup repair {next_attempt}"
    retry["hypothesis"] = (
        "Make the experiment initialize successfully before measuring "
        "performance. Focus on pickling/import/configuration/startup issues. "
        f"Previous failure: {failure.message}"
    )
    retry["priority"] = float(node.priority) - 0.01
    return retry


def _spec_from_node(node: _HypothesisNode) -> _WorkSpec:
    node.spec["change_summary"] = _change_summary_for_spec(node.spec)
    return _WorkSpec(
        id=node.node_id,
        priority=node.priority,
        kind=_KIND_EXPERIMENT,
        payload={"node": node.to_dict()},
    )


def _node_from_spec(spec: _WorkSpec) -> _HypothesisNode:
    node_data = spec.payload.get("node", {})
    if not isinstance(node_data, dict):
        raise ValueError(f"Work spec {spec.id} has no node payload")
    return _HypothesisNode.from_dict(node_data)


def _update_spec_from_node(spec: _WorkSpec, node: _HypothesisNode) -> None:
    spec.id = node.node_id
    spec.priority = node.priority
    spec.kind = _KIND_EXPERIMENT
    spec.payload["node"] = node.to_dict()


def _extract_counter(node_id: str) -> int:
    try:
        return int(node_id.split("_", 1)[0])
    except (IndexError, ValueError):
        return 0
