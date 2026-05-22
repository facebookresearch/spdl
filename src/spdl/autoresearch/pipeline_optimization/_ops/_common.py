# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small shared helpers for autoresearch workflow modules."""

from __future__ import annotations

from pathlib import Path

from ._policy import _compare_metric_value

__all__ = [
    "_compare_value",
    "_current_best_metric",
    "_is_headspace_entry",
    "_read_pipeline_code",
]


def _read_pipeline_code(config: dict, workdir: Path) -> str:
    pipeline_script = config.get("pipeline_script", "")
    if pipeline_script and Path(pipeline_script).exists():
        return Path(pipeline_script).read_text()
    if (workdir / "pipeline.py").exists():
        return (workdir / "pipeline.py").read_text()
    return ""


def _compare_value(metrics: dict) -> tuple[str, float]:
    return _compare_metric_value(metrics)


def _is_headspace_entry(entry: dict) -> bool:
    name = entry.get("name", "")
    return "headspace" in name or "cache" in name


def _current_best_metric(state: dict) -> tuple[str, float]:
    """Scan history and return the best metric seen so far.

    Higher values are always better (throughput is positive,
    step_time/duration are negated by ``_compare_metric_value``).
    """
    best_type: str = "none"
    best_val: float = float("-inf")
    for entry in state.get("history", []):
        if _is_headspace_entry(entry):
            continue
        structured = entry.get("structured") or {}
        metrics = structured.get("metrics", {})
        cur_type, cur_val = _compare_metric_value(metrics)
        if cur_type == "none":
            continue
        # Prefer throughput over step_ms over duration_s.
        # If types match, take the higher value (= better).
        if best_type == "none" or (cur_type == best_type and cur_val > best_val):
            best_type = cur_type
            best_val = cur_val
        elif cur_type == "throughput" and best_type != "throughput":
            # Throughput always wins over fallback metrics.
            best_type = cur_type
            best_val = cur_val
    return (best_type, best_val)
