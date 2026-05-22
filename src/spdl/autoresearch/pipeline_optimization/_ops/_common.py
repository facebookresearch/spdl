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
    best_step = float("inf")
    best_dur = float("inf")
    for entry in state.get("history", []):
        if _is_headspace_entry(entry):
            continue
        structured = entry.get("structured") or {}
        metrics = structured.get("metrics", {})
        step = metrics.get("steady_step_time_ms")
        if isinstance(step, (int, float)) and step > 0:
            best_step = min(best_step, float(step))
        dur = metrics.get("duration_s")
        if isinstance(dur, (int, float)) and dur > 0:
            best_dur = min(best_dur, float(dur))

    if best_step < float("inf"):
        return ("step_ms", best_step)
    if best_dur < float("inf"):
        return ("duration_s", best_dur)
    return ("none", float("inf"))
