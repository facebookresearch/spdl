# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment state, config, and master table helpers."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 2

MASTER_TABLE_HEADERS = [
    "run_id",
    "name",
    "job_id",
    "status",
    "step_time_ms",
    "steady_step_time_ms",
    "ttfb_s",
    "sm_util_pct",
    "steady_sm_util_pct",
    "data_readiness_pct",
    "duration_s",
    "changes",
    "change_summary",
    "notes",
]

__all__ = [
    "MASTER_TABLE_HEADERS",
    "SCHEMA_VERSION",
    "_append_master_row",
    "_normalize_config",
    "_normalize_state",
    "_read_master_table",
    "append_master_row",
    "read_config",
    "read_master_table",
    "read_state",
    "write_state",
]


def read_state(workdir: Path) -> dict:
    return _normalize_state(json.loads((workdir / "state.json").read_text()))


def write_state(workdir: Path, state: dict) -> None:
    state = _normalize_state(state)
    (workdir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def read_config(workdir: Path) -> dict:
    return _normalize_config(json.loads((workdir / "config.json").read_text()))


def _read_master_table(workdir: Path) -> str:
    return (workdir / "master_table.tsv").read_text()


read_master_table = _read_master_table


def _escape_tsv(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def _append_master_row(workdir: Path, row: dict) -> None:
    with open(workdir / "master_table.tsv", "a") as f:
        values = [_escape_tsv(str(row.get(h, ""))) for h in MASTER_TABLE_HEADERS]
        f.write("\t".join(values) + "\n")


append_master_row = _append_master_row


def _normalize_config(config: dict) -> dict:
    normalized = dict(config)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("stopping_criteria", {})
    normalized["stopping_criteria"].setdefault("max_iterations", 10)
    normalized["stopping_criteria"].setdefault("patience", 3)
    normalized.setdefault("platform", "auto")
    normalized.setdefault("agent", "claude")
    normalized.setdefault("local_execution_mode", "full")
    normalized.setdefault("startup_failure_retries", 2)
    normalized.setdefault("startup_retryable_experiments", ["subprocess_mtp"])
    return normalized


def _normalize_state(state: dict) -> dict:
    normalized = dict(state)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("iteration", 0)
    normalized.setdefault("status", "initialized")
    normalized.setdefault("baseline_job", None)
    normalized.setdefault("current_best", None)
    normalized.setdefault("best_metric", None)
    normalized.setdefault("plateau_count", 0)
    normalized.setdefault("best_practices_tried", [])
    normalized.setdefault("anchor_commit", "")
    normalized.setdefault("history", [])
    return normalized
