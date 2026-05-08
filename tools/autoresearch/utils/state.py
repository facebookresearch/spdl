# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment state, config, and master table helpers."""

from __future__ import annotations

import json
from pathlib import Path

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
    "notes",
]


def read_state(workdir: Path) -> dict:
    return json.loads((workdir / "state.json").read_text())


def write_state(workdir: Path, state: dict) -> None:
    (workdir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def read_config(workdir: Path) -> dict:
    return json.loads((workdir / "config.json").read_text())


def read_master_table(workdir: Path) -> str:
    return (workdir / "master_table.tsv").read_text()


def _escape_tsv(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def append_master_row(workdir: Path, row: dict) -> None:
    with open(workdir / "master_table.tsv", "a") as f:
        values = [_escape_tsv(str(row.get(h, ""))) for h in MASTER_TABLE_HEADERS]
        f.write("\t".join(values) + "\n")
