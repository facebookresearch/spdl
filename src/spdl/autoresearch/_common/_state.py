# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic experiment state, config, and master-table I/O.

This module provides workflow-agnostic file I/O for the autoresearch
framework.  Workflow-specific schema defaults (``_normalize_config``,
``_normalize_state``) live in
``spdl.autoresearch.pipeline_optimization._ops._policy``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

SCHEMA_VERSION = 2

__all__ = [
    "SCHEMA_VERSION",
    "_append_master_row",
    "_read_master_table",
    "read_config",
    "read_state",
    "write_state",
]


def read_state(
    workdir: Path,
    normalize: Callable[[dict], dict] | None = None,
) -> dict:
    state = json.loads((workdir / "state.json").read_text())
    return normalize(state) if normalize else state


def write_state(
    workdir: Path,
    state: dict,
    normalize: Callable[[dict], dict] | None = None,
) -> None:
    if normalize:
        state = normalize(state)
    (workdir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def read_config(
    workdir: Path,
    normalize: Callable[[dict], dict] | None = None,
) -> dict:
    config = json.loads((workdir / "config.json").read_text())
    return normalize(config) if normalize else config


def _read_master_table(workdir: Path) -> str:
    path = workdir / "master_table.tsv"
    if not path.exists():
        return ""
    return path.read_text()


def _escape_tsv(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def _append_master_row(workdir: Path, row: dict, headers: list[str]) -> None:
    with open(workdir / "master_table.tsv", "a") as f:
        values = [_escape_tsv(str(row.get(h, ""))) for h in headers]
        f.write("\t".join(values) + "\n")
