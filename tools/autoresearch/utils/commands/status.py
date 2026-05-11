# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..log import setup_logging
from ..state import _read_master_table, read_config, read_state

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_failure_summary",
    "_run",
]


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show experiment status.")
    parser.add_argument("workdir")
    return parser.parse_args(args)


def _run(args: list[str]) -> None:
    ns = _parse_args(args)
    workdir = Path(ns.workdir).resolve()
    setup_logging(workdir)

    state = read_state(workdir)
    config = read_config(workdir)

    max_iter = config["stopping_criteria"]["max_iterations"]
    print(f"Experiment : {workdir}")
    print(f"Status     : {state['status']}")
    print(f"Iteration  : {state['iteration']}/{max_iter}")
    print(f"Baseline   : {state.get('baseline_job', 'none')}")
    print(f"Best run   : {state.get('current_best', 'none')}")
    print(f"Image      : {state.get('cached_image', 'none')}")
    print(_failure_summary(workdir))
    print(f"\nMaster table:\n{_read_master_table(workdir)}")


def _failure_summary(workdir: Path) -> str:
    tree = workdir / "engine" / "tree.json"
    failed = []
    if tree.exists():
        for raw in json.loads(tree.read_text()):
            failure = raw.get("failure")
            if not failure:
                continue
            failed.append(
                "  "
                + "\t".join(
                    [
                        str(raw.get("node_id", "")),
                        str(raw.get("job_id") or ""),
                        str(failure.get("kind", "")),
                        _node_message(raw, failure),
                    ]
                )
            )
    setup = _setup_failure_lines(workdir)
    if not failed and not setup:
        return "Failures  : none"
    lines = ["Failures  :"]
    lines.extend(failed)
    lines.extend(setup)
    return "\n".join(lines)


def _setup_failure_lines(workdir: Path) -> list[str]:
    path = workdir / "engine" / "setup_failures.json"
    if not path.exists():
        return []
    failures = json.loads(path.read_text())
    return [
        "  "
        + "\t".join(
            [
                "setup",
                "",
                str(failure.get("kind", "")),
                str(failure.get("message", "")),
            ]
        )
        for failure in failures
    ]


def _node_message(raw: dict, failure: dict) -> str:
    spec = raw.get("spec") or {}
    parts = []
    retry_of = spec.get("_startup_retry_of")
    retry_attempt = spec.get("_startup_retry_attempt")
    if retry_of:
        parts.append(f"retry {retry_attempt} of {retry_of}")
    parts.append(str(failure.get("message", "")))
    return "; ".join(part for part in parts if part)
