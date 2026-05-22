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

from spdl.autoresearch._common._log import setup_logging
from spdl.autoresearch._common._state import _read_master_table, read_config, read_state
from spdl.autoresearch.pipeline_optimization._platform import create_platform

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_read_failures",
    "_run",
]


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final report.")
    parser.add_argument("workdir")
    return parser.parse_args(args)


def _run(args: list[str]) -> None:
    ns = _parse_args(args)
    workdir = Path(ns.workdir).resolve()
    setup_logging(workdir)
    _LG.info("Generating report for %s", workdir)

    state = read_state(workdir)
    config = read_config(workdir)
    platform = create_platform(config, workdir)

    master_table = _read_master_table(workdir)
    failures = _read_failures(workdir)

    all_analyses = []
    for entry in state.get("history", []):
        run_id = entry["run_id"]
        name = entry["name"]
        for candidate in [
            workdir / "runs" / f"{run_id}_{name}",
            workdir / "runs" / "000_baseline",
        ]:
            afile = candidate / "analysis.md"
            if afile.exists():
                all_analyses.append(f"\n## Run {run_id}: {name}\n{afile.read_text()}")
                break

    prompt = f"""You are generating a final report for an autoresearch \
experiment that optimized SPDL data loading pipeline performance.

## Master Table
{master_table}

## Individual Analyses
{"".join(all_analyses)}

## Failures
{failures}

## Task
Write a comprehensive final report in clean markdown:
1. Executive summary — what was tested, what worked, what didn't
2. Key findings — which parameters/changes had the most impact
3. Best configuration discovered and its metrics
4. Recommendations for further optimization

End with a single JSON block containing the best configuration found:
```json
{{"best_run_id": "...", "best_config": {{...}}, "best_sm_util_pct": ...}}
```"""

    print("Generating report...")
    output = platform.agent.run(prompt, workdir, "report")

    report_path = workdir / "report.md"
    report_path.write_text(output)
    print(output)
    print(f"\nReport saved to {report_path}")


def _read_failures(workdir: Path) -> str:
    lines = []
    tree = workdir / "engine" / "tree.json"
    if tree.exists():
        for raw in json.loads(tree.read_text()):
            failure = raw.get("failure")
            if failure:
                lines.append(
                    "- "
                    f"{raw.get('node_id', '')}: {failure.get('kind', '')} - "
                    f"{failure.get('message', '')}"
                )
    setup = workdir / "engine" / "setup_failures.json"
    if setup.exists():
        for failure in json.loads(setup.read_text()):
            lines.append(
                f"- setup: {failure.get('kind', '')} - {failure.get('message', '')}"
            )
    return "\n".join(lines) if lines else "(none)"
