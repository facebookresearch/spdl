# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runner import _WorkSpec

__all__ = ["_run"]


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or edit queued work.")
    parser.add_argument("workdir")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    remove = sub.add_parser("remove")
    remove.add_argument("node_id")
    priority = sub.add_parser("priority")
    priority.add_argument("node_id")
    priority.add_argument("value", type=float)
    return parser.parse_args(args)


def _run(args: list[str]) -> None:
    ns = _parse_args(args)
    checkpoint = Path(ns.workdir).resolve() / "engine" / "checkpoint.json"
    data = _read_checkpoint(checkpoint)
    queued = [_WorkSpec.from_dict(spec) for spec in data.get("queued", [])]

    if ns.action == "list":
        for spec in sorted(queued, key=lambda item: item.priority):
            print(f"{spec.priority:g}\t{spec.id}")
        return

    if ns.action == "remove":
        data["queued"] = [spec.to_dict() for spec in queued if spec.id != ns.node_id]
        _write_checkpoint(checkpoint, data)
        print(f"Removed queued spec {ns.node_id}")
        return

    if ns.action == "priority":
        found = False
        for spec in queued:
            if spec.id == ns.node_id:
                spec.priority = ns.value
                found = True
                break
        if not found:
            raise SystemExit(f"Queued spec not found: {ns.node_id}")
        data["queued"] = [spec.to_dict() for spec in queued]
        _write_checkpoint(checkpoint, data)
        print(f"Updated {ns.node_id} priority to {ns.value:g}")


def _read_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"No checkpoint found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid checkpoint: {path}")
    return data


def _write_checkpoint(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
