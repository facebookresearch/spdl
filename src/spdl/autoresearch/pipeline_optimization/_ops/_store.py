# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent autoresearch workflow state and monitoring views.

Design note

This module owns files under ``<workdir>/engine`` and the persistent
``state.json`` view written by ``state.py``. ``checkpoint.json`` is the resume
source of truth for queued and running ``TaskSpec`` objects; ``queue.json`` and
``active.json`` are compatibility views for humans and monitoring tools.

Keep persistence details here instead of spreading JSON writes across the
runner or workflow. The runner should only ask the adapter to checkpoint; it
should not know the autoresearch workdir layout.

Structured failures are persisted next to node status as ``failure.json`` and
summarized in ``engine_state.json``. Keep that write path centralized here so
future failure categories remain durable and queryable.

.. mermaid::

   flowchart TB
       Store["_WorkflowStateStore"]
       Checkpoint["engine/checkpoint.json\nresume source of truth"]
       Tree["engine/tree.json"]
       Queue["engine/queue.json\nmonitoring view"]
       Active["engine/active.json\nmonitoring view"]
       Nodes["engine/nodes/<node_id>/"]
       Failure["nodes/<node_id>/failure.json"]
       State["state.json"]

       Store --> Checkpoint
       Store --> Tree
       Store --> Queue
       Store --> Active
       Store --> Nodes
       Nodes --> Failure
       Store --> State
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from spdl.autoresearch._common._state import SCHEMA_VERSION
from spdl.autoresearch.core import HypothesisNode, TaskSpec

from ._policy import (
    _extract_counter,
    _node_from_spec,
    _update_spec_from_node,
    write_state,
)

__all__ = [
    "_WorkflowStateStore",
    "_set_queued_priority",
    "_write_text_atomic",
]


class _WorkflowStateStore:
    """Own durable runner checkpoints and autoresearch monitoring files.

    .. mermaid::

       sequenceDiagram
           participant Adapter as PipelineOptimizationWorkflow
           participant Store as _WorkflowStateStore
           participant Disk as workdir/engine

           Adapter->>Store: save_scheduler_state(queued, running, status)
           Store->>Disk: checkpoint.json
           Store->>Disk: queue.json / active.json
           Adapter->>Store: update_spec(spec, node)
           Store->>Disk: nodes/<node_id>/status.txt
           Store->>Disk: tree.json
    """

    def __init__(self, workdir: Path, state: dict) -> None:
        self.workdir = workdir
        self.state = state
        self.engine_dir = workdir / "engine"
        self.nodes_dir = self.engine_dir / "nodes"
        self.tree: dict[str, HypothesisNode] = {}
        self.queued: list[TaskSpec] = []
        self.running: list[TaskSpec] = []
        self.status = "running"
        self.node_counter = 0

    def load_checkpoint(self) -> list[TaskSpec] | None:
        checkpoint = self.engine_dir / "checkpoint.json"
        if not checkpoint.exists():
            return None
        data = json.loads(checkpoint.read_text())
        self._validate_checkpoint(data)
        self.queued = [TaskSpec.from_dict(spec) for spec in data.get("queued", [])]
        self.running = [TaskSpec.from_dict(spec) for spec in data.get("running", [])]
        specs = self.queued + self.running
        self.status = str(data.get("status", "running"))
        for spec in specs:
            self.upsert_node(_node_from_spec(spec))
        return specs

    def _validate_checkpoint(self, data: object) -> None:
        if not isinstance(data, dict):
            raise ValueError("engine/checkpoint.json must contain a JSON object")
        for key in ("queued", "running"):
            specs = data.get(key, [])
            if not isinstance(specs, list):
                raise ValueError(f"engine/checkpoint.json field {key!r} must be a list")
            seen = set()
            for index, raw_spec in enumerate(specs):
                if not isinstance(raw_spec, dict):
                    raise ValueError(f"{key}[{index}] must be a TaskSpec object")
                spec_id = raw_spec.get("id")
                if not isinstance(spec_id, str) or not spec_id:
                    raise ValueError(f"{key}[{index}] must have a non-empty string id")
                if spec_id in seen:
                    raise ValueError(f"Duplicate TaskSpec id in {key}: {spec_id}")
                seen.add(spec_id)
                payload = raw_spec.get("payload")
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("node"),
                    dict,
                ):
                    raise ValueError(f"{key}[{index}] must contain payload.node")

    def save_scheduler_state(
        self,
        queued: list[TaskSpec],
        running: list[TaskSpec],
        status: str,
    ) -> None:
        self.queued = queued
        self.running = running
        self.status = status
        for spec in queued + running:
            self.upsert_node(_node_from_spec(spec))
        self.write_all()

    def update_spec(self, spec: TaskSpec, node: HypothesisNode) -> None:
        _update_spec_from_node(spec, node)
        self.upsert_node(node)
        self._replace_spec(spec, self.queued)
        self._replace_spec(spec, self.running)
        self.write_all()

    def upsert_node(self, node: HypothesisNode) -> None:
        existing = self.tree.get(node.node_id)
        if existing is not None and existing.children and not node.children:
            node.children = existing.children
        self.tree[node.node_id] = node
        self.node_counter = max(self.node_counter, _extract_counter(node.node_id) + 1)

    def add_child(self, parent: HypothesisNode, spec: TaskSpec) -> None:
        node = _node_from_spec(spec)
        self.upsert_node(node)
        stored_parent = self.tree.get(parent.node_id)
        if stored_parent is not None and node.node_id not in stored_parent.children:
            stored_parent.children.append(node.node_id)
        self.write_all()

    def tree_dict(self) -> dict[str, dict]:
        return {node_id: node.to_dict() for node_id, node in self.tree.items()}

    def write_all(self) -> None:
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        for node in self.tree.values():
            self._persist_node(node)
        self._persist_tree()
        self._persist_checkpoint()
        self._persist_queue_view()
        self._persist_active_view()
        self._persist_engine_state()
        self.state["status"] = self.status
        write_state(self.workdir, self.state)

    def _replace_spec(self, spec: TaskSpec, specs: list[TaskSpec]) -> None:
        for index, existing in enumerate(specs):
            if existing.id == spec.id:
                specs[index] = spec
                return

    def _persist_checkpoint(self) -> None:
        _write_text_atomic(
            self.engine_dir / "checkpoint.json",
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": self.status,
                    "timestamp": _timestamp(),
                    "queued": [spec.to_dict() for spec in self.queued],
                    "running": [spec.to_dict() for spec in self.running],
                },
                indent=2,
            )
            + "\n",
        )

    def _persist_queue_view(self) -> None:
        queued = []
        for spec in self.queued:
            node = _node_from_spec(spec)
            queued.append(
                {
                    "priority": spec.priority,
                    "node_id": spec.id,
                    "retry_of": node.spec.get("_startup_retry_of", ""),
                    "retry_attempt": node.spec.get("_startup_retry_attempt", ""),
                }
            )
        (self.engine_dir / "queue.json").write_text(json.dumps(queued, indent=2) + "\n")

    def _persist_active_view(self) -> None:
        active = []
        for spec in self.running:
            node = _node_from_spec(spec)
            if node.status == "running" and node.job_id:
                active.append(
                    {
                        "node_id": node.node_id,
                        "job_id": node.job_id,
                        "launched_at_iso": _timestamp(),
                    }
                )
        (self.engine_dir / "active.json").write_text(
            json.dumps(active, indent=2) + "\n"
        )

    def _persist_engine_state(self) -> None:
        failed_by_kind: dict[str, int] = {}
        for node in self.tree.values():
            if node.status != "failed" or node.failure is None:
                continue
            kind = node.failure.kind.value
            failed_by_kind[kind] = failed_by_kind.get(kind, 0) + 1

        _write_text_atomic(
            self.engine_dir / "engine_state.json",
            json.dumps(
                {
                    "status": self.status,
                    "timestamp": _timestamp(),
                    "total_nodes": len(self.tree),
                    "queued": len(self.queued),
                    "running": len(self.running),
                    "completed": sum(
                        1 for node in self.tree.values() if node.status == "completed"
                    ),
                    "failed": sum(
                        1 for node in self.tree.values() if node.status == "failed"
                    ),
                    "failed_by_kind": failed_by_kind,
                },
                indent=2,
            )
            + "\n",
        )

    def _persist_node(self, node: HypothesisNode) -> None:
        node_dir = self.nodes_dir / node.node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "spec.json").write_text(json.dumps(node.spec, indent=2) + "\n")
        (node_dir / "status.txt").write_text(node.status + "\n")
        if node.result is not None:
            (node_dir / "result.json").write_text(
                json.dumps(node.result, indent=2) + "\n"
            )
        failure_path = node_dir / "failure.json"
        if node.failure is not None:
            _write_text_atomic(
                failure_path,
                json.dumps(node.failure.to_dict(), indent=2) + "\n",
            )
        elif failure_path.exists():
            failure_path.unlink()

    def _persist_tree(self) -> None:
        _write_text_atomic(
            self.engine_dir / "tree.json",
            json.dumps([node.to_dict() for node in self.tree.values()], indent=2)
            + "\n",
        )


def _set_queued_priority(workdir: Path, node_id: str, value: float) -> None:
    """Update the priority of a queued spec in ``engine/checkpoint.json``.

    Operator helper for re-ordering pending experiments without restarting
    the engine. Raises ``SystemExit`` if the checkpoint or queued spec is
    missing.
    """
    checkpoint = workdir.resolve() / "engine" / "checkpoint.json"
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint found: {checkpoint}")
    data = json.loads(checkpoint.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid checkpoint: {checkpoint}")
    queued = [TaskSpec.from_dict(spec) for spec in data.get("queued", [])]
    found = False
    for spec in queued:
        if spec.id == node_id:
            spec.priority = float(value)
            found = True
            break
    if not found:
        raise SystemExit(f"Queued spec not found: {node_id}")
    data["queued"] = [spec.to_dict() for spec in queued]
    checkpoint.write_text(json.dumps(data, indent=2) + "\n")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)
