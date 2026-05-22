# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autoresearch workflow adapter for the generic async runner.

Design note

This module is the domain side of the runner/adapter boundary. It orchestrates
experiment coroutines, but durable state lives in ``store.py`` and deterministic
policy lives in ``policy.py``. Keep SPDL, coding-agent, source-control, metric,
and hypothesis-planning decisions out of ``runner.py``.

The workflow is the right place to sequence domain operations: restore source,
apply experiment changes, build, launch, poll, collect metrics, analyze, update
state, and produce child ``TaskSpec`` objects. Infrastructure-specific work
should sit behind ``AutoresearchPlatform`` capability objects. Do not turn the
runner adapter boundary back into a large flat callback interface.

Workflow failures are structured domain data. Expected failures should carry a
``FailureRecord`` through ``AutoresearchError`` or ``AnalysisResult.failure``;
the runner should never learn autoresearch failure kinds, and workflow code
should not record durable failures as bare strings.

.. mermaid::

   flowchart TB
       Run["run.py"]
       Runner["Orchestrator"]
       Adapter["PipelineOptimizationWorkflow"]
       Store["_WorkflowStateStore"]
       Policy["autoresearch_policy"]
       Platform["AutoresearchPlatform"]
       Agent["_CodingAgent"]

       Run --> Runner
       Run --> Adapter
       Runner -->|"TaskSpec coroutine"| Adapter
       Adapter --> Store
       Adapter --> Policy
       Adapter --> Platform
       Platform --> Agent
       Adapter -->|"TaskResult(children)"| Runner
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from spdl.autoresearch._common._state import _append_master_row
from spdl.autoresearch.core import (
    AnalysisResult,
    AutoresearchError,
    FailureKind,
    FailurePhase,
    FailureRecord,
    HypothesisNode,
    TaskResult,
    TaskSpec,
    TERMINAL_STATUSES,
)

from .._platform import AutoresearchPlatform
from ._analysis_ops import (
    _analyze_job,
    _update_on_complete,
    _update_summary_and_plot,
)
from ._failures import _failure_note, _make_failure, _unexpected_failure
from ._planning_ops import _build_initial_nodes, _plan_followups
from ._policy import (
    _change_summary_for_spec,
    _is_duplicate_spec,
    _node_from_spec,
    _normalize_status,
    _record_failed_best_practice_attempt,
    _select_planning_node,
    _should_cancel_for_stall,
    _spec_from_node,
    _startup_retry_spec,
)
from ._source_ops import _launch_node, _prepare_node
from ._store import _WorkflowStateStore

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = ["PipelineOptimizationWorkflow"]


class PipelineOptimizationWorkflow:
    """Bridge autoresearch domain logic into the simple work scheduler.

    The adapter contract is intentionally small: load specs, checkpoint specs,
    build a coroutine for a spec, and translate a completed result into child
    specs. New autoresearch behavior should usually be implemented in this
    workflow, in the store, in deterministic policy helpers, or behind the
    platform capability boundary.

    .. mermaid::

       flowchart LR
           Spec["TaskSpec"]
           Node["HypothesisNode"]
           Prepare["prepare source/build"]
           Launch["launch or resume job"]
           Poll["poll status/progress"]
           Failure["FailureRecord"]
           Analyze["collect metrics + analyze"]
           Store["update store/views"]
           Children["child TaskSpecs"]

           Spec --> Node
           Node --> Prepare
           Prepare --> Launch
           Launch --> Poll
           Poll --> Failure
           Poll --> Analyze
           Analyze --> Failure
           Analyze --> Store
           Analyze --> Children
    """

    def __init__(
        self,
        *,
        workdir: Path,
        config: dict,
        state: dict,
        platform: AutoresearchPlatform,
    ) -> None:
        self.workdir = workdir
        self.config = config
        self.state = state
        self.platform = platform
        self.poll_interval = int(config.get("poll_interval", 120))
        self.job_timeout_s = float(config.get("job_timeout_s", 1800))

        self._state_lock = asyncio.Lock()
        self._store = _WorkflowStateStore(workdir, state)

    def load(self) -> list[TaskSpec]:
        specs = self._store.load_checkpoint()
        if specs is not None:
            _LG.info("Loaded %d work specs from checkpoint", len(specs))
            return specs

        specs = [_spec_from_node(node) for node in self._initial_nodes()]
        _LG.info("Created %d initial work specs", len(specs))
        return specs

    def checkpoint(
        self,
        queued: list[TaskSpec],
        running: list[TaskSpec],
        status: str,
    ) -> None:
        self._store.save_scheduler_state(queued, running, status)

    def make_coro(self, spec: TaskSpec) -> Coroutine[Any, Any, TaskResult]:
        return self.run_experiment(spec)

    async def on_result(self, spec: TaskSpec, result: TaskResult) -> list[TaskSpec]:
        children = []
        for child in result.children:
            node = _node_from_spec(child)
            if _is_duplicate_spec(
                node.spec,
                self._store.tree.values(),
                base_launch_command=self.config.get("base_launch_command", ""),
            ):
                _LG.info("Skipping duplicate experiment: %s", node.name)
                continue
            parent = self._store.tree.get(node.parent_id) if node.parent_id else None
            if parent is not None:
                self._store.add_child(parent, child)
            else:
                self._store.upsert_node(node)
                self._store.write_all()
            children.append(child)
        return children

    async def run_experiment(self, spec: TaskSpec) -> TaskResult:
        node = _node_from_spec(spec)
        self._store.upsert_node(node)
        try:
            return await self._run_experiment_inner(spec, node)
        except asyncio.CancelledError:
            self._store.update_spec(spec, node)
            raise
        except AutoresearchError as error:
            await self._record_failure(spec, node, error.failure)
            return TaskResult()
        except Exception as error:
            _LG.exception("Experiment %s failed unexpectedly", node.node_id)
            await self._record_failure(
                spec,
                node,
                _unexpected_failure(error, job_id=node.job_id),
            )
            return TaskResult()

    async def _run_experiment_inner(
        self,
        spec: TaskSpec,
        node: HypothesisNode,
    ) -> TaskResult:
        if node.status == "analyzing" and node.job_id:
            status = str(spec.payload.get("terminal_status", "completed"))
            return await self._analyze_record_and_plan(spec, node, status)

        if node.status == "running" and node.job_id:
            status = await self._poll_until_terminal(spec, node)
            return await self._analyze_record_and_plan(spec, node, status)

        prepared = await self._prepare(spec, node)
        if not prepared:
            await self._record_failure(
                spec,
                node,
                _make_failure(
                    FailureKind.CODE_CHANGE_FAILED,
                    FailurePhase.PREPARE,
                    "Prepare step failed",
                ),
            )
            return TaskResult()

        node.job_id = await asyncio.to_thread(
            _launch_node,
            self.config,
            self.state,
            self.platform,
            node,
            self.workdir,
        )

        node.status = "running"
        node.launched_at = time.monotonic()
        self._store.update_spec(spec, node)

        status = await self._poll_until_terminal(spec, node)
        return await self._analyze_record_and_plan(spec, node, status)

    async def _prepare(self, spec: TaskSpec, node: HypothesisNode) -> bool:
        async with self._state_lock:
            node.status = "preparing"
            self._store.update_spec(spec, node)
            parent = self._store.tree.get(node.parent_id) if node.parent_id else None
            prepared = await asyncio.to_thread(
                _prepare_node,
                self.workdir,
                self.config,
                self.state,
                self.platform,
                node,
                parent,
            )
            self._store.update_spec(spec, node)
            return prepared

    async def _poll_until_terminal(self, spec: TaskSpec, node: HypothesisNode) -> str:
        if node.launched_at is None:
            node.launched_at = time.monotonic()

        last_progress: str | None = None
        stall_start = time.monotonic()
        ever_progressed = False

        while True:
            raw_status = await asyncio.to_thread(
                self.platform.execution.status,
                node.job_id or "",
            )
            status = _normalize_status(raw_status)
            if status in TERMINAL_STATUSES:
                spec.payload["terminal_status"] = status
                spec.payload["progress_seen"] = ever_progressed
                self._store.update_spec(spec, node)
                return status

            if await self._is_stalled(node, stall_start, ever_progressed):
                spec.payload["terminal_status"] = "failed"
                spec.payload["progress_seen"] = ever_progressed
                node.failure = _make_failure(
                    FailureKind.JOB_STALLED,
                    FailurePhase.JOB,
                    "Job stalled and was cancelled",
                    details={"progress_seen": ever_progressed},
                    job_id=node.job_id,
                )
                self._store.update_spec(spec, node)
                return "failed"

            progress = await asyncio.to_thread(
                self.platform.execution.progress,
                node.job_id or "",
            )
            if progress is None:
                last_progress = None
                stall_start = time.monotonic()
            elif progress != last_progress:
                last_progress = progress
                stall_start = time.monotonic()
                ever_progressed = True

            self._store.update_spec(spec, node)
            await asyncio.sleep(self.poll_interval)

    async def _is_stalled(
        self,
        node: HypothesisNode,
        stall_start: float,
        ever_progressed: bool,
    ) -> bool:
        now = time.monotonic()
        if not _should_cancel_for_stall(
            now=now,
            launched_at=node.launched_at or now,
            stall_start=stall_start,
            ever_progressed=ever_progressed,
            timeout_s=self.job_timeout_s,
        ):
            return False
        await asyncio.to_thread(self.platform.execution.cancel, node.job_id or "")
        return True

    async def _analyze_record_and_plan(
        self,
        spec: TaskSpec,
        node: HypothesisNode,
        status: str,
    ) -> TaskResult:
        async with self._state_lock:
            node.status = "analyzing"
            self._store.update_spec(spec, node)

            result = await asyncio.to_thread(
                _analyze_job,
                self.workdir,
                self.config,
                self.state,
                self.platform,
                node,
                status,
                bool(spec.payload.get("progress_seen", False)),
            )
            if not isinstance(result, AnalysisResult):
                raise TypeError(f"Expected AnalysisResult, got {type(result)}")

            node.status = "completed" if status == "completed" else "failed"
            node.result = result.structured
            node.duration = result.duration
            if node.failure is None:
                node.failure = result.failure
            self._store.update_spec(spec, node)

            await asyncio.to_thread(
                _update_on_complete,
                self.workdir,
                self.config,
                self.state,
                node,
                result,
            )
            await asyncio.to_thread(_update_summary_and_plot, self.workdir, self.state)

            retry_spec = _startup_retry_spec(node, self.config)
            if retry_spec is not None:
                self._store.write_all()
                return TaskResult(children=[self._create_child_spec(node, retry_spec)])

            planning_node = _select_planning_node(node, self._store.tree)
            if planning_node is None:
                return TaskResult()

            planning_result = result
            if planning_node.node_id != node.node_id:
                planning_result = AnalysisResult(structured=planning_node.result)

            try:
                followups = await asyncio.to_thread(
                    _plan_followups,
                    self.workdir,
                    self.config,
                    self.state,
                    self.platform,
                    planning_node,
                    planning_result,
                    self._store.tree_dict(),
                )
            except AutoresearchError:
                raise
            except Exception as error:
                raise AutoresearchError(
                    _make_failure(
                        FailureKind.PLANNING_FAILED,
                        FailurePhase.PLANNING,
                        "Failed to plan follow-up experiments",
                        exception=error,
                    )
                )
            children = [
                self._create_child_spec(planning_node, child) for child in followups
            ]
            self._store.write_all()
            return TaskResult(children=children)

    async def _record_failure(
        self,
        spec: TaskSpec,
        node: HypothesisNode,
        failure: FailureRecord,
    ) -> None:
        async with self._state_lock:
            node.status = "failed"
            node.failure = failure
            self._store.update_spec(spec, node)
            note = _failure_note(failure)
            _append_master_row(
                self.workdir,
                {
                    "run_id": node.node_id,
                    "name": node.name,
                    "job_id": node.job_id or "",
                    "status": "failed",
                    "changes": node.spec.get("description", ""),
                    "change_summary": _change_summary_for_spec(node.spec),
                    "notes": note,
                },
            )
            self.state.setdefault("history", []).append(
                {
                    "iteration": self.state.get("iteration", 0),
                    "run_id": node.node_id,
                    "name": node.name,
                    "job_id": node.job_id or "",
                    "commit": node.commit or "",
                    "completed_at": _timestamp(),
                    "structured": {
                        "metrics": {},
                        "findings": [note],
                        "failure": failure.to_dict(),
                    },
                }
            )
            _record_failed_best_practice_attempt(self.state, node)
            self._store.write_all()
            await asyncio.to_thread(_update_summary_and_plot, self.workdir, self.state)

    def _initial_nodes(self) -> list[HypothesisNode]:
        nodes = _build_initial_nodes(self.workdir, self.config, self.state)
        for node in nodes:
            self._store.upsert_node(node)
        self._store.write_all()
        return nodes

    def _create_child_spec(self, parent: HypothesisNode, child_spec: dict) -> TaskSpec:
        name = child_spec.get("name", f"exp_{self._store.node_counter}")
        child_spec["change_summary"] = _change_summary_for_spec(child_spec)
        actual_parent = self._resolve_parent(parent, child_spec)
        node = HypothesisNode(
            node_id=f"{self._store.node_counter:03d}_{name}",
            name=name,
            parent_id=actual_parent.node_id,
            commit=actual_parent.commit,
            spec=child_spec,
            status="queued",
            priority=child_spec.get("priority", 0.0),
        )
        self._store.node_counter += 1
        return _spec_from_node(node)

    def _resolve_parent(
        self,
        default_parent: HypothesisNode,
        child_spec: dict,
    ) -> HypothesisNode:
        """Resolve the actual parent node for a child experiment.

        When ``goto`` is absent or null (experiment starts from the anchor
        commit, i.e. the baseline), the parent should be the baseline node —
        not whichever node triggered the planning round.  When ``goto`` points
        to a specific commit, find the node that produced it.

        This prevents mutually-exclusive experiments (e.g. NVDEC GPU decode)
        from being incorrectly parented under incompatible experiments
        (e.g. MTP subprocess).
        """
        goto = child_spec.get("goto")
        if goto is None:
            # Experiment starts from anchor (baseline).  Find the baseline
            # node so the hypothesis tree reflects the true lineage.
            baseline = self._store.tree.get("000_baseline")
            if baseline is not None:
                return baseline
            _LG.warning(
                "Baseline node 000_baseline not found in tree; "
                "falling back to planning node %s",
                default_parent.node_id,
            )
        else:
            # goto points to a specific commit — find the node that owns it.
            for node in self._store.tree.values():
                if node.commit and node.commit == goto:
                    return node
            _LG.warning(
                "goto commit %s does not match any node; "
                "falling back to planning node %s",
                goto,
                default_parent.node_id,
            )
        return default_parent


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
