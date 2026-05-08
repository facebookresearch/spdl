# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic async experiment execution engine.

This module provides a domain-agnostic engine for running tree-structured
experiments with configurable concurrency. It knows nothing about the
experiment domain — all behavior is injected via async callbacks.

Key components:

    HypothesisNode
        A node in the experiment tree. Each node represents one experiment
        with its spec, status, results, and parent/child relationships.

    AnalysisResult
        Returned by the analyze callback after a job completes.

    ExperimentEngine
        The main event loop. Manages a priority queue of pending nodes,
        launches them up to a concurrency limit, polls for completion,
        triggers analysis, plans follow-ups, and persists state to disk
        for crash recovery.

Usage::

    engine = ExperimentEngine(
        work_dir="/tmp/experiment",
        max_concurrency=3,
        launch_fn=my_launch, check_fn=my_check, cancel_fn=my_cancel,
        analyze_fn=my_analyze, plan_fn=my_plan, prepare_fn=my_prepare,
        should_stop_fn=my_stop,
    )
    asyncio.run(engine.run(initial_nodes))

    # Ctrl+C to stop gracefully — state is persisted to disk.
    # Re-run to resume from where it left off.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

_LG: logging.Logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"completed", "failed"})


@dataclass
class HypothesisNode:
    """A node in the hypothesis tree representing one experiment.

    Attributes:
        node_id: Unique identifier (e.g., "001_baseline").
        name: Human-readable experiment name.
        parent_id: ID of the parent node, or None for root nodes.
        commit: Opaque string representing the source state for this
            experiment (e.g., an SCM commit hash). Set by prepare_fn.
        spec: Experiment specification dict. Opaque to the engine —
            interpreted by callbacks (launch_fn, prepare_fn, etc.).
        status: Current lifecycle status. One of: "queued", "preparing",
            "running", "analyzing", "completed", "failed".
        job_id: Identifier of the launched job. Set after launch_fn.
        launched_at: Monotonic timestamp when the job was launched.
            Used for wall-clock stuck detection.
        result: Analysis result dict. Set after analyze_fn completes.
        children: List of child node IDs.
        priority: Queue priority. Lower values are dequeued first.
        duration: Job duration in seconds, set after analysis.
    """

    node_id: str
    name: str
    parent_id: str | None = None
    commit: str | None = None
    spec: dict = field(default_factory=dict)
    status: str = "queued"
    job_id: str | None = None
    launched_at: float | None = None
    result: dict | None = None
    children: list[str] = field(default_factory=list)
    priority: float = 0.0
    duration: float | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON persistence."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> HypothesisNode:
        """Deserialize from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AnalysisResult:
    """Result returned by the analyze callback.

    Attributes:
        structured: Parsed analysis data (metrics, findings, etc.).
            Opaque to the engine — stored on the node for the consumer.
        duration: Job duration in seconds, or None if unavailable.
        improved: Whether this experiment improved over its parent or
            the current best. Used by the consumer for prioritization.
    """

    structured: dict | None = None
    duration: float | None = None
    improved: bool = False


class _WorkQueue:
    """Priority queue of hypothesis node IDs.

    Lower priority values are dequeued first. Supports re-sorting
    after dynamic priority updates.
    """

    def __init__(self) -> None:
        self._items: list[tuple[float, str]] = []

    def push(self, priority: float, node_id: str) -> None:
        """Add a node ID with the given priority."""
        self._items.append((priority, node_id))
        self._items.sort()

    def pop(self) -> str | None:
        """Remove and return the highest-priority (lowest value) node ID."""
        if not self._items:
            return None
        return self._items.pop(0)[1]

    def reprioritize(self, priorities: dict[str, float]) -> None:
        """Re-sort the queue with updated priority values.

        Args:
            priorities: Mapping of node_id to new priority value.
                Node IDs not in the map keep their current priority.
        """
        self._items = [(priorities.get(nid, pri), nid) for pri, nid in self._items]
        self._items.sort()

    def remove(self, node_id: str) -> None:
        """Remove a specific node from the queue."""
        self._items = [(p, n) for p, n in self._items if n != node_id]

    def node_ids(self) -> list[str]:
        """Return all queued node IDs in priority order."""
        return [nid for _, nid in self._items]

    def to_list(self) -> list[dict]:
        """Serialize for JSON persistence."""
        return [{"priority": p, "node_id": n} for p, n in self._items]

    @classmethod
    def from_list(cls, data: list[dict]) -> _WorkQueue:
        """Deserialize from JSON."""
        q = cls()
        q._items = [(d["priority"], d["node_id"]) for d in data]
        q._items.sort()
        return q

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)


class ExperimentEngine:
    """Async experiment execution engine with tree-structured hypotheses.

    Coordinates concurrent job execution with configurable parallelism,
    immediate analysis on completion, dynamic re-prioritization, and
    disk-backed state for stop/resume and crash recovery.

    The engine is generic — all domain-specific behavior is provided via
    async callbacks. It has no knowledge of the experiment domain.

    Each experiment node runs as an independent ``asyncio.Task`` through
    the lifecycle: prepare → launch → poll → analyze. The main loop
    manages a set of running tasks and a priority queue of waiting nodes.
    ``asyncio.wait(return_when=FIRST_COMPLETED)`` drives event processing:
    whichever job finishes first is immediately analyzed, follow-up
    experiments are enqueued, and the next job is launched.

    Args:
        work_dir: Directory for engine state files (``engine/`` subdirectory).
        max_concurrency: Maximum number of jobs running simultaneously.
        poll_interval: Seconds between job status polls.
        job_timeout_s: Wall-clock timeout in seconds. Jobs running longer
            than this are killed immediately.
        launch_fn: ``async (node) -> job_id | None``.
            Launch a job for the given node. Return the job identifier,
            or None if launch failed.
        check_fn: ``async (job_id) -> status_str``.
            Check job status. Must return one of: ``"completed"``,
            ``"failed"``, ``"running"``, ``"pending"``, or similar.
            Only ``"completed"`` and ``"failed"`` are treated as terminal.
        cancel_fn: ``async (job_id) -> bool``.
            Cancel a running job. Return True on success.
        analyze_fn: ``async (node, status) -> AnalysisResult``.
            Collect metrics and analyze a completed/failed job.
        plan_fn: ``async (parent_node, result, tree_dict) -> list[dict]``.
            Given a completed node and its analysis, propose follow-up
            experiments. Each dict becomes the ``spec`` of a child node.
            Must include at least ``"name"`` and ``"description"`` keys.
        prepare_fn: ``async (node, parent_node | None) -> bool``.
            Prepare a node before launch: apply code changes, build
            images, etc. Return True on success. The engine serializes
            calls to prepare_fn via an asyncio.Lock since preparation
            typically modifies shared working directory state.
        should_stop_fn: ``async (tree_dict) -> bool``.
            Return True if the engine should stop requesting new work.
            Called after each completion. The engine still drains running
            jobs before exiting.
        on_node_complete: ``async (node, result) -> None``. Optional.
            Called after analysis is done. Use for side effects like
            persisting consumer state, regenerating plots, etc.
            Serialized via the same lock as analyze_fn — safe to
            mutate shared state without external locking.
        reprioritize_fn: ``async (queued_nodes, tree_dict) -> list[HypothesisNode]``.
            Optional. Re-sort queued nodes by priority after new
            information is available. Return the nodes with updated
            priority fields.
        progress_fn: ``async (job_id) -> str | None``. Optional.
            Return a progress indicator for a running job (e.g., the
            latest ``[autoresearch] step=N`` line from logs). If the
            returned value changes between polls, the stall timer resets.
            If it stays the same for ``job_timeout_s``, the job is killed.
            When not provided, falls back to wall-clock timeout.

    Lifecycle::

        engine = ExperimentEngine(work_dir=..., callbacks=...)
        asyncio.run(engine.run(initial_nodes))

        # Ctrl+C persists state. Re-run to resume:
        engine = ExperimentEngine(work_dir=..., callbacks=...)
        await engine.load_state()   # rebuilds tree, queue, active jobs
        asyncio.run(engine.run())   # resumes
    """

    def __init__(
        self,
        work_dir: str,
        max_concurrency: int = 3,
        poll_interval: int = 120,
        job_timeout_s: float = 1800,
        *,
        launch_fn: Callable[[HypothesisNode], Awaitable[str | None]],
        check_fn: Callable[[str], Awaitable[str]],
        cancel_fn: Callable[[str], Awaitable[bool]],
        analyze_fn: Callable[[HypothesisNode, str], Awaitable[AnalysisResult]],
        plan_fn: Callable[
            [HypothesisNode, AnalysisResult, dict[str, dict]],
            Awaitable[list[dict]],
        ],
        prepare_fn: Callable[[HypothesisNode, HypothesisNode | None], Awaitable[bool]],
        should_stop_fn: Callable[[dict[str, dict]], Awaitable[bool]],
        on_node_complete: (
            Callable[[HypothesisNode, AnalysisResult], Awaitable[None]] | None
        ) = None,
        reprioritize_fn: (
            Callable[
                [list[HypothesisNode], dict[str, dict]],
                Awaitable[list[HypothesisNode]],
            ]
            | None
        ) = None,
        progress_fn: (Callable[[str], Awaitable[str | None]] | None) = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.max_concurrency = max_concurrency
        self.poll_interval = poll_interval
        self.job_timeout_s = job_timeout_s

        self.launch_fn = launch_fn
        self.check_fn = check_fn
        self.cancel_fn = cancel_fn
        self.analyze_fn = analyze_fn
        self.plan_fn = plan_fn
        self.prepare_fn = prepare_fn
        self.should_stop_fn = should_stop_fn
        self.on_node_complete = on_node_complete
        self.reprioritize_fn = reprioritize_fn
        self.progress_fn = progress_fn

        self._tree: dict[str, HypothesisNode] = {}
        self._queue = _WorkQueue()
        self._state_lock = asyncio.Lock()
        self._node_counter = 0

        self._engine_dir = self.work_dir / "engine"
        self._nodes_dir = self._engine_dir / "nodes"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, initial_nodes: list[HypothesisNode] | None = None) -> None:
        """Run the engine event loop.

        Seeds the queue with ``initial_nodes`` (if any), then enters the
        main loop: fill slots, wait for completions, plan follow-ups.

        Handles SIGINT/SIGTERM by cancelling tasks and persisting state.
        """
        self._seed_queue(initial_nodes or [])
        pending: set[asyncio.Task] = set()
        stop_event = asyncio.Event()
        self._install_signal_handlers(stop_event, pending)

        try:
            await self._event_loop(pending, stop_event)
        except asyncio.CancelledError:
            _LG.info("Engine cancelled")
        finally:
            self._finalize(stop_event, pending)

    def _seed_queue(self, nodes: list[HypothesisNode]) -> None:
        for node in nodes:
            self._add_node(node)
            self._queue.push(node.priority, node.node_id)
            self._persist_node(node)
        self._persist_engine_state("running")
        self._persist_queue()

    async def _event_loop(
        self, pending: set[asyncio.Task], stop_event: asyncio.Event
    ) -> None:
        while (pending or self._queue) and not stop_event.is_set():
            self._fill_slots(pending, stop_event)
            if not pending:
                break
            try:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                break
            await self._process_completions(done)
            if await self._check_should_drain(pending, stop_event):
                break

    def _finalize(self, stop_event: asyncio.Event, pending: set[asyncio.Task]) -> None:
        status = "interrupted" if stop_event.is_set() or pending else "stopped"
        self._persist_all()
        self._persist_engine_state(status)
        _LG.info("Engine %s. Tree has %d nodes.", status, len(self._tree))
        print(f"Engine {status}. State saved to {self._engine_dir}")

    def _install_signal_handlers(
        self, stop_event: asyncio.Event, pending: set[asyncio.Task]
    ) -> None:
        """Register SIGINT/SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        def _on_shutdown(sig_name: str) -> None:
            _LG.info("%s received — shutting down gracefully", sig_name)
            print(f"\n{sig_name} received — persisting state and shutting down...")
            stop_event.set()
            for t in pending:
                t.cancel()

        try:
            loop.add_signal_handler(signal.SIGINT, _on_shutdown, "SIGINT")
            loop.add_signal_handler(signal.SIGTERM, _on_shutdown, "SIGTERM")
        except NotImplementedError:
            pass

    def _fill_slots(
        self, pending: set[asyncio.Task], stop_event: asyncio.Event
    ) -> None:
        """Create tasks from the priority queue up to max_concurrency."""
        while (
            len(pending) < self.max_concurrency
            and self._queue
            and not stop_event.is_set()
        ):
            node_id = self._queue.pop()
            if node_id is None:
                break
            node = self._tree.get(node_id)
            if node is None:
                _LG.warning("Queued node %s not in tree, skipping", node_id)
                continue
            task = asyncio.create_task(self._run_node(node), name=f"node:{node_id}")
            pending.add(task)
            self._persist_queue()

    async def _process_completions(self, done: set[asyncio.Task]) -> None:
        """Analyze completed tasks and enqueue follow-ups."""
        for task in done:
            try:
                node, result = task.result()
            except asyncio.CancelledError:
                continue
            except Exception:
                _LG.exception("Task failed with exception")
                continue

            if result is None:
                continue

            async with self._state_lock:
                followups = await self.plan_fn(node, result, self._get_tree_dict())
            for spec in followups:
                if self._is_duplicate(spec):
                    _LG.info(
                        "Skipping duplicate experiment: %s",
                        spec.get("name", "?"),
                    )
                    continue
                child = self._create_child_node(node, spec)
                self._add_node(child)
                self._queue.push(child.priority, child.node_id)
                self._persist_node(child)
            self._persist_queue()

            if self.reprioritize_fn:
                await self._reprioritize()

    async def _check_should_drain(
        self, pending: set[asyncio.Task], stop_event: asyncio.Event
    ) -> bool:
        """Check stopping condition; drain remaining tasks if met."""
        if stop_event.is_set() or self._queue:
            return False
        if not await self.should_stop_fn(self._get_tree_dict()):
            return False

        _LG.info("Stopping condition met, draining running tasks")
        if pending:
            try:
                done, _ = await asyncio.wait(pending)
                for task in done:
                    try:
                        task.result()
                    except (asyncio.CancelledError, Exception):
                        pass
            except asyncio.CancelledError:
                pass
        return True

    async def load_state(self) -> None:
        """Restore engine state from disk for resumption.

        Reads the hypothesis tree, work queue, and active job list from
        the ``engine/`` directory. For nodes that were running when the
        engine was interrupted, checks their current status via
        ``check_fn``:

        - If completed/failed: marks them for analysis (re-enqueues with
          status "analyzing").
        - If still running: re-creates polling tasks on the next
          ``run()`` call.

        Call this before ``run()`` when resuming a previous session.
        """
        tree_file = self._engine_dir / "tree.json"
        queue_file = self._engine_dir / "queue.json"
        active_file = self._engine_dir / "active.json"

        if tree_file.exists():
            tree_data = json.loads(tree_file.read_text())
            for node_data in tree_data:
                node = HypothesisNode.from_dict(node_data)
                self._tree[node.node_id] = node
                self._node_counter = max(
                    self._node_counter, _extract_counter(node.node_id) + 1
                )
            _LG.info("Loaded %d nodes from tree.json", len(self._tree))

        if queue_file.exists():
            queue_data = json.loads(queue_file.read_text())
            self._queue = _WorkQueue.from_list(queue_data)
            _LG.info("Loaded %d items from queue.json", len(self._queue))

        if active_file.exists():
            active_data = json.loads(active_file.read_text())
            for entry in active_data:
                node_id = entry["node_id"]
                job_id = entry.get("job_id")
                node = self._tree.get(node_id)
                if node is None or job_id is None:
                    continue

                status = await self.check_fn(job_id)
                if status in TERMINAL_STATUSES:
                    node.status = "analyzing"
                    node.job_id = job_id
                    self._queue.push(-2000, node_id)
                    _LG.info(
                        "Previously active node %s (%s) is now %s, queued for analysis",
                        node_id,
                        job_id,
                        status,
                    )
                else:
                    node.status = "running"
                    node.job_id = job_id
                    node.launched_at = time.monotonic()
                    self._queue.push(-2000, node_id)
                    _LG.info(
                        "Previously active node %s (%s) still running, re-queued",
                        node_id,
                        job_id,
                    )

    # ------------------------------------------------------------------
    # Per-node task
    # ------------------------------------------------------------------

    async def _run_node(
        self, node: HypothesisNode
    ) -> tuple[HypothesisNode, AnalysisResult | None]:
        """Execute one hypothesis node through its full lifecycle.

        Stages:
            1. Prepare — serialized via ``_state_lock`` (shared workdir).
            2. Launch — obtain a job ID.
            3. Poll — check status every ``poll_interval`` seconds until
               terminal or wall-clock timeout exceeded.
            4. Analyze — collect metrics, produce ``AnalysisResult``.

        On ``CancelledError`` (from SIGINT), persists the node's current
        state and re-raises so the main loop can handle shutdown.

        Returns:
            Tuple of (node, AnalysisResult) on success, or
            (node, None) if preparation or launch failed.
        """
        try:
            return await self._run_node_inner(node)
        except asyncio.CancelledError:
            self._persist_node(node)
            raise

    async def _run_node_inner(
        self, node: HypothesisNode
    ) -> tuple[HypothesisNode, AnalysisResult | None]:
        # If this node was recovered mid-run and already has a job_id,
        # skip prepare + launch and go straight to polling or analysis.
        if node.status == "analyzing" and node.job_id:
            return await self._analyze_and_complete(node, "completed")

        if node.status == "running" and node.job_id:
            return await self._poll_and_complete(node)

        # 1. Prepare
        node.status = "preparing"
        self._persist_node(node)
        _LG.info("Preparing node %s", node.node_id)

        async with self._state_lock:
            parent = self._tree.get(node.parent_id) if node.parent_id else None
            success = await self.prepare_fn(node, parent)
            if not success:
                node.status = "failed"
                self._persist_node(node)
                _LG.warning("Preparation failed for node %s", node.node_id)
                return node, None

        # 2. Launch
        _LG.info("Launching node %s", node.node_id)
        node.job_id = await self.launch_fn(node)
        if not node.job_id:
            node.status = "failed"
            self._persist_node(node)
            _LG.warning("Launch failed for node %s", node.node_id)
            return node, None

        node.status = "running"
        node.launched_at = time.monotonic()
        self._persist_node(node)
        self._persist_active()
        _LG.info("Node %s launched as job %s", node.node_id, node.job_id)

        return await self._poll_and_complete(node)

    async def _poll_and_complete(
        self, node: HypothesisNode
    ) -> tuple[HypothesisNode, AnalysisResult | None]:
        """Poll a running job until completion or stall timeout, then analyze."""
        if node.launched_at is None:
            node.launched_at = time.monotonic()

        last_progress: str | None = None
        stall_start = time.monotonic()

        status = "running"
        while True:
            status = await self.check_fn(node.job_id)
            if status in TERMINAL_STATUSES:
                _LG.info("Node %s job %s: %s", node.node_id, node.job_id, status)
                break

            if await self._check_stalled(node, last_progress, stall_start):
                status = "failed"
                break

            last_progress, stall_start = await self._update_progress(
                node, last_progress, stall_start
            )

            await asyncio.sleep(self.poll_interval)

        return await self._analyze_and_complete(node, status)

    async def _check_stalled(
        self,
        node: HypothesisNode,
        last_progress: str | None,
        stall_start: float,
    ) -> bool:
        """Check if a job has stalled and kill it if so. Returns True if killed."""
        stall_duration = time.monotonic() - stall_start
        if stall_duration <= self.job_timeout_s:
            return False

        elapsed = time.monotonic() - (node.launched_at or stall_start)
        _LG.warning(
            "Node %s job %s stuck — no progress for %.0fs (%.0fs total), killing",
            node.node_id,
            node.job_id,
            stall_duration,
            elapsed,
        )
        await self.cancel_fn(node.job_id)
        return True

    async def _update_progress(
        self,
        node: HypothesisNode,
        last_progress: str | None,
        stall_start: float,
    ) -> tuple[str | None, float]:
        """Check progress and reset stall timer if advancing.

        Returns (progress, stall_start).

        Three cases:
        - progress_fn returns a new value: job is advancing, reset timer.
        - progress_fn returns same value as before: job is stalled at
          the same step, keep counting.
        - progress_fn returns None: no progress signal available (e.g.,
          job hasn't emitted [autoresearch] lines yet, or is in a phase
          like CacheDataLoader cache-fill). Reset the timer — we can't
          distinguish "no signal" from "stuck" so we give benefit of
          the doubt.
        """
        elapsed = time.monotonic() - (node.launched_at or stall_start)

        if not self.progress_fn:
            _LG.debug(
                "Node %s job %s: running (%.0fs elapsed)",
                node.node_id,
                node.job_id,
                elapsed,
            )
            return last_progress, stall_start

        progress = await self.progress_fn(node.job_id)

        if progress is None:
            _LG.debug(
                "Node %s job %s: no progress signal (%.0fs elapsed)",
                node.node_id,
                node.job_id,
                elapsed,
            )
            return None, time.monotonic()

        if progress != last_progress:
            _LG.debug(
                "Node %s job %s: progressing (%.0fs elapsed)",
                node.node_id,
                node.job_id,
                elapsed,
            )
            return progress, time.monotonic()

        stall_duration = time.monotonic() - stall_start
        _LG.debug(
            "Node %s job %s: stalled at same step for %.0fs (%.0fs total)",
            node.node_id,
            node.job_id,
            stall_duration,
            elapsed,
        )
        return last_progress, stall_start

    async def _analyze_and_complete(
        self, node: HypothesisNode, status: str
    ) -> tuple[HypothesisNode, AnalysisResult | None]:
        """Analyze a completed/failed job and finalize the node.

        Serialized via ``_state_lock`` to prevent concurrent mutations
        of shared consumer state (e.g., history, iteration counters)
        when multiple jobs complete around the same time.
        """
        async with self._state_lock:
            node.status = "analyzing"
            self._persist_node(node)
            _LG.info("Analyzing node %s (status=%s)", node.node_id, status)

            result = await self.analyze_fn(node, status)

            node.status = "completed" if status == "completed" else "failed"
            node.result = result.structured
            node.duration = result.duration
            self._persist_node(node)
            self._persist_active()

            if self.on_node_complete:
                await self.on_node_complete(node, result)

            _LG.info(
                "Node %s complete: duration=%s improved=%s",
                node.node_id,
                f"{result.duration:.0f}s" if result.duration else "N/A",
                result.improved,
            )
            return node, result

    # ------------------------------------------------------------------
    # Tree and queue management
    # ------------------------------------------------------------------

    def _add_node(self, node: HypothesisNode) -> None:
        """Add a node to the tree and update parent's children list."""
        self._tree[node.node_id] = node
        self._node_counter = max(self._node_counter, _extract_counter(node.node_id) + 1)
        if node.parent_id and node.parent_id in self._tree:
            parent = self._tree[node.parent_id]
            if node.node_id not in parent.children:
                parent.children.append(node.node_id)
                self._persist_node(parent)

    def _create_child_node(self, parent: HypothesisNode, spec: dict) -> HypothesisNode:
        """Create a child HypothesisNode from a plan spec dict."""
        name = spec.get("name", f"exp_{self._node_counter}")
        node_id = f"{self._node_counter:03d}_{name}"
        self._node_counter += 1

        return HypothesisNode(
            node_id=node_id,
            name=name,
            parent_id=parent.node_id,
            commit=parent.commit,
            spec=spec,
            status="queued",
            priority=spec.get("priority", 0.0),
        )

    def _is_duplicate(self, spec: dict) -> bool:
        """Check if an experiment with the same name already exists."""
        name = spec.get("name", "")
        if not name:
            return False
        for node in self._tree.values():
            if node.name == name:
                return True
        return False

    def _get_tree_dict(self) -> dict[str, dict]:
        """Return the tree as a dict of node_id → serialized node."""
        return {nid: n.to_dict() for nid, n in self._tree.items()}

    async def _reprioritize(self) -> None:
        """Re-sort the work queue using the reprioritize callback."""
        if not self.reprioritize_fn:
            return
        queued_ids = self._queue.node_ids()
        queued_nodes = [self._tree[nid] for nid in queued_ids if nid in self._tree]
        if not queued_nodes:
            return

        reordered = await self.reprioritize_fn(queued_nodes, self._get_tree_dict())
        new_priorities = {n.node_id: n.priority for n in reordered}
        self._queue.reprioritize(new_priorities)
        for n in reordered:
            self._tree[n.node_id].priority = n.priority
        self._persist_queue()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_node(self, node: HypothesisNode) -> None:
        """Write a node's state to its directory under engine/nodes/."""
        node_dir = self._nodes_dir / node.node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "spec.json").write_text(json.dumps(node.spec, indent=2) + "\n")
        (node_dir / "status.txt").write_text(node.status + "\n")
        if node.result is not None:
            (node_dir / "result.json").write_text(
                json.dumps(node.result, indent=2) + "\n"
            )

    def _persist_queue(self) -> None:
        """Write the current queue to engine/queue.json."""
        self._engine_dir.mkdir(parents=True, exist_ok=True)
        (self._engine_dir / "queue.json").write_text(
            json.dumps(self._queue.to_list(), indent=2) + "\n"
        )

    def _persist_active(self) -> None:
        """Write currently running nodes to engine/active.json."""
        self._engine_dir.mkdir(parents=True, exist_ok=True)
        active = []
        for node in self._tree.values():
            if node.status == "running" and node.job_id:
                active.append(
                    {
                        "node_id": node.node_id,
                        "job_id": node.job_id,
                        "launched_at_iso": (
                            time.strftime("%Y-%m-%dT%H:%M:%S")
                            if node.launched_at
                            else None
                        ),
                    }
                )
        (self._engine_dir / "active.json").write_text(
            json.dumps(active, indent=2) + "\n"
        )

    def _persist_engine_state(self, status: str) -> None:
        """Write engine metadata to engine/engine_state.json."""
        self._engine_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_nodes": len(self._tree),
            "queued": len(self._queue),
            "running": sum(1 for n in self._tree.values() if n.status == "running"),
            "completed": sum(1 for n in self._tree.values() if n.status == "completed"),
            "failed": sum(1 for n in self._tree.values() if n.status == "failed"),
        }
        (self._engine_dir / "engine_state.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )

    def _persist_all(self) -> None:
        """Persist tree, queue, and active state."""
        self._engine_dir.mkdir(parents=True, exist_ok=True)
        tree_data = [n.to_dict() for n in self._tree.values()]
        (self._engine_dir / "tree.json").write_text(
            json.dumps(tree_data, indent=2) + "\n"
        )
        self._persist_queue()
        self._persist_active()


def _extract_counter(node_id: str) -> int:
    """Extract the numeric prefix from a node_id like '003_baseline'."""
    parts = node_id.split("_", 1)
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return 0
