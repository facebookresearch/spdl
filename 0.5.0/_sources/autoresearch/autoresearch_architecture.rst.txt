.. _autoresearch-architecture:

Architecture
============

.. note::

   This section describes internal implementation details of the
   autoresearch engine. The architecture may change at any time
   without notice.

The implementation is organized into several modules with clear
boundaries. The goal is to keep scheduling generic, domain logic
testable, and infrastructure swappable.

.. mermaid::

   flowchart TB
       CLI["spdl autoresearch"]
       Supervisor["Supervisor agent<br>(interactive)"]
       Engine["Orchestrator<br>(core/_orchestrator.py)"]
       Workflow["PipelineOptimizationWorkflow<br>(pipeline_optimization/)"]
       Policy["_policy.py"]
       Store["_store.py"]
       Platform["AutoresearchPlatform<br>(pipeline_optimization/_platform/)"]
       Agent["_CodingAgent"]

       CLI --> Supervisor
       Supervisor --> Engine
       Engine --> Workflow
       Workflow --> Policy
       Workflow --> Store
       Workflow --> Platform
       Platform --> Agent


Design Principles
------------------

Several design choices cut across the modules and are not obvious from
the code alone.

**The engine must remain domain-neutral.** The async work engine must
not grow SPDL, coding-agent, source-control, metrics,
hypothesis-planning, or experiment-phase logic. If a behavior depends
on what an experiment is, it belongs in the workflow, the
policy module, or behind a platform capability — never in the runner.

**Stop criteria live in the planner, not the engine.** The engine
stops when the queue and running set are both empty. Autoresearch
enforces its own stopping conditions (plateau patience, max iterations,
all best practices tried) by returning no children from the planning
step. This keeps the engine simple and avoids a domain-specific
``should_stop`` callback.

**Resume is phase-based.** Each experiment coroutine persists its phase
(queued, preparing, running, analyzing, completed, failed) at every
meaningful boundary. On resume, the coroutine inspects the persisted
phase to skip already-completed steps: a ``running`` experiment with a
known job ID resumes polling rather than re-launching, and an
``analyzing`` experiment skips straight to analysis.

**Domain coroutines own their cancellation behavior.** The engine
cancels asyncio tasks on ``SIGINT``/``SIGTERM``, but each coroutine
decides what state to persist before re-raising ``CancelledError``.
Remote jobs are not automatically cancelled by the engine — if remote
cancellation is needed, the coroutine or workflow must do it explicitly.

**Failures are structured domain data.** Every failure path (prepare,
build, launch, poll, analyze, plan) produces a ``FailureRecord`` with a
``FailureKind`` and ``FailurePhase``. The runner never learns about
failure kinds. Expected failures flow through ``AutoresearchError``;
unexpected exceptions are caught and wrapped into structured records.
This ensures durable accounting even for phases that never reach a
remote job.


Async Work Engine
-----------------

The generic runner (``core/_orchestrator.py``) knows nothing about SPDL,
training jobs, source control, metrics, or hypothesis planning. It
operates on serializable ``WorkSpec`` objects and a ``WorkflowProtocol``
protocol:

- Maintains a priority queue of pending ``WorkSpec`` objects.
- Starts up to ``max_concurrency`` coroutines via the workflow.
- Waits for the first coroutine to complete.
- Passes completed results (which may contain child specs) back to the
  workflow and re-queues children.
- Checkpoints queued and running specs on cancellation.

The runner does not inspect experiment payloads. Infrastructure-specific
work belongs in the platform capability layer, and domain decisions
belong in the workflow.


Workflow
--------

The pipeline optimization workflow
(``pipeline_optimization/_ops/``) is the domain side of the boundary.
It turns an experiment ``WorkSpec`` into a coroutine that performs the
full experiment lifecycle:

- Restore or prepare the source tree.
- Apply code changes when the experiment requires a rebuild.
- Build the image and launch the remote job.
- Poll for completion and detect stalled jobs.
- Collect metrics and run coding agent analysis.
- Record state, master-table rows, findings, and plots.
- Ask the coding agent for follow-up experiments and return them as
  child ``WorkSpec`` objects.

The workflow is split into focused modules:

- **_workflow.py** -- the ``PipelineOptimizationWorkflow`` that
  implements ``WorkflowProtocol`` and orchestrates experiment
  coroutines.
- **_ops/_policy.py** -- deterministic decisions (planning gates,
  duplicate filtering, stall detection) expressed as pure functions
  that can be unit tested without infrastructure.
- **_ops/_store.py** -- durable state persistence (master table,
  findings, tree visualization).
- **_ops/_analysis_ops.py** / **_ops/_planning_ops.py** /
  **_ops/_source_ops.py** -- individual workflow operations that
  interact with the platform.


Platform Capabilities
---------------------

The platform layer (``pipeline_optimization/_platform/``) provides a
capability boundary between the workflow and infrastructure.
``AutoresearchPlatform`` bundles capability objects for:

- **Workspace** -- source control operations (detect SCM, commit,
  restore, check for changes).
- **Artifacts** -- image building and tagging.
- **Execution** -- job launch, status polling, and cancellation.
- **Evidence** -- metrics collection and system profiling.
- **CodingAgent** -- stateless coding agent invocations (analysis,
  planning, code changes).

The workflow can swap local, remote, Claude, Codex, or test
implementations by replacing these capability objects without changing
any orchestration code.

.. mermaid::

   flowchart LR
       Workflow["PipelineOptimizationWorkflow"]
       Platform["AutoresearchPlatform"]
       Workspace["Workspace"]
       Artifacts["Artifacts"]
       Execution["Execution"]
       Evidence["Evidence"]
       Agent["CodingAgent"]

       Workflow --> Platform
       Platform --> Workspace
       Platform --> Artifacts
       Platform --> Execution
       Platform --> Evidence
       Platform --> Agent


Stateless Agent Invocations
---------------------------

Each coding agent call is fully stateless. The workflow constructs a
self-contained prompt that includes everything the agent needs: the
SPDL optimization knowledge base, the full experiment history,
collected metrics, and the pipeline source code. There is no persistent
conversation or session state.

This design makes the system robust to interruptions. After ``Ctrl+C``,
the engine can resume from the last persisted checkpoint without
relying on a conversation session. It also means the coding agent can
be swapped between runs (e.g., switching from Claude to Codex) with
no state migration.


Hypothesis Tree
---------------

Experiments are organized in a tree structure. The seed experiments
(baseline, headspace, MTP) are root nodes. Follow-up experiments
proposed by the coding agent become children of the node that triggered
the planning.

.. code-block:: text

   baseline
   headspace
   mtp
   ├── gpu_nvdec_decode
   │   ├── split_demux_decode
   │   │   └── nvdec_c7_optimal
   │   └── nvdec_c20_oversub
   ├── batch_size_16
   └── torch_compile

Each node tracks its status (queued, preparing, running, analyzing,
completed, failed), the source control commit it was built from, and
the analysis results. The tree is owned by the workflow store and
visualized as ``hypothesis_tree.png`` after each experiment completes.

The following is the hypothesis tree from the
:ref:`video classification example <autoresearch-example>`, showing
116 nodes explored across 120 experiments:

.. image:: /_static/data/autoresearch_video_classification_hypothesis_tree.png
   :alt: Hypothesis tree from video classification optimization
   :width: 100%
