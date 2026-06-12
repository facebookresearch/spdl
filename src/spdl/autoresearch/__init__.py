# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Framework for automated, LLM-driven experiment workflows.

.. versionadded:: 0.5.0

Autoresearch is an automated experiment engine that uses a coding agent
to analyze metrics, identify bottlenecks, propose parameter and code
changes, and iterate toward an objective (for the SPDL pipeline
optimization workflow, minimizing steady-state step time).

Execution model
---------------

The system is split into two layers:

- The public CLI (``spdl autoresearch``) is always interactive. A
  supervisor agent gathers missing configuration from the user, starts
  the engine, monitors progress, and intervenes on errors.
- The engine command is a non-interactive implementation detail. The
  supervisor invokes it with complete config; tests can call it
  directly.

Design principles
-----------------

The architecture is shaped by a few deliberate choices that are not
obvious from reading the code:

- **Generic, domain-agnostic runner.** The runner only schedules
  serializable ``WorkSpec`` coroutines, checkpoints queued and running
  work, and persists an ``interrupted`` checkpoint when cancellation
  reaches the process. It knows nothing about SPDL, coding agents,
  source control, jobs, metrics, or planning. All experiment semantics
  (apply code change, build, launch, poll, analyze, propose follow-ups)
  live in the workflow side of the boundary, and deterministic
  decisions are split into testable policy helpers so they can be
  exercised without a coding agent or live infrastructure.

- **Stateless coding-agent calls.** The engine carries all state and
  assembles self-contained prompts on each invocation from compiled
  knowledge, experiment state, and fetched metrics. Agents are never
  expected to remember anything across calls.

- **Capabilities, not callbacks.** Platform behavior is grouped behind
  explicit capability objects — ``workspace``, ``artifacts``,
  ``execution``, ``evidence``, ``agent`` — instead of a flat callback
  bag. The same workflow can run locally, through a remote executor,
  or with a different coding agent without changing the runner.

- **Persisted state is the source of truth.** Files under ``engine/``
  drive crash recovery and external observability. Ctrl+C persists
  queued and running specs to disk; re-running the same command
  resumes from where it left off, including re-checking the status of
  jobs that kept running on the cluster.

- **Tree-structured experiments.** Experiments branch from parent
  commits, enabling exploration of multiple optimization paths
  concurrently up to a configurable concurrency limit. When a node
  completes, its analysis is immediate and follow-ups are enqueued
  before other in-flight jobs finish.

Layout
------

- :py:mod:`~spdl.autoresearch.core`: the domain-neutral async work
  scheduler and the
  :py:class:`~spdl.autoresearch.core.WorkflowProtocol` /
  :py:class:`~spdl.autoresearch.core.WorkflowSpec` contracts that
  pluggable workflows implement.
- :py:mod:`~spdl.autoresearch.pipeline_optimization`: concrete
  workflow implementation for SPDL data loading pipeline optimization.
"""

__all__: list[str] = []
