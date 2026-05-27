# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pluggable-workflow contract for the autoresearch framework dispatcher.

The framework dispatcher (``spdl.autoresearch._app``) drives two phases:

1. **Supervisor phase** — a coding-agent process is launched against a
   markdown prompt. The agent collects missing config, then ``execvp``s
   the engine binary.
2. **Engine phase** — the orchestrator drives a
   :py:class:`~spdl.autoresearch.core.WorkflowProtocol` adapter to
   completion.

A workflow factory (``Callable[[list[str], Path | None], WorkflowSpec]``)
returns a :py:class:`WorkflowSpec` that exposes both phases. The
framework calls supervisor-phase methods to render the supervisor prompt
and the engine command line, and engine-phase methods (``setup`` then
``build_workflow``) once the engine starts.

This module only defines the contract. Concrete workflows live elsewhere
(for example, ``spdl.autoresearch.pipeline_optimization``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeAlias

from ._orchestrator import WorkflowProtocol

__all__ = [
    "WorkflowFactory",
    "WorkflowSpec",
]


class WorkflowSpec(Protocol):
    """Pluggable workflow contract returned by a workflow factory.

    Combines supervisor-phase metadata (used by the framework to render
    the supervisor prompt and build the engine command) with engine-phase
    lifecycle hooks (used to initialise the workdir and build the
    :py:class:`~spdl.autoresearch.core.WorkflowProtocol` adapter).

    A workflow factory is any callable matching :py:data:`WorkflowFactory`
    that returns a value satisfying this protocol.

    """

    @property
    def max_concurrency(self) -> int:
        """Maximum concurrent coroutines the orchestrator should run."""
        ...

    # --- Supervisor phase (consumed by framework supervisor) ---

    def engine_argv_tail(self) -> list[str]:
        """Return the workflow-specific flags for the engine command.

        Appended after the framework's own arguments when the supervisor
        builds the engine invocation. The returned list typically encodes
        domain config such as ``--build-command`` or ``--source-dir``.

        Returns:
            A list of CLI tokens to forward to the engine subprocess.
        """
        ...

    def description(self) -> str | None:
        """Return free-form domain context for the supervisor.

        Describes the workflow domain, the meaning of its config
        fields, and what happens during a run. The framework joins this
        with its own supervisor instructions and the launch context
        (known config, missing config, engine command).

        This is NOT the place for supervisor instructions (monitoring,
        troubleshooting, SIGINT handling, status meanings, queue
        surgery). Those are framework-owned and identical across every
        workflow. Use :py:meth:`supervisor_known_config` /
        :py:meth:`supervisor_missing_config` for the structured "what
        to ask the user" surface; use this method for the prose around
        those bullets — what each field means, what good values look
        like, what the engine will do with them.

        Return ``None`` (or an empty string) to contribute nothing.

        Returns:
            Markdown context, or ``None``.
        """
        ...

    def supervisor_known_config(self) -> dict[str, str]:
        """Return key/value pairs for the supervisor's "known config" block.

        Rendered as a bullet list inside the supervisor prompt so the
        agent can summarise what is already set. Empty values are still
        rendered to make missing-but-named fields visible.

        Returns:
            A mapping of config name to display value.
        """
        ...

    def supervisor_missing_config(self) -> list[str]:
        """Return human-readable names of required values not yet supplied.

        An empty list signals the workflow is ready to run the engine
        without further user interaction.

        Returns:
            A list of missing-config labels (for example,
            ``["pipeline script", "build command"]``).
        """
        ...

    # --- Engine phase (consumed by framework engine) ---

    def setup(self, workdir: Path) -> None:
        """Perform fresh-run workdir initialisation, idempotent on resume.

        Called once at the start of each engine invocation. Workflows
        detect re-runs internally (for example, by checking for
        ``config.json``) and should be safe to call repeatedly. Fresh
        runs typically record the factory spec in the workdir so later
        commands can re-instantiate the workflow without the original
        argv.

        Args:
            workdir: Directory the engine will run against.
        """
        ...

    def build_workflow(self, workdir: Path) -> WorkflowProtocol:
        """Construct the engine-side adapter.

        Called after :py:meth:`setup`. The returned object is what the
        :py:class:`~spdl.autoresearch.core.Orchestrator` drives.

        Args:
            workdir: Directory the engine will run against.

        Returns:
            An object satisfying
            :py:class:`~spdl.autoresearch.core.WorkflowProtocol`.
        """
        ...


WorkflowFactory: TypeAlias = Callable[[list[str], "Path | None"], WorkflowSpec]
# Callable that constructs a :py:class:`WorkflowSpec`.

# Receives ``(argv, workdir)`` where ``argv`` is the workflow-owned tail of
# the framework CLI (everything after ``--``) and ``workdir`` may be
# ``None`` during the supervisor phase, before the user has supplied a
# workdir.
