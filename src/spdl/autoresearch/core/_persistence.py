# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Default JSON-backed engine-state persistence helpers.

The :py:class:`~spdl.autoresearch.core.Orchestrator` lets each workflow
own how it persists queued/running specs and run status. Most workflows
need the same straightforward thing: write the lists to a JSON file in
the workdir on every checkpoint, and read them back on restart.

This module provides that default. A workflow's ``checkpoint`` and
``load`` methods can call :py:func:`write_engine_state` and
:py:func:`load_or_init` directly; everything else (domain-specific
state, monitoring views, summaries) stays in the workflow.

Example::

    from spdl.autoresearch.core import (
        TaskSpec, WorkflowProtocol,
        load_or_init, write_engine_state,
    )

    class MyWorkflow(WorkflowProtocol):
        def __init__(self, workdir):
            self.workdir = workdir

        def load(self) -> list[TaskSpec]:
            return load_or_init(self.workdir, self._initial_specs)

        def checkpoint(self, queued, running, status) -> None:
            write_engine_state(
                self.workdir,
                queued=queued,
                running=running,
                status=status,
            )

        def _initial_specs(self) -> list[TaskSpec]:
            return [TaskSpec(id="exp_001")]
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from ._orchestrator import TaskSpec

__all__ = [
    "load_or_init",
    "read_engine_state",
    "write_engine_state",
]


_ENGINE_STATE_FILENAME = "engine_state.json"


def write_engine_state(
    workdir: Path,
    *,
    queued: list[TaskSpec],
    running: list[TaskSpec],
    status: str,
) -> None:
    """Persist the orchestrator's queued, running, and status fields.

    Writes ``<workdir>/engine_state.json`` (creating ``<workdir>`` if
    needed). Specs are serialized via :py:meth:`TaskSpec.to_dict`, so
    they round-trip through :py:func:`read_engine_state`.

    The write is atomic from a reader's perspective: the JSON payload
    is written to a sibling temp file under ``<workdir>`` and then
    moved into place via :py:func:`os.replace`. If the process is
    interrupted mid-write, ``engine_state.json`` either still holds
    the previous valid checkpoint or — on the very first checkpoint —
    is simply absent. It is never observed truncated or partially
    written, so :py:func:`read_engine_state` cannot fail with
    ``json.JSONDecodeError`` from corrupt content on resume.

    Args:
        workdir: Directory in which to persist the JSON file.
        queued: Specs currently waiting in the priority queue.
        running: Specs currently scheduled as coroutines.
        status: One of ``"running"``, ``"stopped"``, or ``"interrupted"``,
            matching the values
            :py:meth:`Orchestrator.run <spdl.autoresearch.core.Orchestrator.run>`
            passes to ``checkpoint``.

    Returns:
        ``None``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "queued": [spec.to_dict() for spec in queued],
        "running": [spec.to_dict() for spec in running],
    }
    path = workdir / _ENGINE_STATE_FILENAME
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_engine_state(
    workdir: Path,
) -> tuple[list[TaskSpec], list[TaskSpec], str] | None:
    """Read the orchestrator's queued, running, and status fields.

    Inverse of :py:func:`write_engine_state`. Returns ``None`` if no
    engine-state file exists in ``workdir`` (a fresh run).

    Args:
        workdir: Directory previously passed to
            :py:func:`write_engine_state`.

    Returns:
        A ``(queued, running, status)`` tuple when the file exists, or
        ``None`` for a fresh workdir.

    Raises:
        ValueError: If the file is malformed or contains entries that
            are not ``TaskSpec`` objects.
    """
    path = workdir / _ENGINE_STATE_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    queued = _specs_from_field(data, "queued", path)
    running = _specs_from_field(data, "running", path)
    status = str(data.get("status", "running"))
    return queued, running, status


def load_or_init(
    workdir: Path,
    initial_factory: Callable[[], list[TaskSpec]],
) -> list[TaskSpec]:
    """Resume from checkpoint, or seed initial specs on a fresh run.

    A drop-in body for a workflow's ``load()`` method::

        def load(self) -> list[TaskSpec]:
            return load_or_init(self.workdir, self._initial_specs)

    Args:
        workdir: Directory previously passed to
            :py:func:`write_engine_state`.
        initial_factory: Zero-argument callable that builds the starting
            specs for a fresh run. Only invoked when no engine-state
            file is present.

    Returns:
        Queued specs concatenated with running specs (the running specs
        re-enter the queue on resume so the orchestrator can re-launch
        them), or whatever ``initial_factory()`` returns.
    """
    state = read_engine_state(workdir)
    if state is None:
        return initial_factory()
    queued, running, _ = state
    return queued + running


def _specs_from_field(
    data: dict[str, object],
    field: str,
    path: Path,
) -> list[TaskSpec]:
    raw = data.get(field, [])
    if not isinstance(raw, list):
        raise ValueError(f"{path} field {field!r} must be a list")
    specs: list[TaskSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path} {field}[{index}] must be a TaskSpec object")
        specs.append(TaskSpec.from_dict(item))
    return specs
