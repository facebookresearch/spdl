# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Workflow factory specifier resolution and workdir round-tripping.

A workflow specifier is a string of the form ``module.path:factory_name``
(torchx-style). The dispatcher accepts a specifier on the command line
(``--workflow ...``) and resolves it to a callable via
:py:func:`_resolve_workflow`. On a fresh run, the workflow records the
specifier in the workdir via :py:func:`_record_workflow_factory` so later
commands such as ``autoresearch <wd> summary`` can re-instantiate the
workflow without the original CLI args.

Resolution order
================

1. Treat the specifier as ``module.path:factory_name`` and import the
   module via :py:func:`importlib.import_module`. This is the primary
   path; any importable Python callable can be used.
2. If the specifier contains no ``:``, fall back to the entry-points
   group ``spdl.autoresearch.workflows`` for short-name sugar (for
   example, ``--workflow pipeline_optimization``).
"""

from __future__ import annotations

import importlib
import json
from importlib.metadata import entry_points
from pathlib import Path

from spdl.autoresearch.core import WorkflowFactory

__all__ = [
    "_read_workflow_factory",
    "_record_workflow_factory",
    "_resolve_workflow",
]

_ENTRY_POINTS_GROUP: str = "spdl.autoresearch.workflows"
# Entry-points group used for short-name workflow lookup.

_WORKFLOW_FACTORY_FILENAME: str = "workflow_factory.json"
# Filename inside the workdir that records the workflow factory specifier.


def _resolve_workflow(spec: str) -> WorkflowFactory:
    """Resolve a workflow factory specifier to a callable.

    Accepts either ``module.path:factory_name`` (preferred) or a short
    name registered under the
    :py:data:`spdl.autoresearch.workflows <_ENTRY_POINTS_GROUP>` entry
    points group.

    Args:
        spec: Workflow specifier supplied via ``--workflow``.

    Returns:
        A callable matching :py:data:`spdl.autoresearch.core.WorkflowFactory`.

    Raises:
        ValueError: If ``spec`` is empty or syntactically invalid.
        ModuleNotFoundError: If the named module cannot be imported.
        AttributeError: If the named factory attribute does not exist.
        LookupError: If a short name does not match any registered entry
            point.
    """
    if not spec:
        raise ValueError("workflow specifier must be a non-empty string")
    if ":" in spec:
        module_path, _, factory_name = spec.partition(":")
        if not module_path or not factory_name:
            raise ValueError(
                f"workflow specifier {spec!r} must be 'module.path:factory_name'"
            )
        module = importlib.import_module(module_path)
        try:
            factory = getattr(module, factory_name)
        except AttributeError as exc:
            raise AttributeError(
                f"module {module_path!r} has no attribute {factory_name!r}"
            ) from exc
        return factory  # type: ignore[no-any-return]
    return _resolve_via_entry_points(spec)


def _resolve_via_entry_points(short_name: str) -> WorkflowFactory:
    eps = entry_points(group=_ENTRY_POINTS_GROUP)
    matches = [ep for ep in eps if ep.name == short_name]
    if not matches:
        raise LookupError(
            f"no workflow registered as {short_name!r} under entry-points "
            f"group {_ENTRY_POINTS_GROUP!r}"
        )
    return matches[0].load()  # type: ignore[no-any-return]


def _record_workflow_factory(workdir: Path, spec: str) -> None:
    """Record the workflow factory specifier inside the workdir.

    Writes ``<workdir>/workflow_factory.json`` so that subsequent
    invocations targeting the same workdir (for example,
    ``autoresearch <wd> summary``) can re-resolve the workflow without
    requiring the original ``--workflow`` argument.

    Args:
        workdir: Directory the workflow run targets.
        spec: Workflow specifier in the same form accepted by
            :py:func:`_resolve_workflow`.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / _WORKFLOW_FACTORY_FILENAME
    path.write_text(json.dumps({"spec": spec}, indent=2) + "\n")


def _read_workflow_factory(workdir: Path) -> str | None:
    """Read the workflow factory specifier previously recorded in ``workdir``.

    Args:
        workdir: Directory the workflow run targets.

    Returns:
        The recorded specifier, or ``None`` if no record exists.

    Raises:
        ValueError: If the record file exists but is malformed.
    """
    path = workdir / _WORKFLOW_FACTORY_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    spec = data.get("spec")
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"{path} must contain a non-empty 'spec' string")
    return spec
