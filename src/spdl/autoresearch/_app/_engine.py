# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Framework engine: parse argv, build workflow, drive the orchestrator.

The engine phase is what the supervisor (or a developer running directly)
ends up invoking. It parses the framework engine flags, splits off the
workflow argv tail, resolves the workflow factory, calls
:py:meth:`WorkflowSpec.setup` and :py:meth:`WorkflowSpec.build_workflow`,
and runs the :py:class:`~spdl.autoresearch.core.Orchestrator` to
completion.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from spdl.autoresearch.core import Orchestrator, WorkflowSpec

from ._spec import _record_workflow_factory, _resolve_workflow

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_parse_engine_args",
    "_run_engine",
]


def _parse_engine_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse the framework engine argv.

    Splits arguments at the first ``--`` token. Tokens before are
    framework engine flags; tokens after are forwarded to the workflow
    factory.

    Args:
        argv: Argv list (excluding ``argv[0]``). Defaults to ``sys.argv[1:]``.

    Returns:
        ``(namespace, workflow_argv)``.
    """
    framework_argv, workflow_argv = _split_at_double_dash(argv)
    parser = _build_parser()
    ns = parser.parse_args(framework_argv)
    return ns, workflow_argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive the autoresearch orchestrator for a workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workflow",
        required=True,
        help="Workflow factory specifier in 'module.path:factory_name' form.",
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Experiment workdir.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help=(
            "Maximum concurrent engine coroutines. When omitted, the "
            "workflow's WorkflowSpec.max_concurrency is used as the "
            "default."
        ),
    )
    parser.add_argument(
        "--platform",
        default="auto",
        help="Execution platform: 'auto', a registered provider name, or 'local'.",
    )
    return parser


def _split_at_double_dash(
    argv: list[str] | None,
) -> tuple[list[str], list[str]]:
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def _run_engine(argv: list[str] | None = None) -> None:
    """Resolve the workflow, build the spec, and drive the orchestrator.

    On a clean engine exit, the framework writes
    ``<workdir>/report.md`` with the result of
    :py:meth:`WorkflowProtocol.summarize <spdl.autoresearch.core.WorkflowProtocol.summarize>`
    so operators have a final summary without needing to invoke a
    separate command.

    Args:
        argv: Argv list (excluding ``argv[0]``). Defaults to ``sys.argv[1:]``.
    """
    ns, workflow_argv = _parse_engine_args(argv)
    workdir = Path(ns.workdir)
    factory = _resolve_workflow(ns.workflow)
    spec: WorkflowSpec = factory(workflow_argv, workdir)
    # Record the workflow specifier so later operator commands (e.g.
    # ``autoresearch <workdir> summary``) can re-resolve the same factory
    # without requiring the user to repeat ``--workflow ...``. This is a
    # framework concern; the workflow itself does not need to participate.
    _record_workflow_factory(workdir, ns.workflow)
    spec.setup(workdir)
    workflow = spec.build_workflow(workdir)
    max_concurrency = (
        ns.max_concurrency if ns.max_concurrency is not None else spec.max_concurrency
    )
    engine = Orchestrator(workflow=workflow, max_concurrency=max_concurrency)
    asyncio.run(engine.run())
    _write_final_report(workdir, workflow)


def _write_final_report(workdir: Path, workflow: object) -> None:
    """Write the workflow's final summary to ``<workdir>/report.md``.

    Best-effort: if ``summarize`` raises (for example, the workflow's
    persisted state is missing), the engine still exits cleanly and a
    warning is logged rather than crashing the process.
    """
    summarize = getattr(workflow, "summarize", None)
    if summarize is None:
        return
    try:
        report = summarize(workdir)
        (workdir / "report.md").write_text(report)
    except Exception:  # noqa: BLE001 — final-report failures must not crash exit
        _LG.warning(
            "Failed to write final report",
            exc_info=True,
        )
