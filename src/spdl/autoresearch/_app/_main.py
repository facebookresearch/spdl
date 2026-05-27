# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level entry point for the autoresearch framework binary.

The single ``spdl-autoresearch`` binary supports three subcommands,
parsed by :mod:`argparse`:

- ``engine`` — run the orchestrator directly. Used by the supervisor
  when it execs the engine subprocess, and by developers running the
  engine without the agent wrapper.
- ``summary <workdir>`` — re-resolve the workflow factory recorded in
  ``<workdir>/workflow_factory.json`` and print
  ``workflow.summarize(workdir)`` to stdout.
- (default, no subcommand) — run the interactive supervisor wrapper,
  which builds the engine command and ``execvp``\\s a coding agent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spdl.autoresearch._app._spec import _read_workflow_factory, _resolve_workflow

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spdl autoresearch",
        description="Autoresearch framework dispatcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "engine",
        help="Run the orchestrator engine directly.",
        add_help=False,
    )

    summary_parser = subparsers.add_parser(
        "summary",
        help="Print the workflow summary for an existing workdir.",
    )
    summary_parser.add_argument(
        "workdir",
        help="Experiment workdir containing workflow_factory.json.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the supervisor, engine, or summary entry point.

    Args:
        argv: Argv list (excluding ``argv[0]``). Defaults to ``sys.argv[1:]``.
    """
    from ._engine import _run_engine
    from ._supervisor import _run_supervisor

    args = list(sys.argv[1:] if argv is None else argv)
    ns, remaining = _build_parser().parse_known_args(args)

    if ns.command == "engine":
        _run_engine(remaining)
    elif ns.command == "summary":
        _run_summary(ns.workdir)
    else:
        _run_supervisor(args)


def _run_summary(workdir_str: str) -> None:
    """Print the workflow's summary for an existing workdir."""
    workdir = Path(workdir_str).resolve()
    spec_str = _read_workflow_factory(workdir)
    if spec_str is None:
        raise SystemExit(
            f"No workflow factory recorded in {workdir}. "
            f"Run the engine against this workdir first."
        )
    factory = _resolve_workflow(spec_str)
    spec = factory([], workdir)
    workflow = spec.build_workflow(workdir)
    print(workflow.summarize(workdir))
