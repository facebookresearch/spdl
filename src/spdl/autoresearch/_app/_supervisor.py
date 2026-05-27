# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Framework supervisor: parse argv, build engine command, exec coding agent.

The supervisor phase is the entry point an interactive user hits. It
parses framework-level CLI arguments (workdir, agent, platform, etc.),
builds the supervisor prompt by combining framework scaffolding with the
workflow-supplied prompt fragments, builds the engine command line by
appending :py:meth:`WorkflowSpec.engine_argv_tail` to a framework-owned
prefix, and ``execvp``\\s the coding agent so the agent can drive the
engine and watch progress.

The supervisor never imports workflow code directly. It only consumes
the workflow through :py:class:`~spdl.autoresearch.core.WorkflowSpec`.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from spdl.autoresearch._common._supervisor import _resolve_supervisor_agent
from spdl.autoresearch.core import WorkflowSpec

from ._spec import _resolve_workflow

__all__ = [
    "_run_supervisor",
]


def _parse_supervisor_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse the framework supervisor argv.

    Splits arguments at the first ``--`` token: tokens before ``--`` are
    framework flags (workdir, agent, platform, etc.), tokens after
    ``--`` are forwarded to the workflow factory verbatim.

    Args:
        argv: Argv list (excluding ``argv[0]``). Defaults to ``sys.argv[1:]``.

    Returns:
        ``(namespace, workflow_argv)`` where ``namespace`` carries the
        parsed framework flags and ``workflow_argv`` is the workflow tail.
    """
    framework_argv, workflow_argv = _split_at_double_dash(argv)
    parser = _build_parser()
    ns = parser.parse_args(framework_argv)
    return ns, workflow_argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an autoresearch workflow under an interactive supervisor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "workdir",
        nargs="?",
        help="Experiment workdir. Required for fresh runs and resumes.",
    )
    parser.add_argument(
        "--workflow",
        help=(
            "Workflow factory specifier in 'module.path:factory_name' form, "
            "or a short name registered under the "
            "'spdl.autoresearch.workflows' entry-points group."
        ),
    )
    parser.add_argument(
        "--agent",
        choices=("auto", "claude", "codex"),
        default="auto",
        help="Interactive supervisor agent.",
    )
    parser.add_argument(
        "--platform",
        default="auto",
        help="Execution platform: 'auto', a registered provider name, or 'local'.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum concurrent engine coroutines.",
    )
    parser.add_argument(
        "--engine-command",
        help=(
            "Override the engine command shown to the supervisor. Defaults "
            "to '<argv0> engine ...' so the framework binary re-invokes "
            "itself in engine mode."
        ),
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Pass --dangerously-skip-permissions to Claude engine invocations.",
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


def _build_engine_command(
    *,
    engine_command_override: str | None,
    workflow_spec: str,
    workdir: Path,
    framework_flags: list[str],
    workflow_argv_tail: list[str],
) -> list[str]:
    """Assemble the full engine command line.

    The supervisor renders this command into the agent prompt; the agent
    then runs it in the background to start the engine.

    Args:
        engine_command_override: Value of ``--engine-command``, or
            ``None`` to default to ``<argv0> engine``.
        workflow_spec: Workflow specifier to forward via ``--workflow``.
        workdir: Workdir to forward to the engine.
        framework_flags: Framework-owned flags (such as
            ``--max-concurrency 3 --platform auto``).
        workflow_argv_tail: Workflow-specific flags returned by
            :py:meth:`WorkflowSpec.engine_argv_tail`.

    Returns:
        A token list suitable for shell quoting and display.
    """
    if engine_command_override:
        prefix = shlex.split(engine_command_override)
    else:
        prefix = [sys.argv[0], "engine"]
    command = list(prefix)
    command.extend(["--workflow", workflow_spec])
    command.extend(["--workdir", str(workdir)])
    command.extend(framework_flags)
    if workflow_argv_tail:
        command.append("--")
        command.extend(workflow_argv_tail)
    return command


def _build_supervisor_prompt(
    spec: WorkflowSpec,
    *,
    workdir: Path | None,
    engine_command_text: str,
) -> str:
    """Combine framework scaffolding with workflow prompt and known config.

    Args:
        spec: Workflow spec returned by the factory.
        workdir: Workdir resolved from CLI, or ``None`` if not supplied.
        engine_command_text: Pre-rendered engine command (shell-quoted).

    Returns:
        Markdown prompt for the supervisor agent.
    """
    workflow_md = spec.description() or ""
    known = spec.supervisor_known_config()
    missing = spec.supervisor_missing_config()
    known_block = (
        "\n".join(f"- {key}: {value}" for key, value in known.items())
        if known
        else "- (none)"
    )
    missing_block = "\n".join(f"- {item}" for item in missing) if missing else "- none"
    workdir_text = str(workdir) if workdir else "(missing)"
    context = (
        "## Supervisor Launch Context\n\n"
        f"Workdir: {workdir_text}\n\n"
        f"Known configuration:\n{known_block}\n\n"
        f"Missing required configuration:\n{missing_block}\n\n"
        "Engine command template:\n\n"
        f"```bash\n{engine_command_text}\n```\n"
    )
    sections = [section for section in (workflow_md, context) if section]
    return "\n\n---\n\n".join(sections)


def _run_supervisor(argv: list[str] | None = None) -> None:
    """Resolve workflow, render supervisor prompt, exec the coding agent.

    Replaces the current process with the supervisor agent (Claude or
    Codex) via :py:func:`os.execvp`, exactly mirroring the legacy
    ``pipeline_optimization._cli`` path but without any pipeline-opt
    knowledge in the framework.

    Args:
        argv: Argv list (excluding ``argv[0]``). Defaults to
            ``sys.argv[1:]``.
    """
    ns, workflow_argv = _parse_supervisor_args(argv)
    if not ns.workflow:
        raise SystemExit(
            "--workflow is required. Pass module.path:factory_name or a "
            "short name registered under the spdl.autoresearch.workflows "
            "entry-points group."
        )
    workdir = Path(ns.workdir).resolve() if ns.workdir else None
    factory = _resolve_workflow(ns.workflow)
    spec: WorkflowSpec = factory(workflow_argv, workdir)

    supervisor = _resolve_supervisor_agent(ns.agent)
    framework_flags = _supervisor_framework_flags(ns)
    engine_command = _build_engine_command(
        engine_command_override=ns.engine_command,
        workflow_spec=ns.workflow,
        workdir=workdir or Path("(missing)"),
        framework_flags=framework_flags,
        workflow_argv_tail=spec.engine_argv_tail(),
    )
    engine_command_text = " ".join(shlex.quote(part) for part in engine_command)
    prompt = _build_supervisor_prompt(
        spec, workdir=workdir, engine_command_text=engine_command_text
    )
    initial_request = (
        f"Start supervising this autoresearch run.\n\n"
        f"Workdir: {workdir or '(missing)'}\n\n"
        f"Engine command:\n\n```bash\n{engine_command_text}\n```\n"
    )
    command = supervisor.command(prompt, initial_request)
    os.execvp(command[0], command)


def _supervisor_framework_flags(ns: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if ns.max_concurrency is not None:
        flags.extend(["--max-concurrency", str(ns.max_concurrency)])
    if ns.platform:
        flags.extend(["--platform", ns.platform])
    if ns.dangerously_skip_permissions:
        flags.append("--dangerously-skip-permissions")
    return flags
