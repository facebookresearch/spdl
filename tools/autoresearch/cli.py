#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public autoresearch CLI.

This entry point always launches an interactive supervisor agent. The
supervisor gathers missing configuration, starts the lower-level engine, and
keeps diagnostics easy by inspecting workdir state while the engine runs.
``run.py`` remains the non-interactive engine implementation invoked by the
supervisor; it is intentionally not the normal user-facing path.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from spdl.autoresearch.pipeline_optimization._prompts import load_prompt_directory
from spdl.tools.autoresearch.utils.supervisor import (
    _resolve_supervisor_agent,
    _SupervisorAgent,
)

_SCRIPT_DIR = Path(__file__).resolve().parent

__all__ = ["main"]


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "items",
        nargs="*",
        help=(
            "Optional workdir followed by a free-form request. Use --workdir "
            "for relative workdir names that do not look like paths."
        ),
    )
    parser.add_argument("--workdir", dest="workdir_opt", help="Experiment workdir")
    parser.add_argument(
        "--agent",
        choices=("auto", "claude", "codex"),
        default="auto",
        help="Interactive supervisor agent. The engine coding agent defaults to this choice.",
    )
    parser.add_argument(
        "--workflow-agent",
        choices=("claude", "codex", "mock"),
        help="Coding agent used by the engine workflow. Defaults to --agent when explicit.",
    )
    parser.add_argument("--pipeline-script", help="Pipeline script to optimize")
    parser.add_argument("--source-dir", help="Source directory to modify in-place")
    parser.add_argument("--build-command", help="Command to build the job image")
    parser.add_argument(
        "--base-launch-command",
        help="Base job launch command; use $IMAGE for image placeholder",
    )
    parser.add_argument("--notes", help="Free-form notes about the experiment")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--job-timeout", type=int, default=1800)
    parser.add_argument("--platform", default="auto")
    parser.add_argument(
        "--local-execution-mode",
        choices=("full", "dataloader_only", "dry_run"),
        default="full",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Pass --dangerously-skip-permissions to Claude engine invocations",
    )
    parser.add_argument(
        "--skip-instrument",
        action="store_true",
        help="Ask the engine to skip automatic TTFB/step-time instrumentation",
    )
    parser.add_argument(
        "--engine-command",
        help=(
            "Override the engine command shown to the supervisor. Defaults to "
            "the local run.py path."
        ),
    )
    return parser.parse_args(args)


def _looks_like_workdir(value: str) -> bool:
    expanded = os.path.expanduser(value)
    return (
        value.startswith(("/", "./", "../", "~"))
        or os.sep in value
        or Path(expanded).exists()
    )


def _split_workdir_and_request(ns: argparse.Namespace) -> tuple[str | None, str]:
    if ns.workdir_opt:
        return ns.workdir_opt, " ".join(ns.items).strip()
    if not ns.items:
        return None, ""
    if _looks_like_workdir(ns.items[0]):
        return ns.items[0], " ".join(ns.items[1:]).strip()
    return None, " ".join(ns.items).strip()


def _engine_agent(ns: argparse.Namespace, supervisor: _SupervisorAgent) -> str:
    if ns.workflow_agent:
        return str(ns.workflow_agent)
    if ns.agent == "auto":
        return supervisor.name
    return str(ns.agent)


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _append_option(command: list[str], flag: str, value: object | None) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def _build_engine_command(
    ns: argparse.Namespace,
    workdir: str | None,
    supervisor: _SupervisorAgent,
) -> list[str]:
    if ns.engine_command:
        command = shlex.split(ns.engine_command)
    else:
        command = [sys.executable, str(_SCRIPT_DIR / "run.py")]

    if workdir:
        command.append(workdir)

    _append_option(command, "--pipeline-script", ns.pipeline_script)
    _append_option(command, "--source-dir", ns.source_dir)
    _append_option(command, "--build-command", ns.build_command)
    _append_option(command, "--base-launch-command", ns.base_launch_command)
    _append_option(command, "--notes", ns.notes)
    _append_option(command, "--max-iterations", ns.max_iterations)
    _append_option(command, "--patience", ns.patience)
    _append_option(command, "--poll-interval", ns.poll_interval)
    _append_option(command, "--max-concurrency", ns.max_concurrency)
    _append_option(command, "--job-timeout", ns.job_timeout)
    _append_option(command, "--platform", ns.platform)
    _append_option(command, "--agent", _engine_agent(ns, supervisor))
    _append_option(command, "--local-execution-mode", ns.local_execution_mode)
    if ns.dangerously_skip_permissions:
        command.append("--dangerously-skip-permissions")
    if ns.skip_instrument:
        command.append("--skip-instrument")
    return command


def _missing_engine_inputs(ns: argparse.Namespace, workdir: str | None) -> list[str]:
    missing = []
    if not workdir:
        missing.append("workdir")
    for attr, label in (
        ("pipeline_script", "pipeline script"),
        ("source_dir", "source directory"),
        ("build_command", "build command"),
        ("base_launch_command", "launch command template"),
    ):
        if not getattr(ns, attr):
            missing.append(label)
    return missing


def _build_supervisor_context(
    ns: argparse.Namespace,
    workdir: str | None,
    supervisor: _SupervisorAgent,
) -> str:
    engine_command = _build_engine_command(ns, workdir, supervisor)
    missing = _missing_engine_inputs(ns, workdir)
    missing_text = "\n".join(f"- {item}" for item in missing) if missing else "- none"
    known = {
        "workdir": workdir or "",
        "platform": ns.platform,
        "supervisor_agent": supervisor.name,
        "workflow_agent": _engine_agent(ns, supervisor),
        "pipeline_script": ns.pipeline_script or "",
        "source_dir": ns.source_dir or "",
        "build_command": ns.build_command or "",
        "base_launch_command": ns.base_launch_command or "",
        "local_execution_mode": ns.local_execution_mode,
    }
    known_text = "\n".join(f"- {key}: {value}" for key, value in known.items())
    return f"""## Supervisor Launch Context

The public autoresearch CLI has already selected an interactive supervisor
agent. You are that supervisor. Do not run autoresearch directly until required
engine configuration is known.

Known configuration:
{known_text}

Missing required first-run configuration:
{missing_text}

Engine command template:

```bash
{_quote_command(engine_command)}
```

If configuration is missing, ask the user for the missing values first. Once
the configuration is complete, run the engine command in the background so you
can monitor progress and inspect diagnostics. On resume, the workdir alone is
enough because the engine persists config in `config.json`.
"""


def _build_supervisor_prompt(
    ns: argparse.Namespace,
    workdir: str | None,
    supervisor: _SupervisorAgent,
) -> str:
    supervisor_prompt = load_prompt_directory("supervisor")
    platform_prompt = load_prompt_directory("platform")
    context = _build_supervisor_context(ns, workdir, supervisor)
    sections = [supervisor_prompt, platform_prompt, context]
    sections = [section for section in sections if section]
    return "\n\n---\n\n".join(sections)


def _build_initial_prompt(
    ns: argparse.Namespace,
    workdir: str | None,
    supervisor: _SupervisorAgent,
    user_request: str,
) -> str:
    if user_request:
        return user_request

    engine_command = _quote_command(_build_engine_command(ns, workdir, supervisor))
    missing = _missing_engine_inputs(ns, workdir)
    if missing:
        missing_text = ", ".join(missing)
        next_step = (
            "Ask me for the missing required configuration before starting the "
            f"engine: {missing_text}."
        )
    else:
        next_step = (
            "Start the engine in the background with the command below, then "
            "monitor the workdir and report progress."
        )
    return f"""Start supervising this autoresearch run.

Workdir: {workdir or "(missing)"}

{next_step}

Engine command:

```bash
{engine_command}
```
"""


def _run(args: list[str] | None = None) -> None:
    ns = _parse_args(args)
    workdir, user_request = _split_workdir_and_request(ns)
    supervisor = _resolve_supervisor_agent(ns.agent)
    prompt = _build_supervisor_prompt(ns, workdir, supervisor)
    initial_prompt = _build_initial_prompt(ns, workdir, supervisor, user_request)
    command = supervisor.command(prompt, initial_prompt)
    os.execvp(command[0], command)


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
