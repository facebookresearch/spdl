# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Workflow factory for the SPDL pipeline-optimization autoresearch workflow.

Exposes :py:func:`create_workflow`, a
:py:data:`~spdl.autoresearch.core.WorkflowFactory` that the framework
dispatcher resolves via ``--workflow
spdl.autoresearch.pipeline_optimization:create_workflow``.

The factory parses pipeline-optimization-specific flags
(``--pipeline-script``, ``--build-command``, ``--base-launch-command``,
``--source-dir``, ``--notes``, etc.), records the workflow factory
specifier in the workdir on a fresh run, performs first-run workdir
initialisation and pipeline instrumentation when ``setup()`` is called,
and constructs a
:py:class:`~spdl.autoresearch.pipeline_optimization._ops._adapter.PipelineOptimizationWorkflow`
adapter on demand.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from spdl.autoresearch._common._log import setup_logging
from spdl.autoresearch._common._state import (
    read_config,
    read_state,
    SCHEMA_VERSION,
    write_state,
)
from spdl.autoresearch.core import (
    FailureKind,
    FailurePhase,
    FailureRecord,
    WorkflowProtocol,
    WorkflowSpec,
)
from spdl.autoresearch.pipeline_optimization._ops._analysis_ops import (
    MASTER_TABLE_HEADERS,
)

from ._ops import PipelineOptimizationWorkflow
from ._ops._failures import _make_failure
from ._platform import AutoresearchPlatform, create_platform
from ._prompts import load_prompt_directory

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "create_workflow",
]


def create_workflow(
    argv: list[str],
    workdir: Path | None,
) -> WorkflowSpec:
    """Build a :py:class:`~spdl.autoresearch.core.WorkflowSpec` from argv.

    Args:
        argv: Workflow-specific argv tail forwarded by the framework
            dispatcher (everything after ``--`` on the framework CLI).
        workdir: Workdir resolved by the framework, or ``None`` during
            the supervisor phase before the user has supplied one.

    Returns:
        A :py:class:`~spdl.autoresearch.core.WorkflowSpec` exposing the
        supervisor- and engine-phase hooks the framework dispatcher
        consumes.
    """
    ns = _parse_args(argv)
    return _PipelineOptimizationSpec(ns, workdir)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spdl.autoresearch.pipeline_optimization",
        description=(
            "Workflow-specific arguments for the SPDL pipeline-optimization "
            "autoresearch workflow."
        ),
    )
    parser.add_argument("--pipeline-script", help="Pipeline script to optimize.")
    parser.add_argument("--build-command", help="Command to build the job image.")
    parser.add_argument(
        "--base-launch-command",
        help="Base job launch command. Use $IMAGE for the image placeholder.",
    )
    parser.add_argument(
        "--source-dir",
        help="Source directory containing the pipeline code to modify in place.",
    )
    parser.add_argument("--notes", help="Free-form notes about the experiment.")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--job-timeout", type=int, default=1800)
    parser.add_argument(
        "--platform",
        default="auto",
        help=(
            "Job execution platform provider. 'auto' uses the best available "
            "provider discovered in this environment."
        ),
    )
    parser.add_argument(
        "--local-execution-mode",
        choices=("full", "dataloader_only", "dry_run"),
        default="full",
        help="How the local platform launches experiment commands.",
    )
    parser.add_argument(
        "--skip-instrument",
        action="store_true",
        help="Skip automatic TTFB/step-time instrumentation on fresh runs.",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Pass --dangerously-skip-permissions to Claude invocations.",
    )
    parser.add_argument(
        "--agent",
        choices=("claude", "codex"),
        default="claude",
        help="Coding agent used by the engine workflow.",
    )
    return parser.parse_args(argv)


class _PipelineOptimizationSpec:
    """Concrete :py:class:`~spdl.autoresearch.core.WorkflowSpec` for pipeline opt.

    Carries the parsed argparse namespace plus the workdir resolved by
    the framework. Implements supervisor-phase metadata methods
    (engine argv tail, supervisor prompt, known/missing config) and
    engine-phase lifecycle hooks (``setup``, ``build_workflow``).
    """

    def __init__(
        self,
        ns: argparse.Namespace,
        workdir: Path | None,
    ) -> None:
        self._ns = ns
        self._workdir = workdir
        self._max_concurrency: int = ns.max_concurrency

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    # --- Supervisor phase ---

    def engine_argv_tail(self) -> list[str]:
        """Render the workflow-specific tail of the engine command."""
        tokens: list[str] = []
        for flag, value in (
            ("--pipeline-script", self._ns.pipeline_script),
            ("--build-command", self._ns.build_command),
            ("--base-launch-command", self._ns.base_launch_command),
            ("--source-dir", self._ns.source_dir),
            ("--notes", self._ns.notes),
            ("--max-iterations", self._ns.max_iterations),
            ("--patience", self._ns.patience),
            ("--poll-interval", self._ns.poll_interval),
            ("--max-concurrency", self._ns.max_concurrency),
            ("--job-timeout", self._ns.job_timeout),
            ("--platform", self._ns.platform),
            ("--local-execution-mode", self._ns.local_execution_mode),
            ("--agent", self._ns.agent),
        ):
            if value is None or value == "":
                continue
            tokens.extend([flag, str(value)])
        if self._ns.dangerously_skip_permissions:
            tokens.append("--dangerously-skip-permissions")
        if self._ns.skip_instrument:
            tokens.append("--skip-instrument")
        return tokens

    def description(self) -> str | None:
        """Concatenate the supervisor and platform prompt directories."""
        sections = [
            section
            for section in (
                load_prompt_directory("supervisor"),
                load_prompt_directory("platform"),
            )
            if section
        ]
        return "\n\n---\n\n".join(sections) if sections else None

    def supervisor_known_config(self) -> dict[str, str]:
        return {
            "pipeline_script": self._ns.pipeline_script or "",
            "source_dir": self._ns.source_dir or "",
            "build_command": self._ns.build_command or "",
            "base_launch_command": self._ns.base_launch_command or "",
            "local_execution_mode": self._ns.local_execution_mode,
        }

    def supervisor_missing_config(self) -> list[str]:
        missing: list[str] = []
        for attr, label in (
            ("pipeline_script", "pipeline script"),
            ("source_dir", "source directory"),
            ("build_command", "build command"),
            ("base_launch_command", "launch command template"),
        ):
            if not getattr(self._ns, attr):
                missing.append(label)
        return missing

    # --- Engine phase ---

    def setup(self, workdir: Path) -> None:
        """Initialise the workdir on a fresh run; idempotent on resume.

        On a fresh run, runs first-time workdir initialisation (and
        pipeline instrumentation unless ``--skip-instrument``). The
        framework records the workflow factory specifier in the workdir
        separately, before this method is called.
        """
        config_file = workdir / "config.json"
        is_fresh = not config_file.exists()

        platform = create_platform(
            {
                "platform": self._ns.platform,
                "agent": self._ns.agent,
                "local_execution_mode": self._ns.local_execution_mode,
                "source_dir": self._ns.source_dir or "",
            },
            workdir,
        )

        if is_fresh:
            if not self._ns.base_launch_command:
                raise SystemExit(
                    "Fresh run requires --base-launch-command on the workflow tail."
                )
            self._init_fresh(workdir, platform)
        setup_logging(workdir)

    def build_workflow(self, workdir: Path) -> WorkflowProtocol:
        """Construct the engine-side adapter."""
        config = read_config(workdir)
        state = read_state(workdir)
        platform = create_platform(config, workdir)
        state["status"] = "looping"
        write_state(workdir, state)
        self._max_concurrency = int(config.get("max_concurrency", 3))
        return PipelineOptimizationWorkflow(
            workdir=workdir,
            config=config,
            state=state,
            platform=platform,
        )

    def _init_fresh(self, workdir: Path, platform: AutoresearchPlatform) -> None:
        _init_workdir(self._ns, workdir, platform)
        if not self._ns.skip_instrument:
            config = read_config(workdir)
            _instrument_pipeline(workdir, config, platform)
            scm_type = config.get("scm", "")
            source_dir = config.get("source_dir", "")
            if scm_type and source_dir:
                state = read_state(workdir)
                state["anchor_commit"] = platform.workspace.current(
                    scm_type,
                    source_dir,
                )
                write_state(workdir, state)


def _record_setup_failure(
    workdir: Path,
    state: dict,
    failure: FailureRecord,
) -> None:
    state.setdefault("_setup_failures", []).append(failure.to_dict())
    engine_dir = workdir / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    setup_failures = engine_dir / "setup_failures.json"
    setup_failures.write_text(json.dumps(state["_setup_failures"], indent=2) + "\n")
    write_state(workdir, state)


def _init_workdir(
    ns: argparse.Namespace,
    workdir: Path,
    platform: AutoresearchPlatform,
) -> dict:
    """Initialise a fresh workdir with config, state, master table, and seed files."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "runs").mkdir(exist_ok=True)
    (workdir / "logs").mkdir(exist_ok=True)

    source_dir = os.path.abspath(ns.source_dir) if ns.source_dir else ""
    scm_type = ""
    anchor_commit_hash = ""
    setup_failures = []
    if source_dir:
        try:
            scm_type = platform.workspace.detect(source_dir)
            anchor_commit_hash = platform.workspace.current(scm_type, source_dir)
            _LG.info("SCM: %s, anchor commit: %s", scm_type, anchor_commit_hash[:12])
        except Exception as error:  # noqa: BLE001 — record and continue
            failure = _make_failure(
                FailureKind.SETUP_SOURCE_FAILED,
                FailurePhase.SETUP,
                "Failed to inspect source control state during setup",
                exception=error,
            )
            setup_failures.append(failure.to_dict())
            _LG.warning("Continuing without source control anchor: %s", error)

    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "pipeline_script": (
            os.path.abspath(ns.pipeline_script) if ns.pipeline_script else None
        ),
        "source_dir": source_dir,
        "scm": scm_type,
        "build_command": ns.build_command or "",
        "base_launch_command": ns.base_launch_command or "",
        "notes": ns.notes or "",
        "stopping_criteria": {
            "max_iterations": ns.max_iterations,
            "patience": ns.patience,
        },
        "max_concurrency": getattr(ns, "max_concurrency", 3),
        "job_timeout_s": ns.job_timeout,
        "poll_interval": ns.poll_interval,
        "startup_failure_retries": 2,
        "startup_retryable_experiments": ["mtp"],
        "platform": ns.platform,
        "agent": ns.agent,
        "local_execution_mode": ns.local_execution_mode,
        "claude_flags": (
            ["--dangerously-skip-permissions"]
            if ns.dangerously_skip_permissions
            else []
        ),
    }
    (workdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    state = {
        "schema_version": SCHEMA_VERSION,
        "iteration": 0,
        "status": "initialized",
        "baseline_job": None,
        "current_best": None,
        "best_metric": None,
        "plateau_count": 0,
        "best_practices_tried": [],
        "cached_image": None,
        "anchor_commit": anchor_commit_hash,
        "history": [],
        "_setup_failures": setup_failures,
    }
    write_state(workdir, state)
    if setup_failures:
        engine_dir = workdir / "engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        (engine_dir / "setup_failures.json").write_text(
            json.dumps(setup_failures, indent=2) + "\n"
        )

    (workdir / "master_table.tsv").write_text("\t".join(MASTER_TABLE_HEADERS) + "\n")

    if ns.pipeline_script and os.path.exists(ns.pipeline_script):
        shutil.copy2(ns.pipeline_script, workdir / "pipeline.py")

    print(f"Initialized experiment at {workdir}")
    return config


def _instrument_pipeline(
    workdir: Path,
    config: dict,
    platform: AutoresearchPlatform,
) -> None:
    """Add TTFB/step-time instrumentation to the pipeline script via the agent."""
    pipeline_script = config.get("pipeline_script", "")
    if not pipeline_script or not Path(pipeline_script).exists():
        return

    pipeline_code = Path(pipeline_script).read_text()
    _LG.info("Instrumenting %s with TTFB/step-time logging", pipeline_script)
    print("Instrumenting pipeline with TTFB/step-time logging...")

    prompt = platform.agent._load_prompt(
        "instrument",
        PIPELINE_SCRIPT=pipeline_script,
        PIPELINE_CODE=pipeline_code,
    )
    output = _run_instrument_agent(workdir, platform, prompt, "instrument")
    if output is None:
        return

    modified_code = _extract_python_block(output)
    if modified_code is None:
        _LG.warning("First instrument attempt returned no code block, retrying")
        retry_prompt = (
            prompt + "\n\n**You MUST output the ENTIRE modified file in a single "
            "```python code block. Do NOT describe changes in prose. "
            "Output ONLY code.**\n"
        )
        output = _run_instrument_agent(
            workdir, platform, retry_prompt, "instrument_retry"
        )
        if output is None:
            return
        modified_code = _extract_python_block(output)

    if modified_code is None:
        _LG.warning("Instrumentation failed — no code block in response")
        print("  Warning: instrumentation failed")
        _record_setup_failure(
            workdir,
            read_state(workdir),
            _make_failure(
                FailureKind.SETUP_INSTRUMENTATION_FAILED,
                FailurePhase.INSTRUMENTATION,
                "Coding agent did not return a Python code block for instrumentation",
            ),
        )
        return

    Path(pipeline_script).write_text(modified_code)
    _LG.info("Wrote instrumented code to %s", pipeline_script)
    print(f"  Wrote instrumented code to {pipeline_script}")

    scm = config.get("scm", "")
    source_dir = config.get("source_dir", "")
    if scm and source_dir and platform.workspace.has_changes(scm, source_dir):
        commit_hash = platform.workspace.commit(
            scm, source_dir, "[autoresearch] Add TTFB/step-time instrumentation"
        )
        _LG.info("Committed instrumentation: %s", commit_hash[:12])
        print(f"  Committed: {commit_hash[:12]}")


def _run_instrument_agent(
    workdir: Path,
    platform: AutoresearchPlatform,
    prompt: str,
    label: str,
) -> str | None:
    try:
        return platform.agent.run(prompt, workdir, label)
    except Exception as error:  # noqa: BLE001 — record and continue
        _record_setup_failure(
            workdir,
            read_state(workdir),
            _make_failure(
                FailureKind.SETUP_INSTRUMENTATION_FAILED,
                FailurePhase.INSTRUMENTATION,
                f"Coding agent failed during {label}",
                exception=error,
            ),
        )
        return None


def _extract_python_block(output: str) -> str | None:
    matches = list(re.finditer(r"```python\s*\n(.*?)\n```", output, re.DOTALL))
    return matches[-1].group(1) if matches else None
