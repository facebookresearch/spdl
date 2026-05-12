# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Source mutation, build, and launch operations for workflow experiments."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..platform import AutoresearchPlatform
from ..types import _AutoresearchError, _HypothesisNode, FailureKind, FailurePhase
from .common import _read_pipeline_code
from .failures import _make_failure, _raise_failure

_LG: logging.Logger = logging.getLogger(__name__)


def _extract_code_block(text: str) -> str | None:
    for pattern in [
        r"```python\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
    ]:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            candidate = matches[-1].group(1)
            if "import " in candidate or "def " in candidate or "class " in candidate:
                return candidate
    return None


def _build_apply_prompt(
    platform: AutoresearchPlatform,
    exp: dict,
    run_id: str,
    knowledge: str,
    pipeline_script: str,
    pipeline_code: str,
) -> str:
    if exp.get("_startup_retry_attempt"):
        return platform.agent._load_prompt(
            "apply_startup_repair",
            KNOWLEDGE=knowledge,
            EXPERIMENT_NAME=exp.get("name", run_id),
            EXPERIMENT_DESCRIPTION=exp.get("description", ""),
            EXPERIMENT_HYPOTHESIS=exp.get("hypothesis", ""),
            RETRY_ATTEMPT=str(exp.get("_startup_retry_attempt", "")),
            STARTUP_FAILURE_JSON=json.dumps(
                exp.get("_startup_failure", {}),
                indent=2,
            ),
            PIPELINE_SCRIPT=pipeline_script,
            PIPELINE_CODE=pipeline_code,
        )
    return platform.agent._load_prompt(
        "apply_changes",
        KNOWLEDGE=knowledge,
        EXPERIMENT_NAME=exp.get("name", run_id),
        EXPERIMENT_DESCRIPTION=exp.get("description", ""),
        EXPERIMENT_HYPOTHESIS=exp.get("hypothesis", ""),
        PIPELINE_SCRIPT=pipeline_script,
        PIPELINE_CODE=pipeline_code,
    )


def _apply_code_changes(
    workdir: Path,
    config: dict,
    state: dict,
    platform: AutoresearchPlatform,
    exp: dict,
    run_id: str,
    knowledge: str,
    pipeline_code: str,
) -> bool:
    pipeline_script = config.get("pipeline_script", "")
    if not pipeline_script or not pipeline_code:
        _LG.warning("No pipeline script configured; cannot apply code changes")
        _raise_failure(
            FailureKind.CODE_CHANGE_FAILED,
            FailurePhase.CODE_CHANGE,
            "No pipeline script configured for source-changing experiment",
        )

    _LG.info("Applying code changes for %s: %s", run_id, exp.get("description", ""))
    print(f"  Applying code changes for {exp['name']}...")

    prompt = _build_apply_prompt(
        platform, exp, run_id, knowledge, pipeline_script, pipeline_code
    )
    try:
        output = platform.agent.run(prompt, workdir, f"apply_{run_id}")
    except Exception as error:
        raise _AutoresearchError(
            _make_failure(
                FailureKind.CODE_CHANGE_FAILED,
                FailurePhase.CODE_CHANGE,
                "Coding agent failed while applying experiment source changes",
                exception=error,
            )
        ) from error
    modified_code = _extract_code_block(output)

    if not modified_code:
        _LG.warning(
            "First apply attempt returned no code block for %s, retrying", run_id
        )
        retry_prompt = (
            prompt + "\n\n**You MUST output the ENTIRE modified file in a single "
            "```python code block. Do NOT describe changes in prose.**\n"
        )
        try:
            output = platform.agent.run(retry_prompt, workdir, f"apply_{run_id}_retry")
        except Exception as error:
            raise _AutoresearchError(
                _make_failure(
                    FailureKind.CODE_CHANGE_FAILED,
                    FailurePhase.CODE_CHANGE,
                    "Coding agent failed while retrying experiment source changes",
                    exception=error,
                )
            ) from error
        modified_code = _extract_code_block(output)

    if not modified_code:
        _LG.warning("Could not extract modified code from agent output for %s", run_id)
        _LG.debug("Agent output (first 2000 chars): %s", output[:2000])
        print("  Warning: code change failed - no code block in response")
        failed_log = workdir / "logs" / f"apply_{run_id}_failed.md"
        failed_log.write_text(output)
        print(f"  Full output saved to {failed_log}")
        _raise_failure(
            FailureKind.CODE_CHANGE_FAILED,
            FailurePhase.CODE_CHANGE,
            "Coding agent did not return a Python code block for experiment changes",
            details={"log": str(failed_log)},
        )

    target = Path(pipeline_script)
    target.write_text(modified_code)
    _LG.info("Wrote modified code to %s", target)
    print(f"  Wrote modified code to {target}")

    source_dir = config.get("source_dir", "")
    if source_dir:
        platform.workspace.apply_lint(source_dir)

    scm = config.get("scm", "")
    if scm and source_dir and platform.workspace.has_changes(scm, source_dir):
        msg = f"[autoresearch] {run_id}: {exp.get('description', exp['name'])}"
        commit_hash = platform.workspace.commit(scm, source_dir, msg)
        exp["commit"] = commit_hash
        run_dir = workdir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "commit.txt").write_text(commit_hash + "\n")
        _LG.info("Committed code changes: %s", commit_hash[:12])
        print(f"  Committed: {commit_hash[:12]}")

    return True


def _prepare_node(
    workdir: Path,
    config: dict,
    state: dict,
    platform: AutoresearchPlatform,
    node: object,
    parent: object | None,
) -> bool:
    """Prepare a node: restore source, apply code changes, and build."""
    assert isinstance(node, _HypothesisNode)

    exp = node.spec
    run_id = node.node_id
    knowledge = platform.agent._load_knowledge()

    scm = config.get("scm", "")
    source_dir = config.get("source_dir", "")
    anchor = state.get("anchor_commit", "")
    target_commit = exp.get("goto") or anchor
    if target_commit and scm and source_dir:
        try:
            platform.workspace.goto(scm, source_dir, target_commit, anchor)
        except Exception as error:
            _raise_failure(
                FailureKind.SOURCE_RESTORE_FAILED,
                FailurePhase.SOURCE,
                "Failed to restore source revision",
                exception=error,
            )

    pipeline_code = _read_pipeline_code(config, workdir)
    _apply_code_changes(
        workdir,
        config,
        state,
        platform,
        exp,
        run_id,
        knowledge,
        pipeline_code,
    )
    node.commit = exp.get("commit")

    if scm and source_dir and platform.workspace.has_changes(scm, source_dir):
        msg = f"[autoresearch] {run_id} changes"
        platform.workspace.commit(scm, source_dir, msg)

    build_cmd = config.get("build_command", "")
    if build_cmd:
        try:
            image = platform.artifacts.build(build_cmd, workdir, source_dir)
        except Exception as error:
            raise _AutoresearchError(
                _make_failure(
                    FailureKind.BUILD_FAILED,
                    FailurePhase.BUILD,
                    "Build command raised an exception",
                    exception=error,
                )
            ) from error
        if image:
            exp["_image"] = image
        else:
            _LG.warning("Build failed for %s", run_id)
            _raise_failure(
                FailureKind.BUILD_FAILED,
                FailurePhase.BUILD,
                "Build command failed or produced no image",
            )

    run_dir = workdir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(exp, indent=2) + "\n")
    return True


def _launch_node(
    config: dict,
    state: dict,
    platform: AutoresearchPlatform,
    node: object,
    workdir: Path,
) -> str | None:
    """Launch a job for a node. Returns job_id or raises structured failure."""
    assert isinstance(node, _HypothesisNode)

    exp = node.spec
    launch_cmd = exp.get("launch_command", "") or config.get("base_launch_command", "")
    if not launch_cmd:
        print(f"  Skipping {node.name}: no launch_command")
        _raise_failure(
            FailureKind.LAUNCH_FAILED,
            FailurePhase.LAUNCH,
            "No launch command configured for experiment",
        )

    image = exp.get("_image")
    if "$IMAGE" in launch_cmd and image:
        launch_cmd = launch_cmd.replace("$IMAGE", image)

    try:
        job_id = platform.execution.launch(launch_cmd, workdir)
    except Exception as error:
        raise _AutoresearchError(
            _make_failure(
                FailureKind.LAUNCH_FAILED,
                FailurePhase.LAUNCH,
                "Platform raised while launching job",
                exception=error,
            )
        ) from error

    run_dir = workdir / "runs" / node.node_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if job_id:
        print(f"  Job ID: {job_id}")
        (run_dir / "job_id.txt").write_text(job_id + "\n")
    else:
        (run_dir / "job_id.txt").write_text("LAUNCH_FAILED\n")
        _raise_failure(
            FailureKind.LAUNCH_FAILED,
            FailurePhase.LAUNCH,
            "Platform did not return a job id",
        )
    return job_id


__all__ = [
    "_build_apply_prompt",
    "_launch_node",
    "_prepare_node",
]
