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

from spdl.autoresearch.core import (
    AutoresearchError,
    FailureKind,
    FailurePhase,
    HypothesisNode,
)

from .._platform import AutoresearchPlatform
from ._common import _read_pipeline_code
from ._failures import _make_failure, _raise_failure

_LG: logging.Logger = logging.getLogger(__name__)

_MAX_TITLE_LEN = 72


def _build_commit_message(
    run_id: str,
    exp: dict,
    *,
    extra_lines: list[str] | None = None,
) -> str:
    """Build a commit message with a short title and detailed body.

    Format follows the Git/GitHub convention:
    - Line 1: title (≤72 chars) describing the delta
    - Line 2: blank
    - Line 3+: body with description, hypothesis, and context

    Works with both ``git commit -m`` and ``sl commit -m``.
    """
    name = exp.get("change_summary") or exp.get("name", run_id)
    prefix = f"ar: {run_id}: "
    max_name_len = _MAX_TITLE_LEN - len(prefix)
    if len(name) > max_name_len:
        name = name[: max_name_len - 1] + "…"
    title = prefix + name

    body_parts: list[str] = []

    description = exp.get("description", "")
    if description:
        body_parts.append(description)

    hypothesis = exp.get("hypothesis", "")
    if hypothesis:
        body_parts.append(f"Hypothesis: {hypothesis}")

    changes = exp.get("changes")
    if changes:
        body_parts.append(f"Changes: {', '.join(changes)}")

    if extra_lines:
        body_parts.extend(extra_lines)

    if not body_parts:
        return title

    return title + "\n\n" + "\n\n".join(body_parts)


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


def _extract_top_level_names(source: str) -> set[str]:
    """Extract top-level function and class names from Python source code.

    Uses regex rather than ``ast.parse`` to avoid failures on code that
    contains syntax specific to newer Python versions.
    """
    names: set[str] = set()
    for match in re.finditer(r"^(?:def|class)\s+(\w+)", source, re.MULTILINE):
        names.add(match.group(1))
    return names


def _extract_import_modules(source: str) -> set[str]:
    """Extract imported module names from ``import`` / ``from … import`` lines.

    Returns dotted module paths (e.g. ``{"torch", "torch.distributed",
    "spdl.io"}``).  Used to detect wholesale file replacement where the
    agent outputs a different module's content.
    """
    modules: set[str] = set()
    for match in re.finditer(r"^(?:import|from)\s+([\w.]+)", source, re.MULTILINE):
        modules.add(match.group(1))
    return modules


def _validate_preserved_symbols(original_code: str, modified_code: str) -> list[str]:
    """Return names of top-level symbols present in *original_code* but
    missing from *modified_code*.  An empty list means all symbols are
    preserved.

    Also detects wrong-file replacement by checking whether a majority
    of the original file's imported modules were dropped — a strong
    signal that the agent output the content of a different file.
    """
    original_names = _extract_top_level_names(original_code)
    modified_names = _extract_top_level_names(modified_code)
    missing = original_names - modified_names

    # Detect wrong-file replacement: if more than half of the original
    # imported modules disappeared, the agent almost certainly output
    # the content of a different file (e.g. utils/pipeline.py instead
    # of the training script).
    original_imports = _extract_import_modules(original_code)
    modified_imports = _extract_import_modules(modified_code)
    dropped_imports = original_imports - modified_imports
    if original_imports and len(dropped_imports) > len(original_imports) // 2:
        missing.add(
            f"imports:dropped {len(dropped_imports)}/{len(original_imports)} "
            f"({', '.join(sorted(dropped_imports))})"
        )

    return sorted(missing)


def _build_apply_prompt(
    platform: AutoresearchPlatform,
    exp: dict,
    run_id: str,
    knowledge: str,
    pipeline_script: str,
    pipeline_code: str,
) -> str:
    if exp.get("_is_headspace"):
        return platform.agent._load_prompt(
            "headspace",
            KNOWLEDGE=knowledge,
            PIPELINE_SCRIPT=pipeline_script,
            PIPELINE_CODE=pipeline_code,
        )
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
        raise AutoresearchError(
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
            raise AutoresearchError(
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

    # Validate that the agent didn't accidentally replace the file with
    # content from a different module (a common failure mode when the
    # experiment description references a file other than pipeline_script).
    missing = _validate_preserved_symbols(pipeline_code, modified_code)
    if missing:
        _LG.warning(
            "Modified code is missing top-level symbols from the original: %s",
            ", ".join(missing),
        )
        print(f"  Warning: modified code drops symbols: {', '.join(missing)}")
        _raise_failure(
            FailureKind.CODE_CHANGE_FAILED,
            FailurePhase.CODE_CHANGE,
            f"Agent output is missing top-level symbols from the original "
            f"file ({', '.join(missing)}). This usually means the agent "
            f"output the content of a different file instead of modifying "
            f"the pipeline script. The experiment description may reference "
            f"a file that is not the pipeline script.",
            details={"missing_symbols": missing},
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
        msg = _build_commit_message(run_id, exp)
        try:
            commit_hash = platform.workspace.commit(scm, source_dir, msg)
        except Exception as error:
            raise AutoresearchError(
                _make_failure(
                    FailureKind.CODE_CHANGE_FAILED,
                    FailurePhase.CODE_CHANGE,
                    "Failed to commit code changes to source control",
                    exception=error,
                )
            ) from error
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
    assert isinstance(node, HypothesisNode)

    exp = node.spec
    run_id = node.node_id
    knowledge = platform.agent._load_knowledge()

    scm = config.get("scm", "")
    source_dir = config.get("source_dir", "")
    anchor = state.get("anchor_commit", "")
    parent_ok = (
        isinstance(parent, HypothesisNode)
        and parent.status == "completed"
        and not parent.spec.get("_is_headspace")
    )
    parent_commit = node.commit if parent_ok else None
    target_commit = exp.get("goto") or parent_commit or anchor
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
        msg = _build_commit_message(
            run_id, exp, extra_lines=["Residual changes after lint/build."]
        )
        try:
            platform.workspace.commit(scm, source_dir, msg)
        except Exception as error:
            raise AutoresearchError(
                _make_failure(
                    FailureKind.CODE_CHANGE_FAILED,
                    FailurePhase.CODE_CHANGE,
                    "Failed to commit code changes to source control",
                    exception=error,
                )
            ) from error

    build_cmd = config.get("build_command", "")
    if build_cmd:
        try:
            image = platform.artifacts.build(build_cmd, workdir, source_dir)
        except Exception as error:
            raise AutoresearchError(
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
    assert isinstance(node, HypothesisNode)

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
        raise AutoresearchError(
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
