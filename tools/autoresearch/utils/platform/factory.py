# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public platform factory."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

from .agents import _create_agent
from .local import (
    _DefaultArtifacts,
    _DefaultEvidence,
    _DefaultExecution,
    _DefaultWorkspace,
    _LocalEvidence,
    _LocalExecution,
    _LocalWorkspace,
)
from .types import _CodingAgent, AutoresearchPlatform

__all__ = ["create_platform"]

_LOCAL_EXECUTION_MODES = frozenset({"full", "dataloader_only", "dry_run"})
_AGENT_KINDS = frozenset({"claude", "codex", "mock"})


def create_platform(
    config_or_kind: dict[str, Any] | str,
    workdir: Path | None = None,
) -> AutoresearchPlatform:
    """Create a platform from persisted config or a direct platform name."""

    if isinstance(config_or_kind, str):
        config: dict[str, Any] = {"platform": config_or_kind}
    else:
        config = config_or_kind

    kind = str(config.get("platform", "auto"))
    agent_kind = str(config.get("agent", "claude"))
    _validate_platform_config(config, workdir)
    agent = cast(
        _CodingAgent, _create_agent(agent_kind, command=config.get("agent_command"))
    )

    if kind == "local":
        mode = str(config.get("local_execution_mode", "full"))
        execution = _LocalExecution(mode, config.get("local_dataloader_command"))
        if workdir is not None:
            execution.bind_workdir(workdir)
        return AutoresearchPlatform(
            workspace=_LocalWorkspace(),
            artifacts=_DefaultArtifacts(),
            execution=execution,
            evidence=_LocalEvidence(),
            agent=agent,
        )

    if kind in ("auto", "remote"):
        fb_platform = _create_fb_platform(config, workdir, agent)
        if fb_platform is not None:
            return fb_platform
        if kind == "remote":
            raise ValueError("Remote autoresearch platform is not available")
        return AutoresearchPlatform(
            workspace=_DefaultWorkspace(),
            artifacts=_DefaultArtifacts(),
            execution=_DefaultExecution(),
            evidence=_DefaultEvidence(),
            agent=agent,
        )

    raise ValueError(f"Unknown autoresearch platform: {kind}")


def _validate_platform_config(config: dict[str, Any], workdir: Path | None) -> None:
    kind = str(config.get("platform", "auto"))
    if kind not in ("auto", "remote", "local"):
        raise ValueError(f"Unknown autoresearch platform: {kind}")

    agent = str(config.get("agent", "claude"))
    if agent not in _AGENT_KINDS:
        raise ValueError(f"Unknown autoresearch agent: {agent}")

    mode = str(config.get("local_execution_mode", "full"))
    if mode not in _LOCAL_EXECUTION_MODES:
        raise ValueError(f"Unknown local execution mode: {mode}")

    source_dir = str(config.get("source_dir", ""))
    if source_dir and not Path(source_dir).exists():
        raise ValueError(f"Autoresearch source_dir does not exist: {source_dir}")

    pipeline_script = config.get("pipeline_script")
    if pipeline_script and not Path(str(pipeline_script)).exists():
        raise ValueError(
            f"Autoresearch pipeline_script does not exist: {pipeline_script}"
        )

    if workdir is not None and not workdir.exists():
        workdir.mkdir(parents=True, exist_ok=True)


def _create_fb_platform(
    config: dict[str, Any],
    workdir: Path | None,
    agent: object,
) -> AutoresearchPlatform | None:
    try:
        factory = import_module("spdl.tools.autoresearch.utils.platform.fb.factory")
    except ImportError:
        return None
    create = getattr(factory, "_create_platform", None)
    if create is None:
        return None
    return create(config=config, workdir=workdir, agent=agent)
