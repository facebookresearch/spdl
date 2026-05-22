# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Neutral platform provider discovery for autoresearch.

Core autoresearch owns the CLI, runner, workflow, and provider interface.
Environment-specific infrastructure is supplied by provider modules discovered
through standard Python entry points or an explicit environment hook. Keep this
module free of provider-specific imports so OSS code does not know about any
particular internal extension.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Protocol

from ._agents import _create_agent
from ._local import (
    _DefaultArtifacts,
    _DefaultEvidence,
    _DefaultExecution,
    _DefaultWorkspace,
    _LocalEvidence,
    _LocalExecution,
    _LocalWorkspace,
)
from ._types import _CodingAgent, AutoresearchPlatform

__all__ = [
    "_AutoresearchPlatformProvider",
    "_ProviderAvailability",
    "_builtin_providers",
    "_discover_convention_providers",
    "_discover_extension_providers",
    "_find_provider",
    "_provider_from_spec",
    "_resolve_providers",
]

_ENTRY_POINT_GROUP = "spdl.autoresearch.platforms"
_PROVIDER_ENV = "SPDL_AUTORESEARCH_PLATFORM_PROVIDERS"
_LOCAL_EXECUTION_MODES = frozenset({"full", "dataloader_only", "dry_run"})
_AGENT_KINDS = frozenset({"claude", "codex", "mock"})


@dataclass(frozen=True)
class _ProviderAvailability:
    available: bool
    reason: str = ""


class _AutoresearchPlatformProvider(Protocol):
    name: str
    priority: int

    def is_available(
        self, config: dict[str, Any], workdir: Path | None
    ) -> _ProviderAvailability: ...

    def create(
        self,
        config: dict[str, Any],
        workdir: Path | None,
        agent: _CodingAgent,
    ) -> AutoresearchPlatform: ...


class _LocalProvider:
    name = "local"
    priority = 10

    def is_available(
        self, config: dict[str, Any], workdir: Path | None
    ) -> _ProviderAvailability:
        mode = str(config.get("local_execution_mode", "full"))
        if mode not in _LOCAL_EXECUTION_MODES:
            return _ProviderAvailability(False, f"unknown local execution mode: {mode}")
        return _ProviderAvailability(True)

    def create(
        self,
        config: dict[str, Any],
        workdir: Path | None,
        agent: _CodingAgent,
    ) -> AutoresearchPlatform:
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


class _DefaultProvider:
    name = "default"
    priority = 0

    def is_available(
        self, config: dict[str, Any], workdir: Path | None
    ) -> _ProviderAvailability:
        return _ProviderAvailability(True)

    def create(
        self,
        config: dict[str, Any],
        workdir: Path | None,
        agent: _CodingAgent,
    ) -> AutoresearchPlatform:
        return AutoresearchPlatform(
            workspace=_DefaultWorkspace(),
            artifacts=_DefaultArtifacts(),
            execution=_DefaultExecution(),
            evidence=_DefaultEvidence(),
            agent=agent,
        )


def _builtin_providers() -> list[_AutoresearchPlatformProvider]:
    return [_LocalProvider(), _DefaultProvider()]


def _resolve_providers() -> list[_AutoresearchPlatformProvider]:
    providers = _builtin_providers()
    providers.extend(_discover_extension_providers())
    return providers


def _discover_extension_providers() -> list[_AutoresearchPlatformProvider]:
    providers: list[_AutoresearchPlatformProvider] = []
    specs = [
        spec.strip()
        for spec in os.environ.get(_PROVIDER_ENV, "").split(",")
        if spec.strip()
    ]
    for spec in specs:
        providers.append(_provider_from_spec(spec))

    for entry_point in importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP):
        loaded = entry_point.load()
        providers.append(loaded() if callable(loaded) else loaded)
    providers.extend(_discover_convention_providers())
    return providers


def _discover_convention_providers() -> list[_AutoresearchPlatformProvider]:
    parent_package = __package__.rsplit(".", 1)[0]
    root = importlib.import_module(parent_package)
    package_paths = getattr(root, "__path__", None)
    if package_paths is None:
        return []

    providers: list[_AutoresearchPlatformProvider] = []
    for module_info in pkgutil.iter_modules(package_paths):
        try:
            module = importlib.import_module(
                f"{parent_package}.{module_info.name}._platform"
            )
        except ImportError:
            continue
        factory = getattr(module, "create_provider", None)
        if factory is not None:
            providers.append(factory())
    return providers


def _provider_from_spec(spec: str) -> _AutoresearchPlatformProvider:
    module_name, sep, attr = spec.partition(":")
    if not sep:
        attr = "create_provider"
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    provider = factory()
    return provider


def _find_provider(
    providers: list[_AutoresearchPlatformProvider],
    requested: str,
    config: dict[str, Any],
    workdir: Path | None,
) -> _AutoresearchPlatformProvider:
    if requested == "auto":
        available = [
            provider
            for provider in providers
            if provider.is_available(config, workdir).available
        ]
        if not available:
            reasons = "; ".join(
                f"{provider.name}: {provider.is_available(config, workdir).reason}"
                for provider in providers
            )
            raise ValueError(
                f"No autoresearch platform providers are available: {reasons}"
            )
        return max(available, key=lambda provider: provider.priority)

    for provider in providers:
        aliases = getattr(provider, "aliases", ())
        if provider.name != requested and requested not in aliases:
            continue
        availability = provider.is_available(config, workdir)
        if not availability.available:
            raise ValueError(
                f"Autoresearch platform provider {requested!r} is unavailable: "
                f"{availability.reason}"
            )
        return provider

    raise ValueError(
        f"Unknown autoresearch platform provider: {requested}. "
        f"Available providers: {', '.join(provider.name for provider in providers)}"
    )


def _validate_common_config(config: dict[str, Any], workdir: Path | None) -> None:
    agent = str(config.get("agent", "claude"))
    if agent not in _AGENT_KINDS:
        raise ValueError(f"Unknown autoresearch agent: {agent}")

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


def _create_agent_for_config(config: dict[str, Any]) -> _CodingAgent:
    agent_kind = str(config.get("agent", "claude"))
    return cast(
        _CodingAgent, _create_agent(agent_kind, command=config.get("agent_command"))
    )
