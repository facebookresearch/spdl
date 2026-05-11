# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public platform factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .providers import (
    _create_agent_for_config,
    _find_provider,
    _resolve_providers,
    _validate_common_config,
)
from .types import AutoresearchPlatform

__all__ = ["create_platform"]


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
    _validate_common_config(config, workdir)
    provider = _find_provider(_resolve_providers(), kind, config, workdir)
    return provider.create(config, workdir, _create_agent_for_config(config))
