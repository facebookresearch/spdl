# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autoresearch platform boundary."""

from .factory import create_platform
from .types import (
    _Artifacts,
    _CodingAgent,
    _Evidence,
    _Execution,
    _MetricsEvidence,
    _Workspace,
    AutoresearchPlatform,
)

__all__ = [
    "_Artifacts",
    "AutoresearchPlatform",
    "_CodingAgent",
    "create_platform",
    "_Evidence",
    "_Execution",
    "_MetricsEvidence",
    "_Workspace",
]
