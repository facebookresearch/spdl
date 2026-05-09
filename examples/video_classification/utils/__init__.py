# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Compatibility layer: switch between Meta-internal and OSS implementations."""

from __future__ import annotations

from .pipeline import build_pipeline

try:
    from .fb.dataset import (  # pyre-ignore[21]
        add_dataset_args,
        create_dataset,
        get_label_to_index,
    )
    from .fb.helpers import report_progress  # pyre-ignore[21]
except ImportError:
    from .dataset import (  # type: ignore[assignment]
        add_dataset_args,
        create_dataset,
        get_label_to_index,
        report_progress,
    )

__all__ = [
    "add_dataset_args",
    "build_pipeline",
    "create_dataset",
    "get_label_to_index",
    "report_progress",
]
