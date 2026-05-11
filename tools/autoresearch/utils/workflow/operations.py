# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility imports for autoresearch workflow operations.

Implementation lives in cohesive modules:
``source_ops`` for source/build/launch, ``analysis_ops`` for metrics and
result recording, and ``planning_ops`` for initial/follow-up experiment
planning. Keep new code in those modules so future agents do not recreate a
large flat operations file.
"""

from __future__ import annotations

from .analysis_ops import _analyze_job, _update_on_complete, _update_summary_and_plot
from .planning_ops import _build_initial_nodes, _plan_followups, _should_stop
from .source_ops import _launch_node, _prepare_node

__all__ = [
    "_analyze_job",
    "_build_initial_nodes",
    "_plan_followups",
    "_launch_node",
    "_prepare_node",
    "_should_stop",
    "_update_on_complete",
    "_update_summary_and_plot",
]
