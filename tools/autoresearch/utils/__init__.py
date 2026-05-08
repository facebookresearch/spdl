# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autoresearch utilities.

Re-exports all helpers so launch.py can do `from utils import ...`.
When an fb/ backend is present, its implementations override
the generic stubs for job management and metrics collection.
"""

from .claude import (  # noqa: F401
    extract_json_block,
    load_knowledge,
    load_prompt,
    run_claude,
)
from .jobs import (  # noqa: F401
    apply_lint,
    build_image,
    cancel_job,
    check_job_progress,
    check_job_status,
    collect_metrics_summary,
    fetch_pipeline_stats,
    fetch_system_metrics,
    get_job_duration,
    launch_job,
    wait_for_jobs,
)
from .state import (  # noqa: F401
    append_master_row,
    MASTER_TABLE_HEADERS,
    read_config,
    read_master_table,
    read_state,
    write_state,
)

try:
    from .fb.backend import (  # type: ignore[assignment]  # noqa: F401, F811
        apply_lint,
        cancel_job,
        check_job_progress,
        check_job_status,
        fetch_pipeline_stats,
        fetch_system_metrics,
        get_job_duration,
        launch_job,
    )
except ImportError:
    pass
