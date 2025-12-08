#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Export SQLite performance statistics to JSON for web visualization.

This script reads SQLite databases created by performance_simulation.py
and exports the data in a JSON format suitable for plotly.js visualization.

Usage:
    python export_stats_to_json.py --db-path data/pipeline.db --output stats.json
"""

import argparse
import json
import logging
from pathlib import Path

try:
    from examples.sqlite_stats_logger import (  # pyre-ignore[21]
        query_queue_stats,
        query_task_stats,
    )
except ImportError:
    from spdl.examples.sqlite_stats_logger import (
        query_queue_stats,
        query_task_stats,
    )


__all__ = ["export_to_json", "parse_args", "main"]

_LG: logging.Logger = logging.getLogger(__name__)


def export_to_json(
    db_path: str,
    output_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Export SQLite statistics to JSON format.

    Args:
        db_path: Path to the SQLite database file.
        output_path: Path to save the JSON output file.
        start_time: Optional start timestamp filter.
        end_time: Optional end timestamp filter.
    """
    print(f"📊 Exporting statistics from {db_path}...")

    # Query all stats
    task_stats = query_task_stats(
        db_path=db_path,
        start_time=start_time,
        end_time=end_time,
    )

    queue_stats = query_queue_stats(
        db_path=db_path,
        start_time=start_time,
        end_time=end_time,
    )

    if not task_stats and not queue_stats:
        print("⚠️  No statistics found in database")
        return

    # Organize data by stage/queue name and metric
    data: dict[str, list[float | int]] = {}

    # Process task stats
    task_by_stage: dict[str, list[tuple[float, dict[str, float | int]]]] = {}
    for stat in task_stats:
        name = stat["name"]
        if name not in task_by_stage:
            task_by_stage[name] = []
        task_by_stage[name].append(
            (
                stat["timestamp"],
                {
                    "num_tasks": stat["num_tasks"],
                    "num_failures": stat["num_failures"],
                    "ave_time": stat["ave_time"],
                },
            )
        )

    # Sort by timestamp and convert to time series
    for name, values in task_by_stage.items():
        values.sort(key=lambda x: x[0])
        timestamps = [v[0] for v in values]
        data[f"{name}_timestamp"] = timestamps
        data[f"{name}_num_tasks"] = [v[1]["num_tasks"] for v in values]
        data[f"{name}_num_failures"] = [v[1]["num_failures"] for v in values]
        data[f"{name}_ave_time"] = [v[1]["ave_time"] for v in values]

    # Process queue stats
    queue_by_name: dict[str, list[tuple[float, dict[str, float | int]]]] = {}
    for stat in queue_stats:
        name = stat["name"]
        if name not in queue_by_name:
            queue_by_name[name] = []
        # Calculate QPS
        qps = stat["num_items"] / stat["elapsed"] if stat["elapsed"] > 0 else 0
        queue_by_name[name].append(
            (
                stat["timestamp"],
                {
                    "qps": qps,
                    "occupancy_rate": stat["occupancy_rate"],
                    "ave_put_time": stat["ave_put_time"],
                    "ave_get_time": stat["ave_get_time"],
                    "num_items": stat["num_items"],
                },
            )
        )

    # Sort by timestamp and convert to time series
    for name, values in queue_by_name.items():
        values.sort(key=lambda x: x[0])
        timestamps = [v[0] for v in values]
        data[f"{name}_timestamp"] = timestamps
        data[f"{name}_qps"] = [v[1]["qps"] for v in values]
        data[f"{name}_occupancy_rate"] = [v[1]["occupancy_rate"] for v in values]
        data[f"{name}_ave_put_time"] = [v[1]["ave_put_time"] for v in values]
        data[f"{name}_ave_get_time"] = [v[1]["ave_get_time"] for v in values]
        data[f"{name}_num_items"] = [v[1]["num_items"] for v in values]

    # Write to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"✅ Exported statistics to {output_path}")
    print(f"   Task stages: {len(task_by_stage)}")
    print(f"   Queues: {len(queue_by_name)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the JSON output file",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Optional start timestamp filter",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="Optional end timestamp filter",
    )
    return parser.parse_args()


def main() -> None:
    """The main entry point for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname).1s]: %(message)s",
    )

    args = parse_args()

    export_to_json(
        db_path=args.db_path,
        output_path=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
    )


if __name__ == "__main__":
    main()
