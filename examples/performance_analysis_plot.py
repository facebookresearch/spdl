#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Visualize pipeline performance statistics from SQLite database.

This script loads performance statistics collected by SQLiteStatsLogger
and creates time-series plots showing:

- Task execution metrics (throughput, failure rate, average time)
- Queue performance metrics (QPS, occupancy, wait times)
- Per-stage and per-queue breakdowns

The plots help identify performance bottlenecks, throughput patterns,
and resource utilization issues in the pipeline.

Example usage:

.. code-block::

    python3 performance_analysis_plot.py --db-path pipeline_stats.db --output plots/
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

try:
    from examples.sqlite_stats_logger import (  # pyre-ignore[21]
        log_stats_summary,
        query_event_stats,
        query_queue_stats,
        query_task_stats,
        QueueStatsQueryResult,
        TaskStatsQueryResult,
    )
except ImportError:
    from spdl.examples.sqlite_stats_logger import (
        log_stats_summary,
        query_event_stats,
        query_queue_stats,
        query_task_stats,
        QueueStatsQueryResult,
        TaskStatsQueryResult,
    )

__all__ = [
    "parse_args",
    "main",
    "plot_task_stats",
    "plot_queue_stats",
    "add_gc_events_to_plot",
]


def _load_gc_events(
    db_path: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[tuple[float, float]]:
    """Load and pair GC start/stop events from the database.

    Args:
        db_path: Path to the SQLite database file.
        start_time: Optional start timestamp for filtering.
        end_time: Optional end timestamp for filtering.

    Returns:
        List of (start_timestamp, stop_timestamp) tuples for GC events.
    """
    # Query GC events
    event_stats = query_event_stats(
        db_path=db_path,
        start_time=start_time,
        end_time=end_time,
    )

    if not event_stats:
        return []

    # Group events into start-stop pairs
    gc_starts: list[float] = []
    gc_pairs: list[tuple[float, float]] = []

    for event in event_stats:
        if event.event_name == "gc_start":
            gc_starts.append(event.timestamp)
        elif event.event_name == "gc_stop":
            # Pair with the earliest unpaired start event (FIFO)
            if gc_starts:
                start_time_val = gc_starts.pop(0)
                gc_pairs.append((start_time_val, event.timestamp))

    for d in gc_pairs:
        if (duration := d[1] - d[0]) > 1:
            print(f"Long GC time detected: {duration}")

    return gc_pairs


def add_gc_events_to_plot(
    ax: plt.Axes,
    gc_pairs: list[tuple[float, float]],
    label_first: bool = True,
) -> None:
    """Add garbage collection events as vertical spans to a plot.

    Args:
        ax: Matplotlib axes to add GC events to.
        gc_pairs: List of (start_timestamp, stop_timestamp) tuples for GC events.
        label_first: Whether to add a legend label for the first GC span.
    """
    if not gc_pairs:
        return

    # Add vertical spans for each GC period
    for idx, (start, stop) in enumerate(gc_pairs):
        start_dt = datetime.fromtimestamp(start)
        stop_dt = datetime.fromtimestamp(stop)
        # Convert datetime to matplotlib date numbers for axvspan
        start_num = mdates.date2num(start_dt)
        stop_num = mdates.date2num(stop_dt)

        # Only label the first span if requested, use underscore prefix to hide others
        label = "GC" if (idx == 0 and label_first) else "_gc"

        ax.axvspan(
            start_num,
            stop_num,
            facecolor="None",
            hatch="/",
            edgecolor="red",
            alpha=0.3,
            linewidth=0,
            label=label,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("pipeline_stats.db"),
        help="Path to SQLite database containing stats",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots (default: plots/)",
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        help="Filter to specific stage/queue name (optional)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Start timestamp for filtering (Unix timestamp, optional)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="End timestamp for filtering (Unix timestamp, optional)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for saved plots (default: 100)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,8",
        help="Figure size as 'width,height' in inches (default: 12,8)",
    )
    return parser.parse_args()


def plot_task_stats(
    db_path: str,
    output_dir: Path,
    stage_name: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    dpi: int = 100,
    figsize: tuple[int, int] = (12, 20),
) -> None:
    """Plot task statistics over time, combining all stages on the same subplots.

    Args:
        db_path: Path to the SQLite database file.
        output_dir: Directory to save plots.
        stage_name: Optional filter for specific stage.
        start_time: Optional start timestamp.
        end_time: Optional end timestamp.
        dpi: DPI for saved plots.
        figsize: Figure size (width, height) in inches.
    """
    task_stats = query_task_stats(
        db_path=db_path,
        name=stage_name,
        start_time=start_time,
        end_time=end_time,
    )

    if not task_stats:
        print("No task stats found in database")
        return

    # Group by stage name
    stages: dict[str, list[TaskStatsQueryResult]] = {}
    for stat in task_stats:
        name = stat["name"]
        if name not in stages:
            stages[name] = []
        stages[name].append(stat)

    # Sort each stage's stats by timestamp
    for name in stages:
        stages[name].sort(key=lambda x: x["timestamp"])

    # Create a single figure with all stages combined (4 subplots)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle("Task Statistics (All Stages)", fontsize=16, fontweight="bold")

    # Professional color palette inspired by seaborn
    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]
    color_idx = 0

    # Plot all stages on the same subplots
    for name, stage_data in sorted(stages.items()):
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # Convert timestamps to datetime objects
        timestamps = [datetime.fromtimestamp(s["timestamp"]) for s in stage_data]

        # Plot 1: Number of tasks
        ax1 = axes[0]
        ax1.plot(
            timestamps,
            [s["num_tasks"] for s in stage_data],
            label=f"{name} (tasks)",
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

        # Plot 2: Successful tasks
        ax2 = axes[1]
        num_successful = [s["num_tasks"] - s["num_failures"] for s in stage_data]
        ax2.plot(
            timestamps,
            num_successful,
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

        # Plot 3: Failed tasks
        ax3 = axes[2]
        num_failures = [s["num_failures"] for s in stage_data]
        ax3.plot(
            timestamps,
            num_failures,
            label=name,
            marker="x",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

        # Plot 4: Average execution time
        ax4 = axes[3]
        ax4.plot(
            timestamps,
            [s["ave_time"] * 1000 for s in stage_data],
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

    # Configure subplot 1: Task counts
    ax1 = axes[0]
    ax1.set_ylabel("Count", fontsize=11, fontweight="medium")
    ax1.set_title("Task Throughput", fontsize=12, fontweight="semibold", pad=10)
    ax1.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax1.set_facecolor("#F7F7F7")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Configure subplot 2: Successful tasks
    ax2 = axes[1]
    ax2.set_ylabel("Count", fontsize=11, fontweight="medium")
    ax2.set_title("Successful Tasks", fontsize=12, fontweight="semibold", pad=10)
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax2.set_facecolor("#F7F7F7")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Configure subplot 3: Failed tasks
    ax3 = axes[2]
    ax3.set_ylabel("Count", fontsize=11, fontweight="medium")
    ax3.set_title("Failed Tasks", fontsize=12, fontweight="semibold", pad=10)
    ax3.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax3.set_facecolor("#F7F7F7")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Configure subplot 4: Execution times
    ax4 = axes[3]
    ax4.set_ylabel("Time (ms)", fontsize=11, fontweight="medium")
    ax4.set_xlabel("Time", fontsize=11, fontweight="medium")
    ax4.set_title(
        "Average Task Execution Time", fontsize=12, fontweight="semibold", pad=10
    )
    ax4.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax4.set_facecolor("#F7F7F7")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Load GC events once
    gc_pairs = _load_gc_events(db_path, start_time=start_time, end_time=end_time)

    # Format x-axis and add GC events
    for idx, ax in enumerate(axes):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        # Add GC events to all subplots (only label on first subplot)
        add_gc_events_to_plot(ax, gc_pairs, label_first=(idx == 0))

    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "task_stats_combined.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved combined task stats plot to {output_path}")
    plt.close()


def plot_queue_stats(
    db_path: str,
    output_dir: Path,
    queue_name: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    dpi: int = 100,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """Plot queue statistics over time, combining all queues on the same subplots.

    Args:
        db_path: Path to the SQLite database file.
        output_dir: Directory to save plots.
        queue_name: Optional filter for specific queue.
        start_time: Optional start timestamp.
        end_time: Optional end timestamp.
        dpi: DPI for saved plots.
        figsize: Figure size (width, height) in inches.
    """
    queue_stats = query_queue_stats(
        db_path=db_path,
        name=queue_name,
        start_time=start_time,
        end_time=end_time,
    )

    if not queue_stats:
        print("No queue stats found in database")
        return

    # Group by queue name
    queues: dict[str, list[QueueStatsQueryResult]] = {}
    for stat in queue_stats:
        name = stat["name"]
        if name not in queues:
            queues[name] = []
        queues[name].append(stat)

    # Sort each queue's stats by timestamp
    for name in queues:
        queues[name].sort(key=lambda x: x["timestamp"])

    # Create a single figure with all queues combined (5 subplots)
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    fig.suptitle("Queue Statistics (All Queues)", fontsize=16, fontweight="bold")

    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]
    color_idx = 0

    # Plot all queues on the same subplots
    for name, queue_data in sorted(queues.items()):
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # Convert timestamps to datetime objects
        timestamps = [datetime.fromtimestamp(s["timestamp"]) for s in queue_data]

        # Plot 1: QPS (Queries Per Second)
        ax1 = axes[0]
        qps_values = [
            s["num_items"] / s["elapsed"] if s["elapsed"] > 0 else 0 for s in queue_data
        ]
        ax1.plot(
            timestamps,
            qps_values,
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

        # Plot 2: Occupancy rate
        ax2 = axes[1]
        ax2.plot(
            timestamps,
            [s["occupancy_rate"] * 100 for s in queue_data],
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

        # Plot 3: Average put wait time
        ax3 = axes[2]
        ax3.plot(
            timestamps,
            [s["ave_put_time"] * 1000 for s in queue_data],
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

        # Plot 4: Average get wait time
        ax4 = axes[3]
        ax4.plot(
            timestamps,
            [s["ave_get_time"] * 1000 for s in queue_data],
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

        # Plot 5: Number of items processed
        ax5 = axes[4]
        ax5.plot(
            timestamps,
            [s["num_items"] for s in queue_data],
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.85,
        )

    # Configure subplot 1: QPS
    ax1 = axes[0]
    ax1.set_ylabel("QPS", fontsize=11, fontweight="medium")
    ax1.set_title(
        "Queue Throughput (Queries Per Second)",
        fontsize=12,
        fontweight="semibold",
        pad=10,
    )
    ax1.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax1.set_facecolor("#F7F7F7")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Configure subplot 2: Occupancy
    ax2 = axes[1]
    ax2.set_ylabel("Occupancy (%)", fontsize=11, fontweight="medium")
    ax2.set_title("Queue Occupancy Rate", fontsize=12, fontweight="semibold", pad=10)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax2.set_facecolor("#F7F7F7")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Configure subplot 3: Put wait time
    ax3 = axes[2]
    ax3.set_ylabel("Time (ms)", fontsize=11, fontweight="medium")
    ax3.set_title("Average Put Wait Time", fontsize=12, fontweight="semibold", pad=10)
    ax3.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax3.set_facecolor("#F7F7F7")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Configure subplot 4: Get wait time
    ax4 = axes[3]
    ax4.set_ylabel("Time (ms)", fontsize=11, fontweight="medium")
    ax4.set_title("Average Get Wait Time", fontsize=12, fontweight="semibold", pad=10)
    ax4.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax4.set_facecolor("#F7F7F7")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Configure subplot 5: Items processed
    ax5 = axes[4]
    ax5.set_ylabel("Count", fontsize=11, fontweight="medium")
    ax5.set_xlabel("Time", fontsize=11, fontweight="medium")
    ax5.set_title("Items Processed", fontsize=12, fontweight="semibold", pad=10)
    ax5.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax5.set_facecolor("#F7F7F7")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Load GC events once
    gc_pairs = _load_gc_events(db_path, start_time=start_time, end_time=end_time)

    # Format x-axis and add GC events
    for idx, ax in enumerate(axes):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        # Add GC events to all subplots (only label on first subplot)
        add_gc_events_to_plot(ax, gc_pairs, label_first=(idx == 0))

    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "queue_stats_combined.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved combined queue stats plot to {output_path}")
    plt.close()


def main() -> None:
    """Main entry point for the plotting script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname).1s]: %(message)s",
    )

    args = parse_args()

    # Validate database exists
    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    log_stats_summary(args.db_path)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")

    # Parse figsize
    try:
        width, height = map(int, args.figsize.split(","))
        figsize = (width, height)
    except ValueError:
        print(f"Invalid figsize '{args.figsize}', using default (12, 20)")
        figsize = (12, 20)

    # Plot task statistics
    print("Generating task statistics plots...")
    plot_task_stats(
        db_path=str(args.db_path),
        output_dir=args.output,
        stage_name=args.stage_name,
        start_time=args.start_time,
        end_time=args.end_time,
        dpi=args.dpi,
        figsize=figsize,
    )

    # Plot queue statistics
    print("Generating queue statistics plots...")
    plot_queue_stats(
        db_path=str(args.db_path),
        output_dir=args.output,
        queue_name=args.stage_name,  # Same filter applies to queues
        start_time=args.start_time,
        end_time=args.end_time,
        dpi=args.dpi,
        figsize=figsize,
    )

    print(f"✅ All plots saved to {args.output}")


if __name__ == "__main__":
    main()
