#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Simulation script to demonstrate pipeline bottleneck scenarios.

This script creates a configurable multi-stage pipeline and runs with custom
stage configurations to demonstrate how bottlenecks in different stages affect
performance metrics.

.. seealso::

   :doc:`../optimization_guide/stats`
      Uses this script to illustrates how pipeline configuration affects
      the performance statistics.

The script accepts flexible stage configurations via CLI arguments, where each
stage can have:
- Custom processing time (sleep duration in seconds)
- Custom concurrency level (number of parallel tasks)
- Optional aggregation (batch size for grouping items)

Performance statistics are collected at configurable intervals and saved to
a SQLite database for analysis. The foreground thread simulates a consumer
(e.g., training loop) with configurable sleep duration.

Example usage:

.. code-block::

   # Single stage with 50ms processing
   python performance_simulation.py --db-path output.db --stage-configs "0.050,1"

   # Three stages: fast (10ms) -> medium (25ms) -> slow (40ms)
   python performance_simulation.py --db-path output.db --stage-configs "0.010,1;0.025,1;0.040,1"

   # Stage with concurrency and aggregation
   python performance_simulation.py --db-path output.db --stage-configs "0.006,1,1;0.0,1,4"
"""

import argparse
import asyncio
import logging
import random
import time
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Any, TypeVar

from spdl.pipeline import Pipeline, PipelineBuilder

try:
    from examples.sqlite_stats_logger import (  # pyre-ignore[21]
        EventLogEntry,
        log_stats_summary,
        QueueStatsLogEntry,
        SQLiteStatsWriter,
        TaskStatsLogEntry,
    )
except ImportError:
    from spdl.examples.sqlite_stats_logger import (
        EventLogEntry,
        log_stats_summary,
        QueueStatsLogEntry,
        SQLiteStatsWriter,
        TaskStatsLogEntry,
    )

try:
    from examples.performance_analysis import (  # pyre-ignore[21]
        StatsQueueWithLogging,
        TaskStatsHookWithLogging,
    )
except ImportError:
    from spdl.examples.performance_analysis import (
        StatsQueueWithLogging,
        TaskStatsHookWithLogging,
    )


__all__ = [
    "parse_args",
    "main",
    "build_pipeline",
    "SimulatedStage",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


class SimulatedStage:
    """Callable stage with configurable sleep duration.

    This class simulates a processing stage with a specified average
    sleep duration. Each invocation applies ±10% random jitter to the
    sleep duration to simulate processing time variance.

    Args:
        sleep_duration: Average sleep duration in seconds (e.g., 0.050 for 50ms).
            Actual duration will vary by ±10% on each call.
    """

    def __init__(self, sleep_duration: float) -> None:
        self.sleep_duration = sleep_duration

    async def __call__(self, item: T) -> T:
        """Process an item with simulated delay including random jitter.

        Applies the configured sleep duration with ±10% random jitter to simulate
        realistic processing time variance.

        Args:
            item: The item to process. (The data is passed through unmodified)

        Returns:
            The same item (unmodified) after simulated processing delay.
        """
        if self.sleep_duration > 0:
            # Add ±10% jitter to make simulation more realistic
            jitter = random.uniform(-0.1, 0.1)
            actual_duration = self.sleep_duration * (1 + jitter)
            await asyncio.sleep(actual_duration)
        return item


def build_pipeline(
    source: Iterable[int],
    log_interval: float,
    buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry],
    stage_configs: list[tuple[float, int, int]],
) -> Pipeline:
    """Build a pipeline with configurable stage functions and aggregation.

    Creates a pipeline with one or more stages, each having configurable processing
    time, concurrency, and optional aggregation. Stage names are automatically
    generated based on their configuration (e.g., "25ms", "pass", "50ms_c2").

    Args:
        source: A data source (iterable) containing integers.
        log_interval: The interval in seconds at which performance statistics are logged.
        buffer: Shared queue for collecting :py:class:`TaskStatsLogEntry`, :py:class`QueueStatsLogEntry`,
            and :py:class:`EventLogEntry` objects.
        stage_configs: List of tuples (sleep_duration, concurrency, batch_size) for each stage.

            - ``sleep_duration``: Processing time in seconds (0 for passthrough)
            - ``concurrency``: Number of parallel tasks (>= 1)
            - ``batch_size``: Number of items to aggregate (1 for no aggregation)

            Example: ``[(0.050, 1, 1), (0.025, 2, 1), (0.0, 1, 4)]`` creates three stages
            where the last stage aggregates 4 items into batches.

    Returns:
        A configured Pipeline instance ready for execution with performance monitoring.
    """

    def hook_factory(name: str) -> list[Any]:
        return [
            TaskStatsHookWithLogging(
                name=name,
                buffer=buffer,
                interval=log_interval,
            )
        ]

    builder = PipelineBuilder().add_source(source=source)

    # Calculate total num_threads as the sum of all stage concurrencies
    total_threads = sum(concurrency for _, concurrency, _ in stage_configs)

    # Add stages based on configuration with descriptive names
    for sleep_duration, concurrency, batch_size in stage_configs:
        stage = SimulatedStage(sleep_duration)

        # Create short, descriptive name based on configuration
        if sleep_duration == 0:
            name = "pass"
        else:
            sleep_ms = int(sleep_duration * 1000)
            name = f"{sleep_ms}ms"

        # Add concurrency suffix if > 1
        if concurrency > 1:
            name = f"{name}_c{concurrency}"

        builder = builder.pipe(stage, concurrency=concurrency, name=name)

        # If batch_size > 1, insert aggregation after the processing stage
        if batch_size > 1:
            builder = builder.aggregate(batch_size)

    return builder.add_sink().build(
        num_threads=total_threads,
        queue_class=partial(  # pyre-ignore[6]
            StatsQueueWithLogging,
            buffer=buffer,
            interval=log_interval,
        ),
        task_hook_factory=hook_factory,
    )


def infinite_source() -> Iterable[int]:
    """Generate an infinite stream of integers."""
    counter = 0
    while True:
        yield counter
        counter += 1


def run_configuration(
    db_path: Path,
    stage_configs: list[tuple[float, int, int]],
    log_interval: float,
    run_duration: float,
    foreground_sleep: float,
) -> None:
    """Run a single pipeline configuration and collect performance statistics.

    Builds and runs a pipeline with the specified configuration, simulating a
    foreground consumer (e.g., training loop) that processes items with a
    configurable sleep duration. Performance statistics are collected and saved
    to a SQLite database.

    Args:
        db_path: Path where the SQLite database file will be saved.
        stage_configs: List of tuples (sleep_duration, concurrency, batch_size) defining
            each pipeline stage's configuration.
        log_interval: Interval in seconds at which performance statistics are logged
            to the database.
        run_duration: Total duration in seconds to run the pipeline before stopping.
        foreground_sleep: Sleep duration in seconds for the foreground consumer thread
            (e.g., 0.030 for 30ms).
    """
    print(f"\n{'=' * 80}")
    print("🎯 Running Pipeline Simulation")
    print(f"{'=' * 80}")
    print(f"   Database: {db_path}")
    print(f"   Stage configs: {stage_configs}")
    print(f"   Log interval: {log_interval}s")
    print(f"   Run duration: {run_duration}s")
    print(f"   Foreground sleep: {foreground_sleep * 1000:.0f}ms")
    print()

    # Create shared buffer and writer
    buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry] = Queue()
    writer = SQLiteStatsWriter(
        str(db_path),
        buffer,
        flush_interval=max(0.1, log_interval - 1),
    )
    writer.start()

    # Build pipeline
    pipeline = build_pipeline(
        source=infinite_source(),
        log_interval=log_interval,
        buffer=buffer,
        stage_configs=stage_configs,
    )

    try:
        start_time = time.monotonic()
        item_count = 0

        with pipeline.auto_stop():
            for _ in pipeline:
                item_count += 1

                # Simulate foreground work (e.g., training loop)
                time.sleep(foreground_sleep)

                # Check if we've run long enough
                elapsed = time.monotonic() - start_time
                if elapsed >= run_duration:
                    break

        elapsed = time.monotonic() - start_time
        print(f"\n✅ Configuration completed in {elapsed:.2f} seconds")
        print(f"   Items processed: {item_count}")
        print(f"   Average throughput: {item_count / elapsed:.2f} items/sec")

    finally:
        # Ensure all stats are flushed to database
        writer.shutdown()

        # Log stats summary
        log_stats_summary(db_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pipeline simulation configuration.

    Returns:
        Parsed arguments including database path, stage configurations,
        log interval, run duration, and foreground sleep time.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to save the SQLite database file",
    )
    parser.add_argument(
        "--stage-configs",
        type=str,
        required=True,
        help='Stage configurations as "sleep1,concurrency1;sleep2,concurrency2;..." (e.g., "0.050,1;0.030,1;0.010,1")',
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Interval in seconds for logging stats to database (default: 1)",
    )
    parser.add_argument(
        "--run-duration",
        type=float,
        default=10.0,
        help="Duration in seconds to run the configuration (default: 10)",
    )
    parser.add_argument(
        "--foreground-sleep",
        type=float,
        default=0.030,
        help="Sleep duration in foreground thread in seconds (default: 0.030 = 30ms)",
    )
    return parser.parse_args()


def parse_stage_configs(config_str: str) -> list[tuple[float, int, int]]:
    """Parse stage configuration string into list of tuples.

    Args:
        config_str: Configuration string like "0.050,1,1;0.030,1,4;0.010,1,1;" where each stage
                   has format "sleep,concurrency,batch_size". Batch_size is optional (defaults to 1).
                   Examples:
                   - "0.050,1;0.030,1" - two stages with batch_size=1 (no aggregation)
                   - "0.050,1,1;0.030,1,4" - stage 1 aggregates 4 items into 1

    Returns:
        List of (sleep_duration, concurrency, batch_size) tuples

    Raises:
        ValueError: If the configuration string is invalid
    """
    try:
        stages = []
        for stage_str in config_str.split(";"):
            stage_str = stage_str.strip()
            if not stage_str:
                continue
            parts = stage_str.split(",")
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(
                    f"Invalid stage format '{stage_str}'. Expected 'sleep,concurrency[,batch_size]'"
                )
            sleep_duration = float(parts[0])
            concurrency = int(parts[1])
            batch_size = int(parts[2]) if len(parts) == 3 else 1

            if sleep_duration < 0:
                raise ValueError(
                    f"Sleep duration must be non-negative: {sleep_duration}"
                )
            if concurrency < 1:
                raise ValueError(f"Concurrency must be at least 1: {concurrency}")
            if batch_size < 1:
                raise ValueError(f"Batch size must be at least 1: {batch_size}")

            stages.append((sleep_duration, concurrency, batch_size))
        if not stages:
            raise ValueError("At least one stage configuration is required")
        return stages
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid stage configuration '{config_str}': {e}. "
            "Expected format: 'sleep1,concurrency1[,batch_size1];sleep2,concurrency2[,batch_size2];...'"
        ) from e


def main() -> None:
    """Main entry point for the pipeline performance simulation.

    Parses command line arguments, builds a pipeline with the specified
    configuration, runs the simulation, and saves performance statistics
    to a SQLite database.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname).1s]: %(message)s",
    )

    args = parse_args()

    # Parse stage configurations from CLI
    try:
        stage_configs = parse_stage_configs(args.stage_configs)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Ensure output directory for database exists
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Pipeline Bottleneck Simulation")
    print(f"   Database file: {args.db_path}")
    print(f"   Stage configs: {stage_configs}")
    print()

    # Run the configuration
    run_configuration(
        db_path=args.db_path,
        stage_configs=stage_configs,
        log_interval=args.log_interval,
        run_duration=args.run_duration,
        foreground_sleep=args.foreground_sleep,
    )

    print("\n" + "=" * 80)
    print("🎉 Simulation completed!")
    print("=" * 80)
    print(f"\nDatabase file created: {args.db_path}")
    print()


if __name__ == "__main__":
    main()
