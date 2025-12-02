#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This example shows how to collect runtime performance statistics using
custom hooks and then export the data to a database.

.. note::

   To learn how to interpret the performance statistics, please refer to
   `Optimization Guide <../performance_analysis/index.html>`_.

The :py:class:`~spdl.pipeline.Pipeline` class can collect runtime statistics
and periodically publish them via hooks and queues.
This example shows how to use a buffer-based logging pattern with
SQLite as the storage backend to collect and query performance statistics.

The performance stats are exposed as :py:class:`spdl.pipeline.TaskPerfStats` and
:py:class:`spdl.pipeline.QueuePerfStats` classes. These are collected via hooks
and queues into a shared buffer, which is then asynchronously written
to a SQLite database by a dedicated writer thread.

Architecture
------------

The following diagram illustrates the relationship between the pipeline, buffer, and writer thread:

.. mermaid::

   graph LR
       subgraph Pipeline
           A[Task Hooks]
           C[Queues]
       end
       A -->|Task Stats| B[Shared Buffer]
       C -->|Queue Stats| B
       D[GC Events] -->|Event Logs| B
       B -->|Async Write| E[Writer Thread]
       E -->|Persist| F[(SQLite Database)]
       F -->|Query| G[performance_analysis_plot.py]

Example Usage
-------------

This example demonstrates:

#. Using :py:class:`TaskStatsHookWithLogging` to log task performance to a buffer
#. Using :py:class:`StatsQueueWithLogging` to log queue performance to a buffer
#. Using :py:class:`SQLiteStatsWriter` to asynchronously flush the buffer to SQLite

And the accompanying plot script demonstrates:

#. Querying the collected statistics from the database using standalone query functions
#. Analyzing pipeline performance over time

Visualization
-------------

The SQLite database can be queried using standard SQL tools or the
built-in query API to analyze:

- Task execution times and failure rates
- Queue throughput (QPS) and occupancy rates
- Pipeline bottlenecks and resource utilization

You can use the ``performance_analysis_plot.py`` script to query the data from
the database and generate plots for visualization and analysis.
"""

import argparse
import gc
import logging
import time
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from queue import Queue

import spdl.io
import torch
from spdl.pipeline import (
    Pipeline,
    PipelineBuilder,
    QueuePerfStats,
    StatsQueue,
    TaskHook,
    TaskPerfStats,
    TaskStatsHook,
)

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


__all__ = [
    "parse_args",
    "main",
    "build_pipeline",
    "decode",
    "TaskStatsHookWithLogging",
    "StatsQueueWithLogging",
    "SQLiteStatsWriter",
]

# pyre-strict

_LG: logging.Logger = logging.getLogger(__name__)


class TaskStatsHookWithLogging(TaskStatsHook):
    """Task hook that logs statistics to a shared buffer.

    This hook collects task performance statistics and writes them to a
    plain Python queue.
    Args:
        name: Name of the stage/task.
        buffer: Shared queue to write stats entries. This queue is typically
            consumed by a separate writer (e.g., ``SQLiteStatsWriter``).
        interval: The interval (in seconds) to report the performance stats
            periodically.
    """

    def __init__(
        self,
        name: str,
        buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry],
        interval: float = 59,
    ) -> None:
        super().__init__(name=name, interval=interval)
        self._buffer = buffer

    async def interval_stats_callback(self, stats: TaskPerfStats) -> None:
        """Log interval statistics to the buffer."""
        await super().interval_stats_callback(stats)

        entry = TaskStatsLogEntry(
            timestamp=time.time(),
            name=self.name,
            stats=stats,
        )
        self._buffer.put(entry)


class StatsQueueWithLogging(StatsQueue):
    """Queue that logs statistics to a shared buffer.

    This queue collects queue performance statistics and writes them to a
    plain Python queue.

    Args:
        name: Name of the queue. Assigned by PipelineBuilder.
        buffer: Shared queue to write stats entries. This queue is typically
            consumed by a separate writer (e.g., ``SQLiteStatsWriter``).
        buffer_size: The buffer size. Assigned by PipelineBuilder.
        interval: The interval (in seconds) to report the performance stats
            periodically.
    """

    def __init__(
        self,
        name: str,
        buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry],
        buffer_size: int = 1,
        interval: float = 59,
    ) -> None:
        super().__init__(name=name, buffer_size=buffer_size, interval=interval)
        self._buffer = buffer

    async def interval_stats_callback(self, stats: QueuePerfStats) -> None:
        """Log interval statistics to the buffer."""
        await super().interval_stats_callback(stats)

        entry = QueueStatsLogEntry(
            timestamp=time.time(),
            name=self.name,
            stats=stats,
        )
        self._buffer.put(entry)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing video files (*.mp4)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("pipeline_stats.db"),
        help="Path to SQLite database for stats (default: pipeline_stats.db)",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=60,
        help="Interval in seconds for logging stats to database (default: 60)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent decoding tasks (default: 10)",
    )
    return parser.parse_args()


def decode(path: Path, width: int = 128, height: int = 128) -> torch.Tensor:
    """Decode the video from the given path with rescaling.

    Args:
        path: The path to the video file.
        width,height: The resolution of video after rescaling.

    Returns:
        Uint8 tensor in shape of ``[N, C, H, W]``: Video frames in Tensor.
    """
    packets = spdl.io.demux_video(path)
    frames = spdl.io.decode_packets(
        packets,
        filter_desc=spdl.io.get_filter_desc(
            packets,
            scale_width=width,
            scale_height=height,
            pix_fmt="rgb24",
        ),
    )
    buffer = spdl.io.convert_frames(frames)
    return spdl.io.to_torch(buffer).permute(0, 2, 3, 1)


def build_pipeline(
    source: Iterable[Path],
    log_interval: float,
    concurrency: int,
    buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry],
) -> Pipeline:
    """Build the pipeline with stats logging to a buffer.

    Args:
        source: A data source containing video file paths.
        log_interval: The interval (in seconds) the performance data is saved.
        concurrency: The concurrency for video decoding.
        buffer: Shared queue for collecting stats entries. This queue should
            be consumed by a ``SQLiteStatsWriter`` instance.

    Returns:
        A configured Pipeline instance ready for execution.
    """

    def hook_factory(name: str) -> list[TaskHook]:
        return [
            TaskStatsHookWithLogging(
                name=name,
                buffer=buffer,
                interval=log_interval,
            )
        ]

    return (
        PipelineBuilder()
        .add_source(source=source)
        .pipe(decode, concurrency=concurrency)
        .add_sink()
        .build(
            num_threads=concurrency,
            queue_class=partial(  # pyre-ignore[6]
                StatsQueueWithLogging,
                buffer=buffer,
                interval=log_interval,
            ),
            task_hook_factory=hook_factory,
        )
    )


def _validate_dataset(dataset_dir: Path) -> list[Path]:
    """Validate dataset directory and return list of video files.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        List of video file paths.

    Raises:
        ValueError: If the directory doesn't exist or contains no videos.
    """
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    video_files = list(dataset_dir.rglob("*.mp4"))
    if not video_files:
        raise ValueError(f"No *.mp4 files found in {dataset_dir}")

    return video_files


def setup_gc_callbacks(
    buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry],
) -> None:
    """Set up garbage collection callbacks to log GC events.

    Args:
        buffer: Shared queue for collecting stats entries.
    """

    def gc_callback(phase: str, _info: dict[str, int]) -> None:
        """Callback invoked during garbage collection phases.

        Args:
            phase: The GC phase ('start' or 'stop').
            _info: Dictionary containing GC information (unused).
        """
        event_name = f"gc_{phase}"
        entry = EventLogEntry(
            timestamp=time.time(),
            event_name=event_name,
        )
        buffer.put(entry)
        _LG.debug("GC event logged: %s", event_name)

    # Register the callback with the garbage collector
    gc.callbacks.append(gc_callback)
    _LG.info("Garbage collection callbacks registered")


def main() -> None:
    """The main entry point for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname).1s] - %(message)s",
    )

    args = parse_args()

    # Validate dataset and get video files
    video_files = _validate_dataset(args.dataset_dir)

    print("\n🎬 Starting video processing pipeline")
    print(f"   Dataset: {args.dataset_dir}")
    print(f"   Videos found: {len(video_files)}")
    print(f"   Database: {args.db_path}")
    print(f"   Concurrency: {args.concurrency}")
    print(f"   Log interval: {args.log_interval}s\n")
    print("", flush=True)

    # Create shared buffer and writer
    buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry] = Queue()
    writer = SQLiteStatsWriter(
        str(args.db_path),
        buffer,
        flush_interval=max(0.1, args.log_interval - 1),
    )
    writer.start()

    # Set up garbage collection callbacks
    setup_gc_callbacks(buffer)

    # Build and run pipeline
    pipeline = build_pipeline(
        source=video_files,
        log_interval=args.log_interval,
        concurrency=args.concurrency,
        buffer=buffer,
    )

    try:
        start_time = time.monotonic()
        with pipeline.auto_stop():
            for _ in pipeline:
                pass

        elapsed = time.monotonic() - start_time
        print(f"\n✅ Pipeline completed in {elapsed:.2f} seconds")

    finally:
        # Ensure all stats are flushed to database
        writer.shutdown()

        # Log stats summary
        log_stats_summary(args.db_path)


if __name__ == "__main__":
    main()
