#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This examlple shows how to collect runtime performance statistics using TensorBoard.

.. image:: ../_static/data/example_performance_analysis_tensorboard.png
   :alt: An example of how stats shown using TensorBoard UI.

.. note::

   To learn how to interpret the performance statistics please refer to
   `Optimization Guide <../performance_analysis/index.html>`_.

.. py:currentmodule:: spdl.pipeline

The :py:class:`Pipeline` class can collect runtime statistics
and periodically publish it.
This example shows how to write a callback function for publishing the stats, and
how to attach the callback to the Pipeline object.

The performance stats are exposed as :py:class:`TaskPerfStats` and
:py:class:`QueuePerfStats` classes.

You can add custom callbacks by following steps.

For :py:class:`QueuePerfStats`:

#. Subclass :py:class:`StatsQueue` and
   override :py:meth:`StatsQueue.interval_stats_callback`.
#. In the ``interval_stats_callback`` method,
   save the fields of ``QueuePerfStats`` to somewhere you can access later.
#. Provide the new class to :py:meth:`PipelineBuilder.build` method.

Similarly for :py:class:`TaskPerfStats`

#. Subclass :py:class:`TaskStatsHook` and
   override :py:meth:`TaskStatsHook.interval_stats_callback`.
#. In the ``interval_stats_callback`` method,
   save the fields of ``TaskPerfStats`` to somewhere you can access later.
#. Create a factory function that takes a name of the stage functoin and
   return a list of :py:class:`TaskHook`-s applied to the stage.
#. Provide the factory function to :py:meth:`PipelineBuilder.build` method.
"""

import argparse
import asyncio
import contextlib
import logging
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path

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
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "parse_args",
    "main",
    "build_pipeline",
    "decode",
    "CustomTaskHook",
    "CustomQueue",
]

# pyre-strict


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=120,
    )
    return parser.parse_args()


class CustomTaskHook(TaskStatsHook):
    """Extend the :py:class:`~spdl.pipeline.TaskStatsHook` to add logging to TensorBoard.

    This class extends the :py:class:`~spdl.pipeline.TaskStatsHook`, and periodically
    writes the performance statistics to TensorBoard.

    Args:
        name, interval: See :py:class:`spdl.pipeline.TaskStatsHook`
        writer: TensorBoard summary writer object.
    """

    def __init__(
        self,
        name: str,
        interval: float = -1,
        *,
        writer: SummaryWriter,
    ) -> None:
        super().__init__(name=name, interval=interval)
        self._writer = writer
        self._step = -1

    async def interval_stats_callback(self, stats: TaskPerfStats) -> None:
        """Log the performance statistics to TensorBoard.

        Args:
            stats: See :py:meth:`spdl.pipeline.TaskStatsHook.interval_stats_callback`.
        """
        await super().interval_stats_callback(stats)

        self._step += 1
        walltime = time.time()
        vals = {
            "pipeline_task/ave_time": stats.ave_time,
            "pipeline_task/num_invocations": stats.num_tasks,
            "pipeline_task/num_failures": stats.num_failures,
        }
        await asyncio.get_running_loop().run_in_executor(
            None, _log, self._writer, self.name, vals, self._step, walltime
        )


class CustomQueue(StatsQueue):
    """Extend the :py:class:`~spdl.pipeline.StatsQueue` to add logging TensorBoard.


    This class extends the :py:class:`~spdl.pipeline.TaskStatsHook`, and periodically
    writes the performance statistics to TensorBoard.

    Args:
        name, buffer_size, interval: See :py:class:`spdl.pipeline.StatsQueue`
        writer: TensorBoard summary writer object.
    """

    def __init__(
        self,
        name: str,
        buffer_size: int = 1,
        interval: float = -1,
        *,
        writer: SummaryWriter,
    ) -> None:
        super().__init__(name=name, buffer_size=buffer_size, interval=interval)
        self._writer = writer
        self._step = -1

    async def interval_stats_callback(self, stats: QueuePerfStats) -> None:
        """Log the performance statistics to TensorBoard.

        Args:
            stats: See :py:meth:`spdl.pipeline.StatsQueue.interval_stats_callback`.
        """
        await super().interval_stats_callback(stats)

        self._step += 1
        walltime = time.time()

        vals = {
            "pipeline_queue/qps": stats.qps,
            "pipeline_queue/ave_put_time": stats.ave_put_time,
            "pipeline_queue/ave_get_time": stats.ave_get_time,
            "pipeline_queue/occupancy_rate": stats.occupancy_rate,
        }
        await asyncio.get_running_loop().run_in_executor(
            None, _log, self._writer, self.name, vals, self._step, walltime
        )


def _log(
    writer: SummaryWriter,
    name: str,
    vals: dict[str, float],
    step: int | None,
    walltime: float,
) -> None:
    for k, v in vals.items():
        writer.add_scalars(k, {name: v}, global_step=step, walltime=walltime)


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
    source: Iterator[Path],
    writer: SummaryWriter,
    log_interval: float,
    concurrency: int,
) -> Pipeline:
    """Build the pipeline using :py:class`CustomTaskHook`, and :py:class:`CustomQueue`.

    Args:
        source: A data source.
        writer: A TensorBoard SummaryWriter object.
        log_interval: The interval (in second) the performance data is saved.
        concurrency: The concurrency for video decoding.
    """

    def hook_factory(name: str) -> list[TaskHook]:
        return [CustomTaskHook(name=name, interval=log_interval, writer=writer)]

    return (
        PipelineBuilder()
        .add_source(source=source)
        .pipe(decode, concurrency=concurrency)
        .add_sink()
        .build(
            num_threads=concurrency,
            queue_class=partial(
                CustomQueue,
                writer=writer,
                interval=log_interval,
            ),
            task_hook_factory=hook_factory,
        )
    )


def main() -> None:
    """The main entry point for the example."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    source = args.dataset_dir.rglob("*.mp4")
    with contextlib.closing(SummaryWriter(args.log_dir)) as writer:
        pipeline = build_pipeline(
            source=source,
            writer=writer,
            log_interval=args.log_interval,
            concurrency=8,
        )
        with pipeline.auto_stop():
            for _ in pipeline:
                pass


if __name__ == "__main__":
    main()
