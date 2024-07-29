#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the performance of loading images from local file system to GPUs.

Given a list of image files to process, this script spawns subprocesses,
each of which load images and send them to the corresponding GPUs, then
collect the runtime statistics.

.. mermaid::

   flowchart
       A[Main Process]
       subgraph P1[Worker Process 1]
           subgraph TP1[Thread Pool]
              t11[Thread]
              t12[Thread]
           end
       end
       G1[GPU 1]

       subgraph P3[Worker Process N]
           subgraph TP3[Thread Pool]
              t31[Thread]
              t32[Thread]
           end
       end
       G3[GPU N]

       A --> P1
       A --> P3
       t11 --> G1
       t12 --> G1
       t31 --> G3
       t32 --> G3

A file list can be created, for example, by:

.. code-block:: bash

   cd /data/users/moto/imagenet/
   find train -name '*.JPEG' > ~/imagenet.train.flist

To run the benchmark,  pass it to the script like the following.

.. code-block::

   python image_dataloading.py
       --input-flist ~/imagenet.train.flist
       --prefix /data/users/moto/imagenet/
       --num-workers 8 # The number of GPUs
"""

# pyre-ignore-all-errors

import logging
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import spdl.io
import spdl.utils

import torch
from spdl.dataloader import Pipeline, PipelineBuilder
from spdl.io import CUDAConfig
from torch import Tensor

_LG = logging.getLogger(__name__)

__all__ = [
    "entrypoint",
    "worker_entrypoint",
    "benchmark",
    "source",
    "batch_decode",
    "get_pipeline",
    "PerfResult",
]


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prefix")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--buffer-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args(args)
    if args.trace:
        args.max_samples = args.batch_size * 40
    return args


def source(
    path: Path,
    prefix: str = "",
    split_size: int = 1,
    split_id: int = 0,
) -> Iterator[str]:
    """Iterate a file containing a list of paths, while optionally skipping some.

    Args:
        path: Path to the file containing list of file paths.
        prefix: Prepended to the paths in the list.
        split_size: Split the paths in to this number of subsets.
        split_id: The index of this split.
            Paths at ``line_number % split_size == split_id`` are returned.

    Yields:
        Path: The paths of the specified split.
    """
    with open(path) as f:
        for i, line in enumerate(f):
            if i % split_size == split_id:
                if line := line.strip():
                    yield prefix + line


async def batch_decode(
    srcs: list[str],
    width: int = 224,
    height: int = 224,
    cuda_config: spdl.io.CUDAConfig | None = None,
) -> Tensor:
    """Given image paths, decode, resize, batch and optionally send them to GPU.

    Args:
        srcs: List of image paths.
        width, height: The size of the images to batch.
        cuda_config: When provided, the data are sent to the specified GPU.

    Returns:
        The batch tensor.
    """
    buffer = await spdl.io.async_load_image_batch(
        srcs,
        width=width,
        height=height,
        pix_fmt="rgb24",
        cuda_config=cuda_config,
        strict=False,
    )
    return spdl.io.to_torch(buffer)


def get_pipeline(
    src: Iterator[str],
    batch_size: int,
    cuda_config: CUDAConfig,
    buffer_size: int,
    num_threads: int,
) -> Pipeline:
    """Build image data loading pipeline.

    The pipeline uses :py:func:`batch_decode` for decoding images concurrently
    and send the resulting data to GPU.

    Args:
        src: Pipeline source. Generator that yields image paths.
            See :py:func:`source`.
        batch_size: The number of images in a batch.
        cuda_config: The configuration of target CUDA device.
        buffer_size: The size of buffer for the resulting batch image Tensor.
        num_threads: The number of threads in the pipeline.

    Returns:
        The pipeline that performs batch image decoding and device transfer.
    """

    async def _batch_decode(srcs):
        return await batch_decode(srcs, cuda_config=cuda_config)

    pipeline = (
        PipelineBuilder()
        .add_source(src)
        .aggregate(batch_size)
        .pipe(_batch_decode, concurrency=num_threads, report_stats_interval=15)
        .add_sink(buffer_size)
        .build(num_threads=num_threads)
    )
    return pipeline


def _get_pipeline(args):
    return get_pipeline(
        source(args.input_flist, args.prefix, args.num_workers, args.worker_id),
        args.batch_size,
        cuda_config=(
            None
            if args.worker_id is None
            else spdl.io.cuda_config(
                device_index=args.worker_id,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            )
        ),
        buffer_size=args.buffer_size,
        num_threads=args.num_threads,
    )


@dataclass
class PerfResult:
    """Used to report the worker performance to the main process."""

    elapsed: float
    """The time it took to process all the inputs."""

    num_batches: int
    """The number of batches processed."""

    num_frames: int
    """The number of frames processed."""


def worker_entrypoint(args: list[str]) -> PerfResult:
    """Entrypoint for worker process. Load images to a GPU and measure its performance.

    It builds a :py:class:`~spdl.dataloader.Pipeline` object using :py:func:`get_pipeline`
    function and run it with :py:func:`benchmark` function.
    """
    args = _parse_args(args)
    _init(args.debug, args.worker_id)

    _LG.info(args)

    pipeline = _get_pipeline(args)
    print(pipeline)

    device = torch.device(f"cuda:{args.worker_id}")

    ev = Event()

    def handler_stop_signals(signum, frame):
        ev.set()

    signal.signal(signal.SIGTERM, handler_stop_signals)

    # Warm up
    torch.zeros([1, 1], device=device)

    trace_path = f"{args.trace}.{args.worker_id}"
    with (
        pipeline.auto_stop(),
        spdl.utils.tracing(trace_path, enable=args.trace is not None),
    ):
        return benchmark(pipeline.get_iterator(), ev)


def benchmark(loader: Iterator[Tensor], stop_requested: Event) -> PerfResult:
    """The main loop that measures the performance of dataloading.

    Args:
        loader: The dataloader to benchmark.
        stop_requested: Used to interrupt the benchmark loop.

    Returns:
        The performance result.
    """
    t0 = time.monotonic()
    num_frames = num_batches = 0
    try:
        for batch in loader:
            num_frames += batch.shape[0]
            num_batches += 1

            if stop_requested.is_set():
                break

    finally:
        elapsed = time.monotonic() - t0

    return PerfResult(elapsed, num_batches, num_frames)


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


def _init(debug, worker_id):
    _init_logging(debug, worker_id)

    spdl.utils.set_ffmpeg_log_level(16)


def _parse_process_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_known_args(args)


def entrypoint(args: list[str] | None = None):
    """CLI entrypoint. Launch the worker processes,
    each of which load images and send them to GPU."""
    ns, args = _parse_process_args(args)

    args_set = [
        args + [f"--worker-id={i}", f"--num-workers={ns.num_workers}"]
        for i in range(ns.num_workers)
    ]

    from multiprocessing import Pool

    with Pool(processes=ns.num_workers) as pool:
        _init_logging()
        _LG.info("Spawned: %d workers", ns.num_workers)

        vals = pool.map(worker_entrypoint, args_set)

    ave_time = sum(v.elapsed for v in vals) / len(vals)
    total_frames = sum(v.num_frames for v in vals)
    total_batches = sum(v.num_batches for v in vals)

    _LG.info(f"{ave_time=:.2f}, {total_frames=}, {total_batches=}")

    FPS = total_frames / ave_time
    BPS = total_batches / ave_time
    _LG.info(f"Aggregated {FPS=:.2f}, {BPS=:.2f}")


if __name__ == "__main__":
    entrypoint()
