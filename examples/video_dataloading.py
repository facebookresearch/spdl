# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This example uses SPDL to decode and batch video frames, then send them to GPU.

The structure of the pipeline is identical to that of
:py:mod:`image_dataloading`.

Basic Usage
-----------

Running this example requires a dataset consists of videos.

For example, to run this example with Kinetics dataset.

1. Download Kinetics dataset.
   https://github.com/cvdfoundation/kinetics-dataset provides scripts to facilitate this.
2. Create a list containing the downloaded videos.

   .. code-block::

      cd /data/users/moto/kinetics-dataset/k400/
      find train -name '*.mp4' > ~/imagenet.train.flist

3. Run the script.

   .. code-block:: shell

      python examples/video_dataloading.py
        --input-flist ~/kinetics400.train.flist
        --prefix /data/users/moto/kinetics-dataset/k400/
        --num-threads 8

Using GPU video decoder
-----------------------

When SPDL is built with NVDEC integration enabled, and the GPUs support NVDEC,
providing ``--nvdec`` option switches the video decoder to NVDEC, using
:py:func:`spdl.io.decode_packets_nvdec`. When using this option, adjust the
number of threads (the number of concurrent decoding) to accomodate
the number of hardware video decoder availabe on GPUs.
For the details, please refer to https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new

.. note::

   This example decodes videos from the beginning to the end, so using NVDEC
   speeds up the whole decoding speed. But in cases where framees are sampled,
   CPU decoding with higher concurrency often yields higher throughput.
"""

# pyre-ignore-all-errors

import logging
import signal
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import spdl.io
import spdl.utils
import torch
from spdl.dataloader import Pipeline, PipelineBuilder
from torch import Tensor

_LG = logging.getLogger(__name__)

__all__ = [
    "entrypoint",
    "worker_entrypoint",
    "benchmark",
    "source",
    "decode_video",
    "decode_video_nvdec",
    "get_pipeline",
    "PerfResult",
]


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=float("inf"))
    parser.add_argument("--prefix", default="")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--nvdec", action="store_true")
    args = parser.parse_args(args)
    if args.trace:
        args.max_samples = 320
    return args


def source(
    input_flist: str,
    prefix: str,
    max_samples: int,
    split_size: int = 1,
    split_id: int = 0,
) -> Iterable[str]:
    """Iterate a file containing a list of paths, while optionally skipping some.

    Args:
        input_flist: A file contains list of video paths.
        prefix: Prepended to the paths in the list.
        max_samples: The maximum number of items to yield.
        split_size: Split the paths in to this number of subsets.
        split_id: The index of this split. Paths at ``line_number % split_size == split_id`` are returned.

    Yields:
        The paths of the specified split.
    """
    with open(input_flist, "r") as f:
        num_yielded = 0
        for i, line in enumerate(f):
            if i % split_size != split_id:
                continue
            if line := line.strip():
                yield prefix + line

                if (num_yielded := num_yielded + 1) >= max_samples:
                    return


async def decode_video(
    src: str | bytes,
    width: int,
    height: int,
    device_index: int,
) -> Tensor:
    """Decode video and send decoded frames to GPU.

    Args:
        src: Data source. Passed to :py:func:`spdl.io.demux_video`.
        width, height: The target resolution.
        device_index: The index of the target GPU.

    Returns:
        A GPU tensor represents decoded video frames.
        The dtype is uint8, the shape is ``[N, C, H, W]``, where ``N`` is the number
        of frames in the video, ``C`` is RGB channels.
    """
    packets = await spdl.io.async_demux_video(src)
    frames = await spdl.io.async_decode_packets(
        packets,
        filter_desc=spdl.io.get_filter_desc(
            packets,
            scale_width=width,
            scale_height=height,
            pix_fmt="rgb24",
        ),
    )
    buffer = await spdl.io.async_convert_frames(frames)
    buffer = await spdl.io.async_transfer_buffer(
        buffer,
        device_config=spdl.io.cuda_config(
            device_index=device_index,
            allocator=(
                torch.cuda.caching_allocator_alloc,
                torch.cuda.caching_allocator_delete,
            ),
        ),
    )
    return spdl.io.to_torch(buffer).permute(0, 2, 3, 1)


async def decode_video_nvdec(
    src: str,
    device_index: int,
    width: int,
    height: int,
):
    """Decode video using NVDEC.

    Args:
        src: Data source. Passed to :py:func:`spdl.io.demux_video`.
        device_index: The index of the target GPU.
        width, height: The target resolution.

    Returns:
        A GPU tensor represents decoded video frames.
        The dtype is uint8, the shape is ``[N, C, H, W]``, where ``N`` is the number
        of frames in the video, ``C`` is RGB channels.
    """
    packets = await spdl.io.async_demux_video(src)
    buffer = await spdl.io.async_decode_packets_nvdec(
        packets,
        device_config=spdl.io.cuda_config(
            device_index=device_index,
            allocator=(
                torch.cuda.caching_allocator_alloc,
                torch.cuda.caching_allocator_delete,
            ),
        ),
        width=width,
        height=height,
        pix_fmt="rgba",
    )
    return spdl.io.to_torch(buffer)[..., :3].permute(0, 2, 3, 1)


def _get_decode_fn(device_index, use_nvdec, width=222, height=222):
    if use_nvdec:

        async def _decode_func(src):
            return await decode_video_nvdec(src, device_index, width, height)

    else:

        async def _decode_func(src):
            return await decode_video(src, width, height, device_index)

    return _decode_func


def get_pipeline(
    src: Iterable[str],
    decode_fn: Callable[[str], Tensor],
    decode_concurrency: int,
    num_threads: int,
    buffer_size: int = 3,
) -> Pipeline:
    """Construct the video loading pipeline.

    Args:
        src: Pipeline source. Generator that yields image paths. See :py:func:`source`.
        decode_fn: Function that decode the given image and send the decoded frames to GPU.
        decode_concurrency: The maximum number of decoding scheduled concurrently.
        num_threads: The number of threads in the pipeline.
        buffer_size: The size of buffer for the resulting batch image Tensor.
    """
    return (
        PipelineBuilder()
        .add_source(src)
        .pipe(decode_fn, concurrency=decode_concurrency, report_stats_interval=15)
        .add_sink(buffer_size)
        .build(num_threads=num_threads)
    )


def _get_pipeline(args):
    src = source(
        input_flist=args.input_flist,
        prefix=args.prefix,
        max_samples=args.max_samples,
        split_id=args.worker_id,
        split_size=args.num_workers,
    )

    decode_fn = _get_decode_fn(args.worker_id, args.nvdec)
    pipeline = get_pipeline(
        src,
        decode_fn,
        decode_concurrency=args.num_threads,
        num_threads=args.num_threads + 3,
        buffer_size=args.queue_size,
    )
    print(pipeline)
    return pipeline


@dataclass
class PerfResult:
    """Used to report the worker performance to the main process."""

    elapsed: float
    """The time it took to process all the inputs."""

    num_batches: int
    """The number of batches processed."""

    num_frames: int
    """The number of frames processed."""


def benchmark(
    dataloader: Iterable[Tensor],
    stop_requested: Event,
) -> PerfResult:
    """The main loop that measures the performance of dataloading.

    Args:
        dataloader: The dataloader to benchmark.
        stop_requested: Used to interrupt the benchmark loop.

    Returns:
        The performance result.
    """
    t0 = time.monotonic()
    num_frames = num_batches = 0
    try:
        for batches in dataloader:
            for batch in batches:
                num_frames += batch.shape[0]
                num_batches += 1

            if stop_requested.is_set():
                break

    finally:
        elapsed = time.monotonic() - t0
        fps = num_frames / elapsed
        _LG.info(f"FPS={fps:.2f} ({num_frames} / {elapsed:.2f}), (Done {num_frames})")

    return PerfResult(elapsed, num_batches, num_frames)


def worker_entrypoint(args: list[str]) -> PerfResult:
    """Entrypoint for worker process. Load images to a GPU and measure its performance.

    It builds a Pipeline object using :py:func:`get_pipeline` function and run it with
    :py:func:`benchmark` function.
    """
    args = _parse_args(args)
    _init(args.debug, args.worker_id)

    _LG.info(args)

    pipeline = _get_pipeline(args)

    device = torch.device(f"cuda:{args.worker_id}")

    ev = Event()

    def handler_stop_signals(_signum, _frame):
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


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
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
    )
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_known_args(args)


def entrypoint(args: list[str] | None = None):
    """CLI entrypoint. Launch the worker processes, each of which load videos and send them to GPU."""
    ns, args = _parse_process_args(args)

    args_set = [
        [*args, f"--worker-id={i}", f"--num-workers={ns.num_workers}"]
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
