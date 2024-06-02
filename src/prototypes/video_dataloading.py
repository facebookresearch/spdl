"""Experiment for running asyncio.loop in background thread and decode media"""

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import spdl.io
import spdl.utils
import torch
from spdl.dataloader import apply_async, BackgroundGenerator
from spdl.dataloader._utils import _iter_flist

_LG = logging.getLogger(__name__)


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--nvdec", action="store_true")
    args = parser.parse_args(args)
    if args.trace:
        args.max_samples = args.batch_size * 10
    return args


def _get_decode_fn(cuda_device_index, width=222, height=222, pix_fmt="rgba"):
    async def _decode_window(packets):
        frames = await spdl.io.async_decode_packets(
            packets,
            filter_desc=spdl.io.get_filter_desc(
                packets,
                scale_width=width,
                scale_height=height,
                pix_fmt=pix_fmt,
            ),
        )
        buffer = await spdl.io.async_convert_frames(frames)
        buffer = await spdl.io.async_transfer_buffer(
            buffer,
            transfer_config=spdl.io.transfer_config(
                device_index=cuda_device_index,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            ),
        )
        return buffer

    async def _decode(srcs):
        src = srcs[0]
        timestamps = [(0, float("inf"))]
        demuxer = spdl.io.async_streaming_demux_video(src, timestamps=timestamps)
        tasks = []
        async for packets in demuxer:
            tasks.append(asyncio.create_task(_decode_window(packets)))

        await asyncio.wait(tasks)
        buffers = []
        for task in tasks:
            try:
                buffers.append(task.result())
            except asyncio.CancelledError:
                _LG.error("Task %s was cancelled", task)
            except Exception as err:
                _LG.error("Failed to process task %s: %s", task.get_name(), err)
        return [spdl.io.to_torch(buffer) for buffer in buffers]

    return _decode


def _get_decode_nvdec_fn(cuda_device_index, width=222, height=222, pix_fmt="rgba"):
    async def _decode_nvdec(srcs):
        src = srcs[0]
        timestamps = [(0, float("inf"))]
        demuxer = spdl.io.async_streaming_demux_video(src, timestamps=timestamps)
        tasks = []
        async for packets in demuxer:
            coro = spdl.io.async_decode_packets_nvdec(
                packets,
                cuda_device_index=cuda_device_index,
                width=width,
                height=height,
                pix_fmt=pix_fmt,
            )
            tasks.append(asyncio.create_task(coro))

        await asyncio.wait(tasks)
        buffers = []
        for task in tasks:
            try:
                buffers.append(task.result())
            except asyncio.CancelledError:
                _LG.error("Task %s was cancelled", task)
            except Exception as err:
                _LG.error("Failed to process task %s: %s", task.get_name(), err)
        return [spdl.io.to_torch(buffer) for buffer in buffers]

    return _decode_nvdec


def _get_batch_generator(args):
    srcs_gen = _iter_flist(
        args.input_flist,
        prefix=args.prefix,
        n=args.worker_id,
        N=args.num_workers,
        max=args.max_samples,
    )
    _decode_fn = (
        _get_decode_nvdec_fn(args.worker_id)
        if args.nvdec
        else _get_decode_fn(args.worker_id)
    )
    return apply_async(_decode_fn, srcs_gen)


@dataclass
class PerfResult:
    elapsed: float
    num_batches: int
    num_frames: int


def _iter_dataloader(dataloader, ev):
    t0 = t_int = time.monotonic()
    num_frames = num_frames_int = 0
    num_batches = 0
    try:
        for batches in dataloader:
            for batch in batches:
                num_frames += batch.shape[0]
                num_batches += 1

            t1 = time.monotonic()
            if (elapsed := t1 - t_int) > 10:
                n = num_frames - num_frames_int
                _LG.info(f"Interval FPS={n / elapsed:.2f} (Done {num_batches=})")

                t_int = t1
                num_frames_int = num_frames

            if ev.is_set():
                break

    finally:
        elapsed = time.monotonic() - t0
        fps = num_frames / elapsed
        _LG.info(f"FPS={fps:.2f} ({num_frames} / {elapsed:.2f}), (Done {num_frames})")

    return PerfResult(elapsed, num_batches, num_frames)


def _benchmark(args):
    args = _parse_args(args)
    _init(args.debug, args.worker_id)

    _LG.info(args)

    batch_gen = _get_batch_generator(args)

    device = torch.device(f"cuda:{args.worker_id}")

    ev = Event()

    def handler_stop_signals(signum, frame):
        ev.set()

    signal.signal(signal.SIGTERM, handler_stop_signals)

    # Warm up
    torch.zeros([1, 1], device=device)

    trace_path = f"{args.trace}.{args.worker_id}"
    dataloader = BackgroundGenerator(
        batch_gen, num_workers=args.num_threads, queue_size=args.queue_size
    )
    with spdl.utils.tracing(trace_path, enable=args.trace is not None):
        return _iter_dataloader(dataloader, ev)


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


def _init(debug, worker_id):
    _init_logging(debug, worker_id)

    spdl.utils.set_ffmpeg_log_level(16)
    spdl.utils.init_folly(
        [
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


def _parse_process_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_known_args(args)


def _main(args=None):
    ns, args = _parse_process_args(args)

    args_set = [
        args + [f"--worker-id={i}", f"--num-workers={ns.num_workers}"]
        for i in range(ns.num_workers)
    ]

    from multiprocessing import Pool

    with Pool(processes=ns.num_workers) as pool:
        _init_logging()
        _LG.info("Spawned: %d workers", ns.num_workers)

        vals = pool.map(_benchmark, args_set)

    ave_time = sum(v.elapsed for v in vals) / len(vals)
    total_frames = sum(v.num_frames for v in vals)
    total_batches = sum(v.num_batches for v in vals)

    _LG.info(f"{ave_time=:.2f}, {total_frames=}, {total_batches=}")

    FPS = total_frames / ave_time
    BPS = total_batches / ave_time
    _LG.info(f"Aggregated {FPS=:.2f}, {BPS=:.2f}")


if __name__ == "__main__":
    _main()
