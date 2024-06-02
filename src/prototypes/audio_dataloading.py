#!/usr/bin/env python3
"""Benchmark loading audio dataset"""

import logging
import time
from pathlib import Path

import spdl.io
import spdl.utils
import torch
from spdl.dataloader import apply_async, BackgroundGenerator
from spdl.dataloader._utils import _iter_flist
from spdl.dataset.librispeech import get_flist

_LG = logging.getLogger(__name__)


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--split", default="test-other", choices=["test-clean", "test-other"]
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prefix")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args(args)
    if args.trace:
        args.max_samples = args.batch_size * 20
    return args


def _iter_dataloader(dataloader):
    t0 = t_int = time.monotonic()
    num_frames = num_frames_int = 0
    num_batches = 0
    try:
        for batch in dataloader:
            num_frames += batch.shape[0]
            num_batches += 1

            t1 = time.monotonic()
            if (elapsed := t1 - t_int) > 10:
                n = num_frames - num_frames_int
                _LG.info(f"Interval FPS={n / elapsed:.2f} (Done {num_batches=})")

                t_int = t1
                num_frames_int = num_frames
    finally:
        elapsed = time.monotonic() - t0
        fps = num_frames / elapsed
        _LG.info(
            f"FPS={fps:.2f} ({num_frames} / {elapsed:.2f}), (Done {num_frames=}, {num_batches=})"
        )


def _get_batch_generator(args):
    srcs_gen = _iter_flist(
        get_flist(args.split),
        prefix=args.prefix,
        batch_size=1,  # args.batch_size,
        n=args.worker_id,
        N=args.num_workers,
        max=args.max_samples,
    )

    async def _async_decode_func(src):
        src = src[0].split("\t")[0]
        buffer = await spdl.io.async_load_audio(
            src,
            filter_desc=spdl.io.get_audio_filter_desc(
                sample_rate=16000,
                num_channels=1,
            ),
            cuda_config=spdl.io.cuda_config(
                device_index=args.worker_id,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            ),
        )
        return spdl.io.to_torch(buffer)

    return apply_async(_async_decode_func, srcs_gen)


def _main(args=None):
    args = _parse_args(args)
    _init(args.debug, args.worker_id)

    _LG.info(args)

    batch_gen = _get_batch_generator(args)

    device = torch.device(f"cuda:{args.worker_id}")

    # Warm up
    torch.zeros([1, 1], device=device)

    trace_path = f"{args.trace}.{args.worker_id}"
    dataloader = BackgroundGenerator(
        batch_gen, num_workers=args.num_threads, queue_size=args.queue_size
    )
    with spdl.utils.tracing(
        trace_path, buffer_size=8192, enable=args.trace is not None
    ):
        return _iter_dataloader(dataloader)


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
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


if __name__ == "__main__":
    _main()
