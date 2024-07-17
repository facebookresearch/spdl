#!/usr/bin/env python3
"""Benchmark loading audio dataset"""

# pyre-ignore-all-errors

import logging
import time
from pathlib import Path

import spdl.io
import spdl.utils
import torch
from spdl.dataloader import PipelineBuilder
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
    parser.add_argument("--max-samples", type=int, default=float("inf"))
    parser.add_argument("--prefix")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=8)
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
    def src():
        with open(get_flist(args.split)) as f:
            i = 0
            for line in f:
                if line := line.strip():
                    yield args.prefix + line
                    if (i := i + 1) >= args.max_samples:
                        return

    async def _async_decode_func(src):
        src = src[0].split("\t")[0]
        buffer = await spdl.io.async_load_audio(
            src,
            filter_desc=spdl.io.get_audio_filter_desc(
                sample_rate=16000,
                num_channels=1,
            ),
            cuda_config=spdl.io.cuda_config(
                device_index=0,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            ),
        )
        return spdl.io.to_torch(buffer)

    apl = (
        PipelineBuilder()
        .add_source(src())
        .aggregate(args.batch_size)
        .pipe(_async_decode_func, concurrency=args.num_threads)
        .add_sink(args.queue_size)
        .build(num_threads=args.num_threads)
    )

    return apl


def _main(args=None):
    args = _parse_args(args)
    _init(args.debug)

    _LG.info(args)

    pipeline = _get_batch_generator(args)

    device = torch.device(f"cuda:0")

    # Warm up
    torch.zeros([1, 1], device=device)

    trace_path = f"{args.trace}"
    with (
        pipeline.auto_stop(),
        spdl.utils.tracing(trace_path, buffer_size=8192, enable=args.trace is not None),
    ):
        return _iter_dataloader(pipeline.get_iterator())


def _init_logging(debug=False):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


def _init(debug):
    _init_logging(debug)
    spdl.utils.set_ffmpeg_log_level(16)


if __name__ == "__main__":
    _main()
