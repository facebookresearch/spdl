# pyre-ignore-all-errors

import logging
import time
from pathlib import Path

import spdl.io
import spdl.utils
import torch
from spdl.dataloader import AsyncPipeline, BackgroundGenerator
from spdl.utils import iter_flist

_LG = logging.getLogger(__name__)


def _parse_args(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args(args)


def _get_test_func(args, use_nvjpeg, width=224, height=224):
    srcs_gen = iter_flist(
        args.input_flist,
        prefix=args.prefix,
        batch_size=32,
        max=args.max_samples,
        drop_last=True,
    )
    cuda_config = spdl.io.cuda_config(
        device_index=args.device,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
    )

    if use_nvjpeg:

        async def _decode(srcs):
            buffer = await spdl.io.async_load_image_batch_nvjpeg(
                srcs, cuda_config=cuda_config, width=width, height=height
            )
            return spdl.io.to_torch(buffer)

    else:

        async def _decode(srcs):
            buffer = await spdl.io.async_load_image_batch(
                srcs,
                cuda_config=cuda_config,
                width=width,
                height=height,
            )
            return spdl.io.to_torch(buffer)

    return (
        AsyncPipeline().add_source(srcs_gen).pipe(_decode, concurrency=args.num_threads)
    )


def _run(dataloader):
    num_frames = 0
    t0 = time.monotonic()
    try:
        for batch in dataloader:
            num_frames += batch.shape[0]
    finally:
        elapsed = time.monotonic() - t0
        fps = num_frames / elapsed
        _LG.info(f"FPS={fps:.2f} ({num_frames}/{elapsed:.2f})")


def _main():
    args = _parse_args()
    _init(args.debug)

    for use_nvjpeg in [True, True, False]:  # Run nvjpeg twice to warmup
        _LG.info("Testing %s.", "nvjpeg" if use_nvjpeg else "ffmpeg")
        batch_gen = _get_test_func(args, use_nvjpeg)

        dataloader = BackgroundGenerator(batch_gen, num_workers=args.num_threads)
        trace_path = f"{args.trace}_{'nvjpeg' if use_nvjpeg else 'ffmpeg'}.pftrace"
        with (spdl.utils.tracing(trace_path, enable=args.trace is not None),):
            _run(dataloader)


def _init(debug, worker_id=None):
    _init_logging(debug, worker_id)

    spdl.utils.set_ffmpeg_log_level(16)


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


if __name__ == "__main__":
    _main()
