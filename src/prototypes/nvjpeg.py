import logging
import time
from pathlib import Path

import spdl.io
from spdl.dataloader import apply_async, BackgroundGenerator
from spdl.dataloader._utils import _iter_flist

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
    parser.add_argument("--num-demux-threads", type=int, default=2)
    parser.add_argument("--num-decode-threads", type=int, default=4)
    return parser.parse_args(args)


def _get_test_func(args, use_nvjpeg):
    srcs_gen = _iter_flist(
        args.input_flist,
        prefix=args.prefix,
        batch_size=1,
        max=args.max_samples,
        drop_last=True,
    )

    if use_nvjpeg:

        async def _decode(src):
            src = src[0]
            with open(src, "rb") as f:
                data = f.read()
            return await spdl.io.async_decode_image_nvjpeg(
                data, cuda_device_index=args.device
            )

    else:

        async def _decode(src):
            src = src[0]
            return await spdl.io.async_load_media(
                "image", src, convert_options={"cuda_device_index": args.device}
            )

    return apply_async(_decode, srcs_gen)


def _run(dataloader):
    num_frames = 0
    t0 = time.monotonic()
    try:
        for _ in dataloader:
            num_frames += 1
    finally:
        elapsed = time.monotonic() - t0
        fps = num_frames / elapsed
        _LG.info(f"FPS={fps:.2f} ({num_frames}/{elapsed:.2f})")


def _main():
    args = _parse_args()
    _init(args.debug, args.num_demux_threads, args.num_decode_threads)

    for use_nvjpeg in [True, True, False]:  # Run nvjpeg twice to warmup
        _LG.info("Testing %s.", "nvjpeg" if use_nvjpeg else "ffmpeg")
        batch_gen = _get_test_func(args, use_nvjpeg)

        dataloader = BackgroundGenerator(batch_gen)
        trace_path = f"{args.trace}_{'nvjpeg' if use_nvjpeg else 'ffmpeg'}.pftrace"
        with (spdl.utils.tracing(trace_path, enable=args.trace is not None),):
            _run(dataloader)


def _init(debug, num_demux_threads, num_decode_threads, worker_id=None):
    _init_logging(debug, worker_id)

    spdl.utils.set_ffmpeg_log_level(16)
    spdl.utils.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demux_threads}",
            f"--spdl_decoder_executor_threads={num_decode_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


if __name__ == "__main__":
    _main()
