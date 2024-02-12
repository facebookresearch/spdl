"""Decode multiple videos using GPU."""
import logging
import time

import spdl.utils
from spdl import libspdl


_LG = logging.getLogger(__name__)


def _parse_python_args():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", help="Input video file.", required=True)
    parser.add_argument("-o", "--output-dir", default=Path("tmp"), type=Path)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--adoptor", default="BasicAdoptor")
    parser.add_argument("--prefix")
    parser.add_argument("--decoder")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--num-jobs", type=int, default=5)
    parser.add_argument("--num-ts", type=int, default=5)
    parser.add_argument("--decoder-threads", type=int, default=8)
    parser.add_argument("--demuxer-threads", type=int, default=4)
    return parser.parse_args()


def _test(
        input_video: str,
        decoder: str,
        gpu: int,
        adoptor: str,
        prefix: str,
        num_jobs: int,
        num_ts: int,
):
    adoptor = getattr(libspdl, adoptor)(prefix)

    timestamps = [(10 * i, 10 * i + 3) for i in range(num_ts)]
    futures = []

    cfg = {
        "src": input_video,
        "timestamps": timestamps,
        "decoder": decoder,
        "decoder_options": None if decoder is None or gpu == -1 else {"gpu": str(gpu)},
        "cuda_device_index": gpu,
        "adoptor": adoptor,
        "pix_fmt": "rgb24",
        "width": 640,
        "height": 480,
    }
    _LG.info("Config: %s", cfg)

    t0 = time.monotonic()
    _LG.info("Launching decoding jobs")
    for _ in range(num_jobs):
        futures.append(libspdl.decode_video(**cfg))

    _LG.info("Checking decoding results")
    results = []
    for future in futures:
        try:
            frames = future.get()
        except Exception as e:
            _LG.exception(e)
            continue
        for f in frames:
            buffer = libspdl.to_numpy(f)
            results.append(buffer)
    elapsed = time.monotonic() - t0
    _LG.info(f"{elapsed} [sec]")
    return results


def _plot(frames, output_dir):
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        img = Image.fromarray(f)
        img.save(str(output_dir / f"{i}.png"))


def _main():
    args = _parse_python_args()
    _init(args.debug, args.demuxer_threads, args.decoder_threads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with spdl.utils.tracing(str(args.output_dir / "benchmark.pftrace")):
        frames = _test(
            args.input_video,
            args.decoder,
            args.gpu,
            args.adoptor,
            args.prefix,
            args.num_jobs,
            args.num_ts)
        if args.plot:
            _plot(frames[0], args.output_dir / "frames")


def _init(debug, demuxer_threads, decoder_threads):
    logging.basicConfig(level=logging.INFO)
    if debug:
        logging.getLogger("spdl").setLevel(logging.DEBUG)

    folly_args = [
        f"--spdl_demuxer_executor_threads={demuxer_threads}",
        f"--spdl_decoder_executor_threads={decoder_threads}",
        f"--logging={'DBG' if debug else 'INFO'}",
    ]
    libspdl.init_folly(folly_args)

    if debug:
        libspdl.set_ffmpeg_log_level(40)


if __name__ == "__main__":
    _main()
