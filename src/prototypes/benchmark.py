"""Decode multiple videos using GPU."""

import logging
import time

import spdl.utils
from spdl import libspdl


_LG = logging.getLogger(__name__)


def _parse_python_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", help="Input video file.", required=True)
    parser.add_argument(
        "-o", "--output-trace", help="Output trace file.", default="benchmark.pftrace"
    )
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--adoptor", default="BasicAdoptor")
    parser.add_argument("--prefix")
    parser.add_argument("--decoder")
    parser.add_argument("--num-jobs", type=int, default=5)
    parser.add_argument("--num-ts", type=int, default=5)
    parser.add_argument("--decoder-threads", type=int, default=8)
    parser.add_argument("--demuxer-threads", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
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

    timestamps = [(60 + i, 60.5 + i) for i in range(num_ts)]
    futures = []

    if gpu == -1:
        func = libspdl.decode_video
        cfg = {
            "src": input_video,
            "timestamps": timestamps,
            "adoptor": adoptor,
        }
    elif decoder is None:
        func = libspdl.decode_video_nvdec
        cfg = {
            "src": input_video,
            "timestamps": timestamps,
            "cuda_device_index": gpu,
            "adoptor": adoptor,
        }
    else:
        func = libspdl.decode_video
        cfg = {
            "src": input_video,
            "timestamps": timestamps,
            "decoder": decoder,
            "decoder_options": {"gpu": f"{gpu}"},
            "cuda_device_index": gpu,
            "adoptor": adoptor,
        }
    _LG.info("Config: %s", cfg)

    t0 = time.monotonic()
    _LG.info("Launching decoding jobs")
    for _ in range(num_jobs):
        futures.append(func(**cfg))

    _LG.info("Checking decoding results")
    frames = []
    for future in futures:
        try:
            frames.extend(future.get())
        except Exception as e:
            _LG.exception(e)
            continue
    t1 = time.monotonic()
    buffers = []
    for frame in frames:
        buffers.append(libspdl.convert_frames(frame, None))
    t2 = time.monotonic()
    elapsed = t2 - t0
    num_frames = sum(b.shape[0] for b in buffers)
    print(buffers[0].shape)
    _LG.info(
        f"Decoded {num_frames} frames. Elapsed {elapsed} [sec]. QPS: {num_frames/elapsed}"
    )


def _main():
    args = _parse_python_args()
    _init(args.debug, args.demuxer_threads, args.decoder_threads)

    with spdl.utils.tracing(args.output_trace, enable=args.trace):
        if args.gpu != -1 and args.decoder is not None:
            libspdl.create_cuda_context(args.gpu, use_primary_context=True)

        _test(
            args.input_video,
            args.decoder,
            args.gpu,
            args.adoptor,
            args.prefix,
            args.num_jobs,
            args.num_ts,
        )


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
