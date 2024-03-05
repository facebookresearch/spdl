#!/usr/bin/env python3
"""Test/Benchmark decoding with ImageNet images"""

import logging
import time
from functools import partial
from queue import Queue
from threading import Thread
from typing import Optional

import spdl
from spdl import libspdl

_LG = logging.getLogger(__name__)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-flist", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--prefix")
    parser.add_argument("--adoptor", default="BasicAdoptor")
    parser.add_argument("--nvdec", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--num-demuxing-threads", type=int)
    parser.add_argument("--num-decoding-threads", type=int)
    args = parser.parse_args()
    if args.nvdec and args.gpu is None:
        raise RuntimeError("Must specify --gpu when using NVDEC")
    return args


def _iter_flist(flist, batch_size):
    paths = []
    with open(flist, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(line)
            if len(paths) >= batch_size:
                yield paths
                paths = []
    if paths:
        yield paths


def _batch_decode(flist, adoptor, batch_size, queue, nvdec: bool, gpu: int, **kwargs):
    decode_fun = (
        partial(libspdl.batch_decode_image_nvdec, cuda_device_index=gpu)
        if nvdec
        else libspdl.batch_decode_image
    )

    for paths in _iter_flist(flist, batch_size):
        future = decode_fun(paths, adoptor=adoptor, **kwargs)
        queue.put(future)
    queue.put(None)


class BackgroundDecoder:
    def __init__(self, flist, adoptor, batch_size, queue_size, nvdec: bool, gpu: int):
        self.flist = flist
        self.adoptor = adoptor
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.nvdec = nvdec
        self.gpu = gpu

        self.queue = None
        self.thread = None

    def __iter__(self):
        self.queue = Queue(maxsize=self.queue_size)
        self.thread = Thread(
            target=_batch_decode,
            args=(
                self.flist,
                self.adoptor,
                self.batch_size,
                self.queue,
                self.nvdec,
                self.gpu,
            ),
            kwargs={"pix_fmt": "rgba", "width": 222, "height": 222},
        )
        self.thread.start()

        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item

        self.thread.join()


def _test_nvdec(input_flist, adoptor_type, prefix, batch_size, queue_size, gpu: int):
    adoptor = getattr(libspdl, adoptor_type)(prefix)

    bgd = BackgroundDecoder(input_flist, adoptor, batch_size, queue_size, True, gpu)

    num_decoded = 0
    t0 = time.monotonic()
    for i, future in enumerate(bgd):
        frames = future.get(strict=False)
        tensor = spdl.to_torch(frames)
        num_decoded += tensor.shape[0]

        if i % 1000 == 0:
            elapsed = time.monotonic() - t0
            print(f"Decode {num_decoded} frames. QPS: {num_decoded / elapsed}")
    elapsed = time.monotonic() - t0
    print(
        f"{elapsed} seconds to decode {num_decoded} frames. QPS: {num_decoded / elapsed}"
    )


def _test_cpu(
    input_flist, adoptor_type, prefix, batch_size, queue_size, gpu: Optional[int]
):
    adoptor = getattr(libspdl, adoptor_type)(prefix)

    bgd = BackgroundDecoder(input_flist, adoptor, batch_size, queue_size, False, gpu)

    device = "cpu" if gpu is None else f"cuda:{gpu}"

    num_decoded = 0
    t0 = time.monotonic()
    for i, future in enumerate(bgd):
        frames = future.get(strict=False)
        tensor = spdl.to_torch(frames).to(device)
        num_decoded += tensor.shape[0]

        if i % 1000 == 0:
            elapsed = time.monotonic() - t0
            print(f"Decode {num_decoded} frames. QPS: {num_decoded / elapsed}")
    elapsed = time.monotonic() - t0
    print(
        f"{elapsed} seconds to decode {num_decoded} frames. QPS: {num_decoded / elapsed}"
    )


def _main():
    args = _parse_args()
    _init(args.debug, args.num_demuxing_threads, args.num_decoding_threads)

    if args.nvdec:
        _test_nvdec(
            args.input_flist,
            args.adoptor,
            args.prefix,
            args.batch_size,
            args.queue_size,
            args.gpu,
        )
    else:
        _test_cpu(
            args.input_flist,
            args.adoptor,
            args.prefix,
            args.batch_size,
            args.queue_size,
            args.gpu,
        )


def _init(debug, num_demuxing_threads, num_decoding_threads):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)

    libspdl.set_ffmpeg_log_level(16)
    libspdl.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demuxing_threads}",
            f"--spdl_decoder_executor_threads={num_decoding_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


if __name__ == "__main__":
    _main()
