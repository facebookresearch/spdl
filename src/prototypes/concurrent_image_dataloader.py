#!/usr/bin/env python3
"""Test/Benchmark decoding with ImageNet images"""

import logging
import time
from queue import Queue
from threading import Thread

import spdl.io
import spdl.utils

import torch
from spdl.dataloader._utils import _iter_flist

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
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num-demux-threads", type=int, default=4)
    parser.add_argument("--num-decode-threads", type=int, default=8)
    return parser.parse_args()


def process_flist(paths_gen, queue, decode_options):
    for paths in paths_gen:
        future = spdl.io.batch_load_image(paths, **decode_options)
        queue.put(future)
    queue.put(None)


class BackgroundDecoder:
    def __init__(self, paths_gen, queue_size):
        self.paths_gen = paths_gen
        self.queue_size = queue_size

        self.queue = None
        self.thread = None

    def __iter__(self):
        self.queue = Queue(maxsize=self.queue_size)
        self.thread = Thread(
            target=process_flist,
            args=(
                self.paths_gen,
                self.queue,
                {"width": 222, "height": 222, "pix_fmt": "rgba"},
            ),
        )
        self.thread.start()

        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item

        self.thread.join()


def _test(paths_gen, queue_size: int, gpu: int):
    bgd = BackgroundDecoder(paths_gen, queue_size)

    device = torch.device(f"cuda:{gpu}")

    num_decoded = 0
    t0 = time.monotonic()
    t1 = t0
    for future in bgd:
        frames = future.result()
        tensor = spdl.io.to_torch(frames).to(device)
        num_decoded += tensor.shape[0]

        t2 = time.monotonic()
        if t2 - t1 > 10:
            elapsed = t2 - t0
            t1 = t2
            _LG.info(f"QPS={num_decoded / elapsed} ({num_decoded} / {elapsed:.2f})")
    elapsed = time.monotonic() - t0
    _LG.info(f"QPS={num_decoded / elapsed} ({num_decoded} / {elapsed:.2f})")


def _main():
    args = _parse_args()
    _init(args.debug, args.num_demux_threads, args.num_decode_threads)

    _test(
        _iter_flist(args.input_flist, prefix=args.prefix, batch_size=args.batch_size),
        args.queue_size,
        args.gpu,
    )


def _init(debug, num_demuxing_threads, num_decoding_threads):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)

    spdl.utils.set_ffmpeg_log_level(16)
    spdl.utils.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demuxing_threads}",
            f"--spdl_decoder_executor_threads={num_decoding_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


if __name__ == "__main__":
    _main()
