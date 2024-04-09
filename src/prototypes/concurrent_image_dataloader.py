#!/usr/bin/env python3
"""Test/Benchmark decoding with ImageNet images"""

import logging
import time
from functools import partial
from queue import Queue
from threading import Thread

import spdl.io

import spdl.utils

import torch

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
    parser.add_argument("--nvdec", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num-demuxing-threads", type=int, default=4)
    parser.add_argument("--num-decoding-threads", type=int, default=8)
    return parser.parse_args()


def _batch_decode(srcs, use_nvdec, gpu, **kwargs):
    decode_func = (
        partial(spdl.io.decode_packets_nvdec, cuda_device_index=gpu, **kwargs)
        if use_nvdec
        else partial(spdl.io.decode_packets, **kwargs)
    )

    futures = []
    for src in srcs:
        future = spdl.io.chain_futures(
            spdl.io.demux_media("image", src),
            decode_func,
        )
        futures.append(future)

    return spdl.io.chain_futures(
        spdl.io.wait_futures(futures),
        spdl.io.convert_frames,
    )


def process_flist(flist, queue, nvdec: bool, gpu: int, **kwargs):
    for paths in flist:
        future = _batch_decode(paths, nvdec, gpu, **kwargs)
        queue.put(future)
    queue.put(None)


class BackgroundDecoder:
    def __init__(self, flist, queue_size, nvdec: bool, gpu: int):
        self.flist = flist
        self.queue_size = queue_size
        self.nvdec = nvdec
        self.gpu = gpu

        self.queue = None
        self.thread = None

    def __iter__(self):
        self.queue = Queue(maxsize=self.queue_size)
        self.thread = Thread(
            target=process_flist,
            args=(
                self.flist,
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


def _test(srcs, queue_size, gpu: int, use_nvdec: bool):
    bgd = BackgroundDecoder(srcs, queue_size, use_nvdec, gpu)

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


def _iter_flist(flist, prefix, batch_size):
    paths = []
    with open(flist, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(prefix + line)
            if len(paths) >= batch_size:
                yield paths
                paths = []
    if paths:
        yield paths


def _main():
    args = _parse_args()
    _init(args.debug, args.num_demuxing_threads, args.num_decoding_threads)

    _test(
        _iter_flist(args.input_flist, args.prefix, args.batch_size),
        args.queue_size,
        args.gpu,
        args.nvdec,
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
