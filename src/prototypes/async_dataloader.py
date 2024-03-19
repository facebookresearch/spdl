"""Experiment for running asyncio.loop in background thread and decode media"""

import asyncio
import logging
import threading
import time
from pathlib import Path

import spdl
import spdl.utils
from spdl import libspdl

_LG = logging.getLogger(__name__)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--num-demux-threads", type=int, required=True)
    parser.add_argument("--num-decode-threads", type=int, required=True)
    return parser.parse_args()


def _iter_flist(flist):
    with open(flist, "r") as f:
        for line in f:
            path = line.strip()
            if path:
                yield path


# def _iter_batch(generator, batch_size=32):
#     batch = []
#     for sample in generator:
#         batch.append(sample)
#         if len(batch) >= batch_size:
#             yield batch
#             batch = []
#     if batch:
#         yield batch


async def _batch_decode_image(paths, adoptor, batch_size):
    demuxing_done = False
    demuxing = set()
    decoding = set()
    conversion = set()
    buffers = []
    while True:
        if not demuxing_done:
            while len(demuxing) < batch_size * 10:
                try:
                    path = next(paths)
                except StopIteration:
                    demuxing_done = True
                    break
                else:
                    coro = spdl.async_demux_image(
                        path,
                        adoptor=adoptor,
                    )
                    demuxing.add(asyncio.create_task(coro))

        if demuxing:
            done, demuxing = await asyncio.wait(
                demuxing, return_when=asyncio.FIRST_COMPLETED
            )
            for result in done:
                if err := result.exception():
                    _LG.error("Failed to demux: %s: %s", type(err).__name__, err)
                    continue
                coro = spdl.async_decode(
                    result.result(), filter_desc="format=pix_fmts=rgb24"
                )
                decoding.add(asyncio.create_task(coro))

        if decoding:
            done, decoding = await asyncio.wait(
                decoding, return_when=asyncio.FIRST_COMPLETED
            )
            for result in done:
                if err := result.exception():
                    _LG.error("Failed to decode: %s: %s", type(err).__name__, err)
                    continue
                conversion.add(asyncio.create_task(spdl.async_convert(result.result())))

        if conversion:
            done, conversion = await asyncio.wait(
                conversion, return_when=asyncio.FIRST_COMPLETED
            )
            for result in done:
                if err := result.exception():
                    _LG.error("Failed to convert: %s: %s", type(err).__name__, err)
                    continue
                buffers.append(result.result())

        while len(buffers) > batch_size:
            yield buffers[:batch_size]
            buffers = buffers[batch_size:]

        if demuxing_done and not demuxing and not decoding and not conversion:
            break

    if buffers:
        yield buffers


async def _process_flist(flist, adoptor, batch_size):
    num_processed = 0
    t0 = time.monotonic()
    t1 = t0
    async for buffers in _batch_decode_image(flist, adoptor, batch_size):
        num_processed += len(buffers)

        t2 = time.monotonic()
        if t2 - t1 > 30:
            elapsed = t2 - t0
            t1 = t2
            _LG.info(f"QPS={num_processed / elapsed} ({num_processed} / {elapsed:.2f})")

    elapsed = time.monotonic() - t0
    _LG.info(f"QPS={num_processed / elapsed} ({num_processed} / {elapsed:.2f})")
    return num_processed


def bg_worker(flist, adoptor, batch_size, loop=None):
    asyncio.set_event_loop(loop or asyncio.new_event_loop())
    asyncio.run(_process_flist(flist, adoptor, batch_size))


def _main():
    args = _parse_args()
    _init(args.debug, args.num_demux_threads, args.num_decode_threads)

    adoptor = spdl.libspdl.BasicAdoptor(args.prefix)

    flist = _iter_flist(args.input_flist)
    with spdl.utils.tracing("async_dataloader.py.pftrace", enable=args.trace):
        thread = threading.Thread(
            target=bg_worker, args=(flist, adoptor, args.batch_size)
        )
        thread.start()
        thread.join()


def _init(debug, num_demux_threads, num_decode_threads):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)

    libspdl.set_ffmpeg_log_level(16)
    libspdl.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demux_threads}",
            f"--spdl_decoder_executor_threads={num_decode_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


if __name__ == "__main__":
    _main()
