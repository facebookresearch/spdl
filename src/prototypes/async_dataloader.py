"""Experiment for running asyncio.loop in background thread and decode media"""

import asyncio
import functools
import logging
import threading
import time
from pathlib import Path

import spdl
import spdl.utils
from spdl import libspdl

_LG = logging.getLogger(__name__)


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--num-demux-threads", type=int, required=True)
    parser.add_argument("--num-decode-threads", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--nvdec", action="store_true")
    return parser.parse_args(args)


def _iter_flist(flist, id, N):
    with open(flist, "r") as f:
        for i, line in enumerate(f):
            path = line.strip()
            if path and i % N == id:
                yield path


_TIMEOUT = 1 / 1000.0

_QPS_INDEX = 0
_PROCESSED_INDEX = 1
_CONVERT_INDEX = 2
_FRAME_INDEX = 3
_DECODE_INDEX = 4
_DEMUX_INDEX = 5


async def _batch_decode_image(
    paths,
    adoptor,
    batch_size,
    *,
    width=222,
    height=222,
    cuda_device_index=None,
):
    demuxing_done = False
    demuxing = set()
    decoding = set()
    frames = []
    conversion = set()

    filter_desc = f"scale=width={width}:height={height},format=pix_fmts=rgb24"
    async_decode_func = (
        functools.partial(spdl.async_decode, filter_desc=filter_desc)
        if cuda_device_index is None
        else functools.partial(
            spdl.async_decode_nvdec,
            cuda_device_index=cuda_device_index,
            width=width,
            height=height,
            pix_fmt="rgba",
        )
    )
    while True:
        # Handle the process from downstream to upstream for better performance.

        # 5. Check the state of buffer conversion
        if conversion:
            with spdl.utils.trace_event("conversion - await"):
                done, conversion = await asyncio.wait(
                    conversion, timeout=_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
                )
                for result in done:
                    if err := result.exception():
                        _LG.error("Failed to convert: %s: %s", type(err).__name__, err)
                        continue
                    yield result.result()
            spdl.utils.trace_counter(_CONVERT_INDEX, len(conversion))

        # 4. Queue buffer conversion
        if frames:
            while len(frames) > batch_size:
                with spdl.utils.trace_event("conversion - push"):
                    coro = spdl.async_convert(frames[:batch_size])
                    conversion.add(asyncio.create_task(coro))
                    frames = frames[batch_size:]
            spdl.utils.trace_counter(_FRAME_INDEX, len(frames))
            spdl.utils.trace_counter(_CONVERT_INDEX, len(conversion))

        # 3. Check the state of decoding
        if decoding:
            with spdl.utils.trace_event("decoding - await"):
                done, decoding = await asyncio.wait(
                    decoding, timeout=_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
                )
                for result in done:
                    if err := result.exception():
                        _LG.error("Failed to decode: %s: %s", type(err).__name__, err)
                        continue
                    frames.append(result.result())
            spdl.utils.trace_counter(_DECODE_INDEX, len(decoding))
            spdl.utils.trace_counter(_FRAME_INDEX, len(frames))

        # 2. Check the state of demuxing and queue decoding
        if demuxing:
            with spdl.utils.trace_event("demuxing - await"):
                done, demuxing = await asyncio.wait(
                    demuxing, timeout=_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
                )
                for result in done:
                    if err := result.exception():
                        _LG.error("Failed to demux: %s: %s", type(err).__name__, err)
                        continue
                    coro = async_decode_func(result.result())
                    decoding.add(asyncio.create_task(coro))
            spdl.utils.trace_counter(_DEMUX_INDEX, len(demuxing))
            spdl.utils.trace_counter(_DECODE_INDEX, len(decoding))

        # 2. Queue demuxing
        if not demuxing_done and len(demuxing) < batch_size * 3:
            with spdl.utils.trace_event("demuxing - push"):
                num_push = batch_size if not demuxing else 3
                for _ in range(num_push):
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
            spdl.utils.trace_counter(_DEMUX_INDEX, len(demuxing))

        if demuxing_done and not demuxing and not decoding and not conversion:
            break

    if frames:
        yield await spdl.async_convert(frames)


async def _process_flist(flist, adoptor, batch_size, cuda_device_index):
    num_processed = 0
    t0 = time.monotonic()

    t_int = t0
    num_processed_int = 0
    async for buffers in _batch_decode_image(
        flist, adoptor, batch_size, cuda_device_index=cuda_device_index
    ):
        num_processed += buffers.shape[0]
        spdl.utils.trace_counter(_PROCESSED_INDEX, num_processed)

        t2 = time.monotonic()
        if t2 - t_int > 10:
            elapsed = t2 - t_int
            n = num_processed - num_processed_int
            qps = n / elapsed
            _LG.info(
                f"Interval QPS={qps:.2f} (= {n} / {elapsed:.2f}), (Done {num_processed})"
            )

            t_int = t2
            num_processed_int = num_processed

    elapsed = time.monotonic() - t0
    _LG.info(f"QPS={num_processed / elapsed} ({num_processed} / {elapsed:.2f})")
    return num_processed


def bg_worker(flist, adoptor, batch_size, worker_id, num_workers, use_nvdec, loop=None):
    asyncio.set_event_loop(loop or asyncio.new_event_loop())
    cuda_device_index = worker_id if use_nvdec else None
    asyncio.run(_process_flist(flist, adoptor, batch_size, cuda_device_index))


def _benchmark(args):
    args = _parse_args(args)
    _init(
        args.debug,
        args.num_demux_threads,
        args.num_decode_threads,
        args.worker_id,
    )

    adoptor = spdl.libspdl.BasicAdoptor(args.prefix)

    flist = _iter_flist(args.input_flist, args.worker_id, args.num_workers)
    trace_path = f"{args.trace}.{args.worker_id}"
    with spdl.utils.tracing(trace_path, enable=args.trace is not None):
        thread = threading.Thread(
            target=bg_worker,
            args=(
                flist,
                adoptor,
                args.batch_size,
                args.worker_id,
                args.num_workers,
                args.nvdec,
            ),
        )
        thread.start()
        thread.join()


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


def _init(debug, num_demux_threads, num_decode_threads, worker_id):
    _init_logging(debug, worker_id)

    libspdl.set_ffmpeg_log_level(16)
    libspdl.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demux_threads}",
            f"--spdl_decoder_executor_threads={num_decode_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


def _parse_process_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_known_args()


def _main():
    ns, args = _parse_process_args()

    args_set = [
        args + [f"--worker-id={i}", f"--num-workers={ns.num_workers}"]
        for i in range(ns.num_workers)
    ]

    from multiprocessing import Pool

    with Pool(processes=ns.num_workers) as pool:
        _init_logging()
        _LG.info("Spawned: %d workers", ns.num_workers)

        t0 = time.monotonic()
        pool.map(_benchmark, args_set)
        elapsed = time.monotonic() - t0
    _LG.info("Elapsed: %.2f", elapsed)


if __name__ == "__main__":
    _main()
