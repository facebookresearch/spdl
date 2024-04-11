"""Experiment for running asyncio.loop in background thread and decode media"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from queue import Queue

import spdl.io
import spdl.utils

import torch

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
    return parser.parse_args(args)


class BulkImageProcessor:
    def __init__(self, queue, decode_func):
        self.queue = queue
        self.decode_func = decode_func

    def _cb(self, task):
        if err := task.exception():
            _LG.error("Failed to process task %s: %s", task.get_name(), err)
        else:
            self.queue.put(task.result())

    async def __call__(self, path_gen, *, max_tasks=30):
        _tasks = set()
        for i, paths in enumerate(path_gen):
            coro = self.decode_func(paths)
            task = asyncio.create_task(coro, name=f"batch_{i}")
            task.add_done_callback(self._cb)
            task.add_done_callback(_tasks.discard)
            _tasks.add(task)

            while len(_tasks) > max_tasks:
                await asyncio.sleep(0)

        while len(_tasks):
            await asyncio.sleep(0)


def _get_decode_func(width, height, pix_fmt="rgba"):
    async def _func(srcs):
        return await spdl.io.async_batch_load_image(
            srcs,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            strict=False,
        )

    return _func


async def _track(queue):
    while True:
        await asyncio.sleep(3 / 1000)
        spdl.utils.trace_default_demux_executor_queue_size()
        spdl.utils.trace_default_decode_executor_queue_size()
        spdl.utils.trace_counter(0, queue.qsize())


async def _process_flist(
    queue,
    paths_gen,
    cuda_device_index,
    width=222,
    height=222,
):
    tracker = asyncio.create_task(_track(queue))

    bip = BulkImageProcessor(
        queue,
        _get_decode_func(width, height),
    )
    _LG.info("Starting decoding job.")
    task = asyncio.create_task(bip(paths_gen))
    await task
    if err := task.exception():
        _LG.error(f"Failed to process the flist: {err}")

    tracker.cancel()


def _iter_file(path, prefix):
    with open(path, "r") as f:
        for line in f:
            if path := line.strip():
                yield prefix + path


def _sample(gen, offset=0, every_n=1, max=None):
    offset = offset % every_n

    num = 0
    for i, item in enumerate(gen):
        if i % every_n == offset:
            yield item
            num += 1

            if max is not None and num >= max:
                return


def _batch(gen, batch_size):
    batch = []
    for item in gen:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_flist(path, prefix, batch_size, *, n=0, N=1, max=None):
    return _batch(_sample(_iter_file(path, prefix), n, N, max), batch_size)


def bg_worker(queue, flist_path, prefix, batch_size, worker_id, num_workers, trace):
    asyncio.set_event_loop(asyncio.new_event_loop())

    trace_path = f"{trace}.{worker_id}"
    paths_gen = _iter_flist(flist_path, prefix, batch_size, n=worker_id, N=num_workers)
    with spdl.utils.tracing(trace_path, enable=trace is not None):
        asyncio.run(_process_flist(queue, paths_gen, worker_id))

    queue.put(None)


def _benchmark(args):
    args = _parse_args(args)
    _init(
        args.debug,
        args.num_demux_threads,
        args.num_decode_threads,
        args.worker_id,
    )

    # Warm up
    torch.zeros([1, 1], device=torch.device(f"cuda:{args.worker_id}"))

    _LG.info(args)

    queue = Queue(maxsize=100)
    device_index = args.worker_id

    thread = threading.Thread(
        target=bg_worker,
        args=(
            queue,
            args.input_flist,
            args.prefix,
            args.batch_size,
            device_index,
            args.num_workers,
            args.trace,
        ),
    )
    thread.start()

    _LG.info("Waiting for the producers...")
    device = torch.device(f"cuda:{device_index}")
    t_int = t0 = time.monotonic()
    num_processed_int = num_processed = 0
    while (buffer := queue.get()) is not None:
        batch = spdl.io.to_torch(buffer).to(device)
        num_processed += batch.shape[0]

        t1 = time.monotonic()
        if (elapsed := t1 - t_int) > 10:
            n = num_processed - num_processed_int
            qps = n / elapsed
            _LG.info(
                f"Interval QPS={qps:.2f} (= {n} / {elapsed:.2f}), (Done {num_processed})"
            )

            t_int = t1
            num_processed_int = num_processed
    elapsed = time.monotonic() - t0
    qps = num_processed / elapsed
    _LG.info(f"QPS={qps:.2f} ({num_processed} / {elapsed:.2f}), (Done {num_processed})")

    thread.join()


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


def _init(debug, num_demux_threads, num_decode_threads, worker_id):
    _init_logging(debug, worker_id)

    spdl.utils.set_ffmpeg_log_level(16)
    spdl.utils.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demux_threads}",
            f"--spdl_decoder_executor_threads={num_decode_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


def _parse_process_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_known_args(args)


def _main(args=None):
    ns, args = _parse_process_args(args)

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
