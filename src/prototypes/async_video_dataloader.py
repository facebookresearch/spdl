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

_TIMEOUT = 1 / 1000


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


async def wait_and_check(tasks, num_tasks=0):
    while len(tasks) > num_tasks:
        done, tasks = await asyncio.wait(
            tasks, timeout=_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if err := task.exception():
                _LG.error("Failed to process task %s: %s", task.get_name(), err)
        await asyncio.sleep(0)
    return tasks


# def _to_tensor(buffer, cuda_device_index):
#     return spdl.to_torch(buffer).to(f"cuda:{cuda_device_index}")


def _to_tensor(buffer, cuda_device_index):
    import numba

    # return numba.cuda.to_device(spdl.to_numba(buffer), to=cuda_device_index)
    return spdl.to_numba(buffer)


class BulkVideoProcessor:
    def __init__(self, queue, *, cuda_device_index, use_nvdec=False):
        self.queue = queue
        self.cuda_device_index = cuda_device_index
        self.use_nvdec = use_nvdec

        self.width = 222
        self.height = 222
        self.pix_fmt = "rgba"

    async def __call__(self, path_gen, *, max_tasks=10):
        tasks = set()
        for path in path_gen:
            tasks = await wait_and_check(tasks, max_tasks)

            coro = self._demux(path)
            tasks.add(asyncio.create_task(coro, name=f"decode={path}"))

        await wait_and_check(tasks)

    async def _demux(self, path):
        timestamps = [(0, float("inf"))]

        tasks = set()
        demuxer = spdl.async_demux_video(path, timestamps=timestamps)
        i = -1
        async for packets in demuxer:
            i += 1
            coro = (
                self._decode_nvdec(packets, self.cuda_device_index)
                if self.use_nvdec
                else self._decode(packets)
            )
            task = asyncio.create_task(coro, name=f"decode={path}:{timestamps[i]}")
            tasks.add(task)

        await wait_and_check(tasks)

    async def _decode(self, packets):
        buffer = await spdl.async_convert(
            await spdl.async_decode(
                packets,
                width=self.width,
                height=self.height,
                pix_fmt=self.pix_fmt,
            )
        )
        tensor = _to_tensor(buffer, self.cuda_device_index)
        await self.queue.put(tensor)

    async def _decode_nvdec(self, packets, cuda_device_index):
        buffer = await spdl.async_convert(
            await spdl.async_decode_nvdec(
                await spdl.async_apply_bsf(packets),
                cuda_device_index=cuda_device_index,
                width=self.width,
                height=self.height,
                pix_fmt=self.pix_fmt,
            )
        )
        tensor = _to_tensor(buffer, self.cuda_device_index)
        await self.queue.put(tensor)


async def _track(queue):
    while True:
        await asyncio.sleep(3 / 1000)
        libspdl.trace_default_demux_executor_queue_size()
        libspdl.trace_default_decode_executor_queue_size()
        spdl.utils.trace_counter(0, queue.qsize())


async def _batch_decode_video(path_gen, *, cuda_device_index=None, use_nvdec=False):
    queue = asyncio.Queue(maxsize=100)
    bmp = BulkVideoProcessor(
        queue=queue, cuda_device_index=cuda_device_index, use_nvdec=use_nvdec
    )
    task = asyncio.create_task(bmp(path_gen))

    tracker = asyncio.create_task(_track(queue))

    while not (task.done() and queue.empty()):
        try:
            yield await asyncio.wait_for(queue.get(), timeout=_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        finally:
            await asyncio.sleep(0)

    tracker.cancel()


async def _process_flist(paths_gen, worker_id, use_nvdec):
    t_int = t0 = time.monotonic()
    num_frames_int = num_frames = num_tensors = 0
    async for batch in _batch_decode_video(
        paths_gen, cuda_device_index=worker_id, use_nvdec=use_nvdec
    ):
        num_frames += batch.shape[0]
        num_tensors += 1

        t1 = time.monotonic()
        if (elapsed := t1 - t_int) > 10:
            n = num_frames - num_frames_int
            qps = n / elapsed
            _LG.info(
                f"Interval QPS={qps:.2f} ({n}/{elapsed:.2f}), "
                f"(Done {num_frames} frames / {num_tensors} tensors)"
            )

            t_int = t1
            num_frames_int = num_frames

    elapsed = time.monotonic() - t0
    qps = num_frames / elapsed
    _LG.info(
        f"QPS={qps:.2f} ({num_frames}/{elapsed:.2f}), "
        f"(Done {num_frames} frames / {num_tensors} tensors)"
    )


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


def bg_worker(flist_path, prefix, worker_id, num_workers, use_nvdec, trace):
    asyncio.set_event_loop(asyncio.new_event_loop())

    # ------------------------------------------------------------------------------
    # Warm up 2
    import numba.cuda

    numba.cuda.select_device(worker_id)
    if use_nvdec:
        path_gen = _sample(_iter_file(flist_path, prefix), max=100)
        asyncio.run(_process_flist(path_gen, worker_id, use_nvdec))
    # ------------------------------------------------------------------------------

    path_gen = _sample(_iter_file(flist_path, prefix), worker_id, num_workers)
    trace_path = f"{trace}.{worker_id}{'.nvdec' if use_nvdec else ''}"
    with spdl.utils.tracing(trace_path, enable=trace is not None):
        asyncio.run(_process_flist(path_gen, worker_id, use_nvdec))


def _benchmark(args):
    args = _parse_args(args)
    _init(
        args.debug,
        args.num_demux_threads,
        args.num_decode_threads,
        args.worker_id,
    )

    # ------------------------------------------------------------------------------
    # Warm up 1
    # ------------------------------------------------------------------------------
    # import torch
    #
    # torch.zeros([1, 1], device=torch.device(f"cuda:{args.worker_id}"))
    # ------------------------------------------------------------------------------

    _LG.info(args)

    thread = threading.Thread(
        target=bg_worker,
        args=(
            args.input_flist,
            args.prefix,
            args.worker_id,
            args.num_workers,
            args.nvdec,
            args.trace,
        ),
    )
    thread.start()
    thread.join()


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
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
