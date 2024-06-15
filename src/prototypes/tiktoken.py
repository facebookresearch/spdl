"""Bulk tokenize text files with OpenAI's TikToken"""

# pyre-ignore-all-errors

import asyncio
import concurrent.futures
import time
from pathlib import Path

import numpy as np

import spdl.io
import spdl.utils
import tiktoken
import torch
from spdl.io import CUDAConfig
from spdl.utils import iter_flist, run_async


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--input-flist", type=Path, required=True, help="File lilst")
    parser.add_argument("--prefix", help="Root directory of the dataset")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--max", type=int)
    return parser.parse_args(args)


def _read(path):
    with open(path, "r") as fp:
        return fp.read()


def _tokenize(path, encoding, cuda_config: CUDAConfig | None = None):
    with spdl.utils.trace_event("read"):
        data = _read(path)
    with spdl.utils.trace_event("encode"):
        tokens = encoding.encode(data)
    with spdl.utils.trace_event("np.array"):
        arr = np.array(tokens)
    if cuda_config is not None:
        buffer = spdl.io.transfer_buffer(
            arr,
            cuda_config=cuda_config,
        )
        arr = spdl.io.to_torch(buffer)
    return arr


async def _bulk_tokenize(
    file_gen, encoding, queue, cuda_device: int = 0, concurrency=32
):
    cuda_config = spdl.io.cuda_config(
        device_index=cuda_device,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
    )

    semaphore = asyncio.BoundedSemaphore(concurrency)

    async def _func(path):
        async with semaphore:
            buffer = await run_async(_tokenize, path, encoding, cuda_config=cuda_config)
            await queue.put(buffer)

    tasks = set()
    for path in file_gen:
        async with semaphore:
            task = asyncio.create_task(_func(path))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    while tasks:
        await asyncio.sleep(0.1)


async def _run(file_gen, encoding, concurrency=32):
    sentinel = object()
    queue = asyncio.Queue()

    async def _task():
        try:
            await _bulk_tokenize(file_gen, encoding, queue)
        finally:
            await queue.put(sentinel)

    task = asyncio.create_task(_task())
    num_tokens = 0
    t0 = time.monotonic()
    while (item := await queue.get()) is not sentinel:
        num_tokens += item.numel()
    elapsed = time.monotonic() - t0
    tps = num_tokens / elapsed
    print(f"{tps} [token/sec] ({num_tokens=}, {elapsed=:.3f})")

    await task


def _test(input_flist, prefix, encoding, num_threads, max):
    encoding = tiktoken.get_encoding(encoding)
    file_gen = iter_flist(input_flist, prefix=prefix, max=max)

    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    )
    loop.run_until_complete(_run(file_gen, encoding))


def _main(args=None):
    args = _parse_args(args)

    path = f"{args.trace}.pftrace"
    with spdl.utils.tracing(path, enable=args.trace):
        _test(args.input_flist, args.prefix, args.encoding, args.num_threads, args.max)


if __name__ == "__main__":
    _main()
