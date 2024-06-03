"""Bulk tokenize text files with OpenAI's TikToken"""

import asyncio
import concurrent.futures
from pathlib import Path

import spdl.io
import spdl.utils
import tiktoken
import torch
from spdl.dataset._utils import _iter_flist
from spdl.io import CUDAConfig
from spdl.io._core import _run_async
from spdl.lib import _libspdl


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--input-flist", type=Path, required=True, help="File lilst")
    parser.add_argument("--prefix", help="Root directory of the dataset")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args(args)


def _tokenize(sentence, encoding, cuda_config: CUDAConfig | None = None):
    with spdl.utils.trace_event("encode"):
        tokens = encoding.encode(sentence)
    with spdl.utils.trace_event("convert_tokens_1d"):
        buffer = _libspdl.convert_tokens_1d(tokens)
    if cuda_config is not None:
        buffer = _libspdl.transfer_buffer(
            buffer,
            cuda_config=cuda_config,
        )
    return buffer


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

    async def _func(data):
        async with semaphore:
            data = _read(path)
            buffer = await _run_async(
                _tokenize, data, encoding, cuda_config=cuda_config
            )
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
    while (item := await queue.get()) is not sentinel:
        print(item)

    await task


def _read(path):
    with open(path, "r") as fp:
        return fp.read()


def _test(input_flist, prefix, encoding, num_threads):
    encoding = tiktoken.get_encoding(encoding)
    file_gen = _iter_flist(input_flist, prefix=prefix, max=2000)

    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    )
    with spdl.utils.tracing("trace_tiktoken"):
        loop.run_until_complete(_run(file_gen, encoding))


def _main(args=None):
    args = _parse_args(args)

    _test(args.input_flist, args.prefix, args.encoding, args.num_threads)


if __name__ == "__main__":
    _main()
