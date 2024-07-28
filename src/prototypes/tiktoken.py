# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bulk tokenize text files with OpenAI's TikToken"""

# pyre-ignore-all-errors

import logging
import time
from pathlib import Path

import numpy as np

import spdl.io
import spdl.utils
import tiktoken
import torch
from spdl.dataloader import PipelineBuilder


_LG = logging.getLogger(__name__)


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
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--max", type=int, default=float("inf"))
    return parser.parse_args(args)


def _iter_file(path, prefix: str, max: int = float("inf")):
    with open(path, "r") as f:
        i = 0
        for line in f:
            if line := line.strip():
                yield prefix + line
                if (i := i + 1) >= max:
                    return


def _read_file(path):
    with spdl.utils.trace_event("read"):
        with open(path, "r") as fp:
            return fp.read()


def _get_pipeline(
    flist, prefix, encoding, *, num_threads, max_items: int = float("inf")
):
    tokenizer = tiktoken.get_encoding(encoding)

    cuda_config = spdl.io.cuda_config(
        device_index=0,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
    )

    def _tokenize(text):
        with spdl.utils.trace_event("encode"):
            tokens = tokenizer.encode(text)
        with spdl.utils.trace_event("np.array"):
            arr = np.array(tokens)
        buffer = spdl.io.transfer_buffer(arr, cuda_config=cuda_config)
        arr = spdl.io.to_torch(buffer)
        return arr

    pipeline = (
        PipelineBuilder()
        .add_source(_iter_file(flist, prefix, max_items))
        .pipe(_read_file, concurrency=2, report_stats_interval=10)
        .pipe(_tokenize, concurrency=1, report_stats_interval=10)
        .add_sink(100)
        .build()
    )

    return pipeline


def _main(args=None):
    args = _parse_args(args)
    logging.basicConfig(level=logging.INFO)

    path = f"{args.trace}.pftrace"

    pipeline = _get_pipeline(
        args.input_flist,
        args.prefix,
        args.encoding,
        num_threads=args.num_threads,
        max_items=args.max,
    )

    with spdl.utils.tracing(path, enable=args.trace):
        t0 = time.monotonic()
        with pipeline.auto_stop():
            num_tokens, num_files = 0, 0
            for tensor in pipeline:
                num_files += 1
                num_tokens += tensor.numel()
        elapsed = time.monotonic() - t0
    _LG.info("#files: %d", num_files)
    _LG.info("#tokens: %d", num_tokens)
    _LG.info("QPS: %.2f", num_tokens / elapsed)


if __name__ == "__main__":
    _main()
