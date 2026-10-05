#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Benchmark equivalent ways to batch inputs at an execution-region boundary.

Compares explicit ``aggregate(N)`` / ``disaggregate()`` stages with the transparent
``to(..., buffer_size=N)`` transport buffer. The worker operation is async, and its output is
aggregated before returning to the main process, so synchronous-op dispatch and outbound IPC do
not hide the cost of unpacking inbound transfers.

The first pass through each topology is discarded as warmup. A fresh pipeline is built for
every pass because this benchmark deliberately uses a finite, non-continuous source to cover
the worker session path.

Reference result with the default parameters (throughput is host-dependent):

* Before the fix: explicit 7,178 items/s; buffered 4,674 items/s (0.651x).
* After the fix: explicit 6,984 items/s; buffered 7,541 items/s (1.080x).

Example::

    python examples/benchmark_region_buffer.py \\
        --num-items 50000 --batch-size 128 --num-workers 1 --runs 3
"""

from __future__ import annotations

import argparse
import multiprocessing
import statistics
import time
from collections.abc import Callable

from spdl.pipeline import Pipeline, PipelineBuilder
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

__all__ = ["main"]


async def _identity(item: int) -> int:
    return item


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _build_explicit(
    num_items: int,
    batch_size: int,
    num_workers: int,
    mp_context: str | None,
) -> Pipeline[list[int]]:
    config = ProcessPoolExecutorConfig(
        max_workers=num_workers,
        mp_context=mp_context,
    )
    return (
        PipelineBuilder()
        .add_source(range(num_items))
        .aggregate(batch_size)
        .to(config)
        .disaggregate()
        .pipe(_identity)
        .aggregate(batch_size)
        .to(MAIN_PROCESS)
        .add_sink(8)
        .build(num_threads=2)
    )


def _build_buffered(
    num_items: int,
    batch_size: int,
    num_workers: int,
    mp_context: str | None,
) -> Pipeline[list[int]]:
    config = ProcessPoolExecutorConfig(
        max_workers=num_workers,
        mp_context=mp_context,
    )
    return (
        PipelineBuilder()
        .add_source(range(num_items))
        .to(config, buffer_size=batch_size)
        .pipe(_identity)
        .aggregate(batch_size)
        .to(MAIN_PROCESS)
        .add_sink(8)
        .build(num_threads=2)
    )


def _measure_once(build: Callable[[], Pipeline[list[int]]], num_items: int) -> float:
    pipeline = build()
    try:
        start = time.perf_counter()
        count = sum(len(batch) for batch in pipeline.get_iterator(timeout=60))
        elapsed = time.perf_counter() - start
    finally:
        pipeline.stop()
    if count != num_items:
        raise RuntimeError(f"expected {num_items} items, got {count}")
    return count / elapsed


def _run(
    builds: list[tuple[str, Callable[[], Pipeline[list[int]]]]],
    runs: int,
    num_items: int,
) -> dict[str, list[float]]:
    throughputs = {name: [] for name, _ in builds}
    for run in range(runs + 1):
        ordered = builds if run % 2 == 0 else list(reversed(builds))
        for name, build in ordered:
            throughput = _measure_once(build, num_items)
            if run:
                throughputs[name].append(throughput)
    return throughputs


def main() -> None:
    """Run both boundary-batching topologies and print their median throughput."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-items", type=_positive_int, default=50_000)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--num-workers", type=_positive_int, default=1)
    parser.add_argument("--runs", type=_positive_int, default=3)
    parser.add_argument(
        "--mp-context",
        choices=multiprocessing.get_all_start_methods(),
        default="spawn",
    )
    args = parser.parse_args()

    builds = [
        (
            "explicit",
            lambda: _build_explicit(
                args.num_items,
                args.batch_size,
                args.num_workers,
                args.mp_context,
            ),
        ),
        (
            "buffered",
            lambda: _build_buffered(
                args.num_items,
                args.batch_size,
                args.num_workers,
                args.mp_context,
            ),
        ),
    ]
    throughputs = _run(builds, args.runs, args.num_items)
    explicit_median = statistics.median(throughputs["explicit"])
    buffered_median = statistics.median(throughputs["buffered"])
    print(f"explicit aggregate/disaggregate: {explicit_median:,.0f} items/s")
    print(f"boundary buffer_size:           {buffered_median:,.0f} items/s")
    print(f"boundary / explicit:            {buffered_median / explicit_median:.3f}x")


if __name__ == "__main__":
    main()
