# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Benchmark: SPDL pipeline handoff latency with and without thread output queue.

Compares main-thread ``get_item`` latency between the default asyncio-based
handoff (``run_coroutine_threadsafe`` + ``Future.result`` polling) and the
``queue.Queue``-based handoff (direct ``queue.Queue.get``).

The benchmark emulates a realistic training loop: each iteration the
foreground thread receives a batch, does simulated work (busy-wait to
mimic a GPU forward/backward pass), and then times how long the *next*
``get_item`` call takes.  When the simulated work is long enough the
producer has time to pre-fill the queue, so the handoff latency
isolates the cross-thread scheduling overhead.

Foreground work is swept from 0ms to 30ms in 3ms steps.

Usage::

    buck2 run //spdl/examples:benchmark_thread_output_queue

Example results (devserver, no GPU, 500 lightweight int items)::

    FG work   default (p50)   default (p99)   thread_q (p50)   thread_q (p99)
    -------   -------------   -------------   --------------   --------------
      0ms         199us           625us           116us            313us
      3ms         246us           642us           223us            646us
      6ms         232us           532us           190us            555us
      9ms         287us           485us           153us            549us
     12ms         280us           822us            14us            464us
     15ms         221us           396us             8us            385us
     18ms         227us           431us             9us             70us
     21ms         224us           450us            11us             41us
     24ms         240us           533us            12us             34us
     27ms         227us           470us            12us             27us
     30ms         242us           512us            12us             26us

The default asyncio path stays flat at ~220-280us regardless of overlap
time — that is the fixed ``run_coroutine_threadsafe`` scheduling tax.
The thread output queue drops to ~8-14us once the foreground work is
long enough (>= ~12ms on this machine) for the producer to pre-fill it.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from spdl.pipeline import PipelineBuilder

__all__ = ["BenchResult", "main"]


@dataclass
class BenchResult:
    """Collected latency measurements for a single benchmark run."""

    name: str
    latencies_us: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        s = sorted(self.latencies_us)
        return s[len(s) // 2]

    @property
    def p95(self) -> float:
        s = sorted(self.latencies_us)
        return s[int(len(s) * 0.95)]

    @property
    def p99(self) -> float:
        s = sorted(self.latencies_us)
        return s[int(len(s) * 0.99)]

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_us)

    def summary(self) -> str:
        """Return a one-line summary string with mean/p50/p95/p99."""
        return (
            f"{self.name:<50s}  "
            f"mean={self.mean:>9.1f}us  "
            f"p50={self.p50:>9.1f}us  "
            f"p95={self.p95:>9.1f}us  "
            f"p99={self.p99:>9.1f}us  "
            f"({len(self.latencies_us)} meas)"
        )


def _busy_wait_us(us: float) -> None:
    """Busy-wait for *us* microseconds (spin-loop)."""
    deadline = time.perf_counter() + us / 1e6
    while time.perf_counter() < deadline:
        pass


def _run_bench(
    name: str,
    n_items: int,
    buffer_size: int,
    use_thread_output_queue: bool,
    work_us: float,
    warmup: int = 20,
) -> BenchResult:
    """Run a single benchmark: source -> sink pipeline with foreground work.

    Builds a trivial pipeline (source of *n_items* integers -> sink) and
    iterates it.  Between each consumed item the foreground thread
    busy-waits for *work_us* microseconds to simulate consumer-side
    compute (e.g. a GPU training step).  After warmup, the time spent
    inside each ``get_item`` call is recorded.

    Args:
        name: Human-readable label for the result.
        n_items: Total number of items the source produces.
        buffer_size: Sink buffer size (number of slots).
        use_thread_output_queue: Whether to use the thread output queue handoff.
        work_us: Duration of simulated foreground work in microseconds.
        warmup: Number of items to consume before recording latencies.
    """
    items = list(range(n_items))
    pipeline = (
        PipelineBuilder()
        .add_source(iter(items))
        .add_sink(buffer_size=buffer_size)
        .build(num_threads=2, use_thread_output_queue=use_thread_output_queue)
    )

    result = BenchResult(name=name)
    consumed = 0
    for _ in pipeline.get_iterator(timeout=60):
        consumed += 1

        # Simulate foreground work (e.g. GPU fwd+bwd).
        # This gives the producer time to pre-fill the queue
        # so the next get_item measures pure handoff overhead.
        _busy_wait_us(work_us)

        if consumed <= warmup:
            continue

        # Time the next get_item call — this is what we're measuring.
        t0 = time.perf_counter()
        try:
            next(pipeline.get_iterator(timeout=60))
        except StopIteration:
            break
        result.latencies_us.append((time.perf_counter() - t0) * 1e6)
        consumed += 1

        # Do work after the timed item too, so the *next* iteration's
        # measurement also has overlap time.
        _busy_wait_us(work_us)

    return result


def main() -> None:
    """Run the full benchmark sweep and print results."""
    n_items = 500
    buffer_size = 8

    print("SPDL Pipeline Thread Output Queue Handoff Benchmark")
    print("=" * 90)

    results: list[BenchResult] = []

    work_ms_values = list(range(0, 31, 3))
    for work_ms in work_ms_values:
        work_us = work_ms * 1000.0
        label = f"{work_ms}ms foreground work"
        print(f"\n--- {label} ({n_items} items, buffer_size={buffer_size}) ---")

        for use_toq, tag in [
            (False, "default asyncio"),
            (True, "thread output queue"),
        ]:
            r = _run_bench(
                f"{tag}, {label}",
                n_items,
                buffer_size,
                use_thread_output_queue=use_toq,
                work_us=work_us,
            )
            results.append(r)
            print(f"  {r.summary()}")

    print(f"\n{'=' * 90}")
    print("SUMMARY — main thread get_item() latency (lower is better)")
    print(f"{'=' * 90}")
    for r in results:
        if r.latencies_us:
            print(f"  {r.summary()}")


if __name__ == "__main__":
    main()
