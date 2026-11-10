# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Compare handoff with False, ThreadOutputQueue and StatsThreadOutputQueue.

Runs a source-to-sink pipeline with 500 integer items, buffer capacity 8 and two
configured executor workers. Foreground CPU busy-work is swept from 0 to 30 ms
in 3 ms steps. After 20 warmup items, timed and untimed iterator reads alternate,
producing 240 latency observations per variant and work setting.

The busy loop creates CPU/GIL contention rather than real GPU work. Results
measure iterator handoff, not sustained-load, training or GPU-overlap performance.
The False control uses the configured async sink (a synchronous mirror for the
built-in StatsQueue); both thread variants are explicitly selected classes.
StatsThreadOutputQueue collects with its default final reporting callback;
periodic reporting is disabled. Existing internal queues and task hooks retain
normal defaults in every variant.

Usage::

    buck2 run //spdl/examples:benchmark_thread_output_queue

The complete measured comparison is documented on
:py:class:`spdl.pipeline.ThreadOutputQueue` and
:py:class:`spdl.pipeline.StatsThreadOutputQueue`.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from spdl.pipeline import PipelineBuilder, StatsThreadOutputQueue, ThreadOutputQueue

__all__ = ["BenchResult", "run_benchmark", "main"]


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


def run_benchmark(
    name: str,
    n_items: int,
    buffer_size: int,
    use_thread_output_queue: bool | type[ThreadOutputQueue],
    work_us: float,
    warmup: int = 20,
) -> BenchResult:
    """Run a source-to-sink pipeline with foreground CPU busy-work.

    After warmup, every other iterator advance is timed. The public build option
    selects the configured async sink or an explicit thread-output class.

    Args:
        name: Human-readable label for the result.
        n_items: Total number of integer items the source produces.
        buffer_size: Sink buffer size in items; zero means unbounded.
        use_thread_output_queue: ``False``, ``True``, or a thread-output queue class.
        work_us: Duration of foreground CPU busy-work in microseconds.
        warmup: Number of items to consume before recording latencies.

    Returns:
        Latencies of timed iterator advances, in microseconds. The workload
        alternates timed and untimed reads after warmup; it is CPU-only and does
        not establish sustained-load or GPU-overlap performance.
    """
    if n_items <= warmup + 1 or warmup < 0:
        raise ValueError("n_items must exceed warmup by at least two")
    if buffer_size < 0 or work_us < 0:
        raise ValueError("buffer_size and work_us must be nonnegative")
    items = list(range(n_items))
    pipeline = (
        PipelineBuilder()
        .add_source(iter(items))
        .add_sink(buffer_size=buffer_size)
        .build(num_threads=2, use_thread_output_queue=use_thread_output_queue)
    )

    result = BenchResult(name=name)
    consumed = 0
    try:
        for _ in pipeline.get_iterator(timeout=60):
            consumed += 1
            _busy_wait_us(work_us)
            if consumed <= warmup:
                continue

            t0 = time.perf_counter()
            try:
                next(pipeline.get_iterator(timeout=60))
            except StopIteration:
                break
            result.latencies_us.append((time.perf_counter() - t0) * 1e6)
            consumed += 1
            _busy_wait_us(work_us)
    finally:
        pipeline.stop(timeout=60)
    return result


def main() -> None:
    """Run the three-way sweep and print every mean/p50/p95/p99 result."""
    n_items = 500
    buffer_size = 8

    print("SPDL Pipeline Thread Output Queue Handoff Benchmark")
    print("=" * 90)

    results: list[BenchResult] = []
    for work_ms in range(0, 31, 3):
        label = f"{work_ms}ms foreground work"
        print(f"\n--- {label} ({n_items} items, buffer_size={buffer_size}) ---")
        for selection, tag in [
            (False, "False"),
            (ThreadOutputQueue, "ThreadOutputQueue"),
            (StatsThreadOutputQueue, "StatsThreadOutputQueue"),
        ]:
            result = run_benchmark(
                f"{tag}, {label}",
                n_items,
                buffer_size,
                use_thread_output_queue=selection,
                work_us=work_ms * 1000.0,
            )
            results.append(result)
            print(f"  {result.summary()}")

    print(f"\n{'=' * 90}")
    print("SUMMARY — foreground iterator handoff latency (lower is better)")
    print(f"{'=' * 90}")
    for result in results:
        print(f"  {result.summary()}")


if __name__ == "__main__":
    main()
