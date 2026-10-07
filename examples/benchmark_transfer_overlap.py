#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark overlap between D2H transfers and foreground CUDA compute.

The intended use of :func:`spdl.io.transfer_tensor_d2h` is to call it from a
background CPU thread while the foreground thread submits independent CUDA
work. Raw transfer latency alone does not measure that benefit, so this
benchmark compares serialized and concurrent transfer-plus-compute makespans.
The ``PyTorch to(cpu)`` method is a simple reference path, not a staging- and
allocation-matched replacement for SPDL.

The source tensors are made ready before timing. The transfer worker owns its
copy stream, while the foreground thread owns a separate compute stream. Each
timed block runs compute-only, transfer-only, serialized, and concurrent cases
in randomized order. Concurrent cases use a CPU barrier and report CPU launch
skew so delayed worker scheduling is visible in the result.

Usage::

    python examples/benchmark_transfer_overlap.py \
        --output /tmp/transfer_overlap.csv

Use ``--samples-output`` to retain paired per-trial measurements.

Measured steady-state result
----------------------------

One run used a 64 MiB uint8 payload on an NVIDIA A100 MIG 1g.10gb with
CUDA 13.3. After five warmup blocks, 30 paired randomized blocks produced the
following SPDL medians. The foreground workload was a 2048-square ``torch.mm``.

::

    Tensors  Compute   Transfer  Serialized  Concurrent  Paired speedup (95% CI)
          1  7.262 ms  9.800 ms   17.012 ms    11.852 ms  1.471x [1.266, 1.670]
         32  7.310 ms 11.443 ms   17.757 ms    11.626 ms  1.625x [1.428, 1.693]

These are steady-state results: validation and warmup prime the pinned-memory
and allocator caches before timing. They show that background SPDL execution
reduced transfer-plus-compute makespan on this system. They do not show that
SPDL is faster than the naive PyTorch reference, which also overlaps, or
characterize first-call and cache-growth latency.
"""

import argparse
import csv
import os
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from functools import partial

import numpy as np
import torch
from spdl.io import transfer_tensor_d2h

_SCHEMA_VERSION = 1
_METHOD_NATIVE = "PyTorch to(cpu)"
_METHOD_SPDL = "SPDL"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for an overlap benchmark run."""

    total_mib: tuple[int, ...]
    num_tensors: tuple[int, ...]
    matrix_size: int
    compute_target_ms: float
    compute_iterations: int | None
    num_warmup: int
    num_runs: int
    bootstrap_resamples: int
    seed: int


@dataclass(frozen=True)
class OverlapSample:
    """Paired measurements from one randomized benchmark block."""

    trial: int
    schedule_order: str
    compute_only_ms: float
    transfer_only_ms: float
    serialized_ms: float
    concurrent_ms: float
    cpu_launch_skew_ms: float

    @property
    def speedup(self) -> float:
        """Return serialized-to-concurrent makespan speedup."""
        return self.serialized_ms / self.concurrent_ms

    @property
    def hidden_ms(self) -> float:
        """Return time saved by concurrent execution."""
        return self.serialized_ms - self.concurrent_ms

    @property
    def estimated_overlap_efficiency(self) -> float:
        """Estimate saved time as a fraction of the paired overlap potential.

        The four schedules are timed separately, so noise can produce estimates
        outside the interval from zero to one.
        """
        potential_ms = min(self.compute_only_ms, self.transfer_only_ms)
        return self.hidden_ms / potential_ms if potential_ms > 0 else 0.0


@dataclass(frozen=True)
class BenchmarkSummary:
    """One summary row written to the benchmark output CSV."""

    schema_version: int
    created_utc: str
    device_name: str
    device_capability: str
    torch_version: str
    cuda_version: str
    seed: int
    schedule_seed: int
    bootstrap_seed: int
    bootstrap_resamples: int
    num_warmup: int
    num_runs: int
    total_bytes: int
    num_tensors: int
    dtype: str
    matrix_size: int
    compute_target_ms: float
    compute_iterations: int
    method: str
    compute_median_ms: float
    compute_p95_ms: float
    transfer_median_ms: float
    transfer_p95_ms: float
    transfer_gb_per_s: float
    serialized_median_ms: float
    serialized_p95_ms: float
    concurrent_median_ms: float
    concurrent_p95_ms: float
    speedup_median: float
    speedup_ci_lower: float
    speedup_ci_upper: float
    hidden_median_ms: float
    estimated_overlap_efficiency_median: float
    estimated_overlap_efficiency_ci_lower: float
    estimated_overlap_efficiency_ci_upper: float
    cpu_launch_skew_p95_ms: float


@dataclass(frozen=True)
class RawSampleRecord:
    """One per-trial row optionally written for reproducibility."""

    schema_version: int
    created_utc: str
    seed: int
    schedule_seed: int
    total_bytes: int
    num_tensors: int
    method: str
    trial: int
    schedule_order: str
    compute_only_ms: float
    transfer_only_ms: float
    serialized_ms: float
    concurrent_ms: float
    speedup: float
    hidden_ms: float
    estimated_overlap_efficiency: float
    cpu_launch_skew_ms: float


@dataclass(frozen=True)
class _TransferOutcome:
    output: list[torch.Tensor]
    started_ns: int


@dataclass
class _MutableTrial:
    transfer_only_ms: float = 0.0
    serialized_ms: float = 0.0
    concurrent_ms: float = 0.0
    cpu_launch_skew_ms: float = 0.0


def _native_d2h(
    tensors: list[torch.Tensor], stream: torch.cuda.Stream
) -> list[torch.Tensor]:
    """Transfer tensors to pageable CPU outputs on a dedicated stream."""
    producer_stream = torch.cuda.current_stream(tensors[0].device)
    if producer_stream != stream:
        stream.wait_stream(producer_stream)
    with torch.cuda.stream(stream):
        output = [tensor.to("cpu", non_blocking=True) for tensor in tensors]
    stream.synchronize()
    return output


def _submit_transfer(
    executor: ThreadPoolExecutor,
    fn: Callable[[], list[torch.Tensor]],
    device: torch.device,
    barrier: threading.Barrier | None = None,
) -> Future[_TransferOutcome]:
    """Submit one transfer to the persistent background worker."""

    def _work() -> _TransferOutcome:
        torch.cuda.set_device(device)
        if barrier is not None:
            barrier.wait(timeout=30)
        started_ns = time.perf_counter_ns()
        output = fn()
        return _TransferOutcome(output, started_ns)

    return executor.submit(_work)


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    """Convert a monotonic nanosecond interval to milliseconds."""
    return (end_ns - start_ns) / 1_000_000


def _time_compute(run_compute: Callable[[], torch.Tensor]) -> float:
    """Measure one synchronous foreground-compute call."""
    start_ns = time.perf_counter_ns()
    run_compute()
    return _elapsed_ms(start_ns, time.perf_counter_ns())


def _time_transfer(
    executor: ThreadPoolExecutor,
    fn: Callable[[], list[torch.Tensor]],
    device: torch.device,
) -> float:
    """Measure one end-to-end background transfer."""
    start_ns = time.perf_counter_ns()
    outcome = _submit_transfer(executor, fn, device).result()
    end_ns = time.perf_counter_ns()
    del outcome
    return _elapsed_ms(start_ns, end_ns)


def _time_serialized(
    executor: ThreadPoolExecutor,
    fn: Callable[[], list[torch.Tensor]],
    run_compute: Callable[[], torch.Tensor],
    device: torch.device,
) -> float:
    """Measure transfer followed by foreground compute."""
    start_ns = time.perf_counter_ns()
    outcome = _submit_transfer(executor, fn, device).result()
    run_compute()
    end_ns = time.perf_counter_ns()
    del outcome
    return _elapsed_ms(start_ns, end_ns)


def _time_concurrent(
    executor: ThreadPoolExecutor,
    fn: Callable[[], list[torch.Tensor]],
    run_compute: Callable[[], torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    """Measure barrier-coordinated transfer and foreground compute."""
    barrier = threading.Barrier(2)
    start_ns = time.perf_counter_ns()
    future = _submit_transfer(executor, fn, device, barrier)
    barrier.wait(timeout=30)
    compute_started_ns = time.perf_counter_ns()
    run_compute()
    outcome = future.result()
    end_ns = time.perf_counter_ns()
    launch_skew_ms = _elapsed_ms(
        min(compute_started_ns, outcome.started_ns),
        max(compute_started_ns, outcome.started_ns),
    )
    del outcome
    return _elapsed_ms(start_ns, end_ns), launch_skew_ms


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return one percentile as a Python float."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _bootstrap_median_ci(
    values: Sequence[float], *, seed: int, num_resamples: int
) -> tuple[float, float]:
    """Return a percentile-bootstrap 95% confidence interval for the median."""
    samples = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(samples), size=(num_resamples, len(samples)))
    medians = np.median(samples[indices], axis=1)
    return (
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def _calibrate_compute_iterations(
    run_compute: Callable[[int], torch.Tensor], target_ms: float
) -> int:
    """Calibrate real matrix multiplication to an independent duration target."""
    iterations = 1
    for _ in range(4):
        samples = []
        for _ in range(3):
            start_ns = time.perf_counter_ns()
            run_compute(iterations)
            samples.append(_elapsed_ms(start_ns, time.perf_counter_ns()))
        measured_ms = _percentile(samples, 50)
        estimate = round(iterations * target_ms / max(measured_ms, 0.001))
        estimate = max(1, min(100_000, estimate))
        if estimate == iterations:
            break
        iterations = estimate
    return iterations


def _create_source_tensors(
    total_bytes: int, num_tensors: int, device: torch.device
) -> list[torch.Tensor]:
    """Create uint8 CUDA tensors whose sizes sum to ``total_bytes``."""
    if num_tensors > total_bytes:
        raise ValueError(
            f"Cannot split {total_bytes} bytes into {num_tensors} nonempty tensors."
        )
    size, remainder = divmod(total_bytes, num_tensors)
    return [
        torch.randint(
            0,
            256,
            (size + (index < remainder),),
            dtype=torch.uint8,
            device=device,
        )
        for index in range(num_tensors)
    ]


def _validate_methods(
    executor: ThreadPoolExecutor,
    methods: dict[str, Callable[[], list[torch.Tensor]]],
    tensors: list[torch.Tensor],
    device: torch.device,
) -> None:
    """Prime each method on the worker and validate exact transfer results."""
    expected = [tensor.cpu() for tensor in tensors]
    for method in methods.values():
        outcome = _submit_transfer(executor, method, device).result()
        for actual, reference in zip(outcome.output, expected, strict=True):
            torch.testing.assert_close(actual, reference)
        del outcome


def _collect_samples(
    executor: ThreadPoolExecutor,
    methods: dict[str, Callable[[], list[torch.Tensor]]],
    run_compute: Callable[[], torch.Tensor],
    device: torch.device,
    *,
    num_warmup: int,
    num_runs: int,
    seed: int,
) -> dict[str, list[OverlapSample]]:
    """Collect randomized, paired timing blocks for every transfer method."""
    for _ in range(num_warmup):
        run_compute()
        for method in methods.values():
            _time_transfer(executor, method, device)
            _time_concurrent(executor, method, run_compute, device)

    rng = random.Random(seed)
    samples = {method: [] for method in methods}
    cases = [("compute", "")]
    cases.extend(
        (method, schedule)
        for method in methods
        for schedule in ("transfer", "serialized", "concurrent")
    )

    for trial in range(num_runs):
        order = list(cases)
        rng.shuffle(order)
        order_label = "|".join(
            method if method == "compute" else f"{method}:{schedule}"
            for method, schedule in order
        )
        compute_only_ms = 0.0
        trial_results = {method: _MutableTrial() for method in methods}
        for method, schedule in order:
            if method == "compute":
                compute_only_ms = _time_compute(run_compute)
                continue

            transfer = methods[method]
            result = trial_results[method]
            if schedule == "transfer":
                result.transfer_only_ms = _time_transfer(executor, transfer, device)
            elif schedule == "serialized":
                result.serialized_ms = _time_serialized(
                    executor, transfer, run_compute, device
                )
            else:
                (
                    result.concurrent_ms,
                    result.cpu_launch_skew_ms,
                ) = _time_concurrent(executor, transfer, run_compute, device)

        for method, result in trial_results.items():
            samples[method].append(
                OverlapSample(
                    trial=trial,
                    schedule_order=order_label,
                    compute_only_ms=compute_only_ms,
                    transfer_only_ms=result.transfer_only_ms,
                    serialized_ms=result.serialized_ms,
                    concurrent_ms=result.concurrent_ms,
                    cpu_launch_skew_ms=result.cpu_launch_skew_ms,
                )
            )
    return samples


def _summarize(
    samples: list[OverlapSample],
    *,
    created_utc: str,
    device: torch.device,
    config: BenchmarkConfig,
    total_bytes: int,
    num_tensors: int,
    compute_iterations: int,
    method: str,
    schedule_seed: int,
    bootstrap_seed: int,
) -> BenchmarkSummary:
    """Summarize one case and method into a stable CSV schema."""
    compute = [sample.compute_only_ms for sample in samples]
    transfer = [sample.transfer_only_ms for sample in samples]
    serialized = [sample.serialized_ms for sample in samples]
    concurrent = [sample.concurrent_ms for sample in samples]
    speedups = [sample.speedup for sample in samples]
    hidden = [sample.hidden_ms for sample in samples]
    efficiencies = [sample.estimated_overlap_efficiency for sample in samples]
    launch_skews = [sample.cpu_launch_skew_ms for sample in samples]
    speedup_ci = _bootstrap_median_ci(
        speedups, seed=bootstrap_seed, num_resamples=config.bootstrap_resamples
    )
    efficiency_ci = _bootstrap_median_ci(
        efficiencies,
        seed=bootstrap_seed + 1,
        num_resamples=config.bootstrap_resamples,
    )
    transfer_median_ms = _percentile(transfer, 50)
    capability = torch.cuda.get_device_capability(device)
    return BenchmarkSummary(
        schema_version=_SCHEMA_VERSION,
        created_utc=created_utc,
        device_name=torch.cuda.get_device_name(device),
        device_capability=f"{capability[0]}.{capability[1]}",
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda or "unknown"),
        seed=config.seed,
        schedule_seed=schedule_seed,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
        num_warmup=config.num_warmup,
        num_runs=config.num_runs,
        total_bytes=total_bytes,
        num_tensors=num_tensors,
        dtype="torch.uint8",
        matrix_size=config.matrix_size,
        compute_target_ms=config.compute_target_ms,
        compute_iterations=compute_iterations,
        method=method,
        compute_median_ms=_percentile(compute, 50),
        compute_p95_ms=_percentile(compute, 95),
        transfer_median_ms=transfer_median_ms,
        transfer_p95_ms=_percentile(transfer, 95),
        transfer_gb_per_s=total_bytes / (transfer_median_ms * 1_000_000),
        serialized_median_ms=_percentile(serialized, 50),
        serialized_p95_ms=_percentile(serialized, 95),
        concurrent_median_ms=_percentile(concurrent, 50),
        concurrent_p95_ms=_percentile(concurrent, 95),
        speedup_median=_percentile(speedups, 50),
        speedup_ci_lower=speedup_ci[0],
        speedup_ci_upper=speedup_ci[1],
        hidden_median_ms=_percentile(hidden, 50),
        estimated_overlap_efficiency_median=_percentile(efficiencies, 50),
        estimated_overlap_efficiency_ci_lower=efficiency_ci[0],
        estimated_overlap_efficiency_ci_upper=efficiency_ci[1],
        cpu_launch_skew_p95_ms=_percentile(launch_skews, 95),
    )


def _print_summary(summary: BenchmarkSummary) -> None:
    """Print one compact, human-readable result row."""
    print(
        f"  {summary.method:<6} "
        f"compute={summary.compute_median_ms:7.3f} ms  "
        f"transfer={summary.transfer_median_ms:7.3f} ms  "
        f"serialized={summary.serialized_median_ms:7.3f} ms  "
        f"concurrent={summary.concurrent_median_ms:7.3f} ms  "
        f"speedup={summary.speedup_median:5.2f}x  "
        f"estimated_overlap="
        f"{summary.estimated_overlap_efficiency_median * 100:6.1f}%"
    )


def _write_csv(records: Sequence[object], output_path: str, record_type: type) -> None:
    """Write dataclass records to a self-contained CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [field.name for field in fields(record_type)]
    with open(output_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {fieldname: getattr(record, fieldname) for fieldname in fieldnames}
            )


def _run_benchmarks(
    config: BenchmarkConfig, device: torch.device
) -> tuple[list[BenchmarkSummary], list[RawSampleRecord]]:
    """Run all configured overlap scenarios and return summary and raw rows."""
    torch.manual_seed(config.seed)
    torch.cuda.set_device(device)
    current_index = torch.cuda.current_device()
    device = torch.device("cuda", current_index)

    left = torch.randn(config.matrix_size, config.matrix_size, device=device)
    right = torch.randn(config.matrix_size, config.matrix_size, device=device)
    compute_output = torch.empty_like(left)
    compute_stream = torch.cuda.Stream(device=device)
    compute_stream.wait_stream(torch.cuda.current_stream(device))

    def _run_compute(iterations: int) -> torch.Tensor:
        with torch.cuda.stream(compute_stream):
            for _ in range(iterations):
                torch.mm(left, right, out=compute_output)
        compute_stream.synchronize()
        return compute_output

    _run_compute(1)
    compute_iterations = config.compute_iterations
    if compute_iterations is None:
        compute_iterations = _calibrate_compute_iterations(
            _run_compute, config.compute_target_ms
        )

    def _compute() -> torch.Tensor:
        return _run_compute(compute_iterations)

    _compute()
    if not torch.isfinite(compute_output).all().item():
        raise RuntimeError(
            "Foreground matrix multiplication produced non-finite output."
        )

    created_utc = datetime.now(timezone.utc).isoformat()
    summaries = []
    raw_records = []
    scenarios = [
        (total_mib * 1024 * 1024, num_tensors)
        for total_mib in config.total_mib
        for num_tensors in config.num_tensors
    ]
    random.Random(config.seed).shuffle(scenarios)

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="spdl-d2h-benchmark"
    ) as executor:
        for scenario_index, (total_bytes, num_tensors) in enumerate(scenarios):
            print(
                f"\n{total_bytes / (1024 * 1024):.0f} MiB in "
                f"{num_tensors} tensor(s), {compute_iterations} compute iteration(s)"
            )
            tensors = _create_source_tensors(total_bytes, num_tensors, device)
            torch.cuda.synchronize(device)
            native_stream = torch.cuda.Stream(device=device)
            methods: dict[str, Callable[[], list[torch.Tensor]]] = {
                _METHOD_NATIVE: partial(_native_d2h, tensors, native_stream),
                _METHOD_SPDL: partial(transfer_tensor_d2h, tensors, device=device),
            }
            _validate_methods(executor, methods, tensors, device)
            scenario_seed = config.seed + scenario_index * 1009
            samples = _collect_samples(
                executor,
                methods,
                _compute,
                device,
                num_warmup=config.num_warmup,
                num_runs=config.num_runs,
                seed=scenario_seed,
            )
            for method_index, (method, method_samples) in enumerate(samples.items()):
                bootstrap_seed = scenario_seed + method_index * 17
                summary = _summarize(
                    method_samples,
                    created_utc=created_utc,
                    device=device,
                    config=config,
                    total_bytes=total_bytes,
                    num_tensors=num_tensors,
                    compute_iterations=compute_iterations,
                    method=method,
                    schedule_seed=scenario_seed,
                    bootstrap_seed=bootstrap_seed,
                )
                summaries.append(summary)
                _print_summary(summary)
                raw_records.extend(
                    RawSampleRecord(
                        schema_version=_SCHEMA_VERSION,
                        created_utc=created_utc,
                        seed=config.seed,
                        schedule_seed=scenario_seed,
                        total_bytes=total_bytes,
                        num_tensors=num_tensors,
                        method=method,
                        trial=sample.trial,
                        schedule_order=sample.schedule_order,
                        compute_only_ms=sample.compute_only_ms,
                        transfer_only_ms=sample.transfer_only_ms,
                        serialized_ms=sample.serialized_ms,
                        concurrent_ms=sample.concurrent_ms,
                        speedup=sample.speedup,
                        hidden_ms=sample.hidden_ms,
                        estimated_overlap_efficiency=(
                            sample.estimated_overlap_efficiency
                        ),
                        cpu_launch_skew_ms=sample.cpu_launch_skew_ms,
                    )
                    for sample in method_samples
                )
            del tensors

    return summaries, raw_records


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _at_least_two_int(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark overlap between D2H transfer and CUDA compute."
    )
    parser.add_argument(
        "--output",
        type=lambda path: os.path.realpath(path),
        required=True,
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--samples-output",
        type=lambda path: os.path.realpath(path),
        help="Optional per-trial CSV output path.",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use.")
    parser.add_argument(
        "--total-mib",
        type=_positive_int,
        nargs="+",
        default=[64],
        help="Fixed total payload sizes in MiB.",
    )
    parser.add_argument(
        "--num-tensors",
        type=_positive_int,
        nargs="+",
        default=[1, 32],
        help="Tensor counts to test for each fixed payload size.",
    )
    parser.add_argument(
        "--matrix-size",
        type=_positive_int,
        default=2048,
        help="Square float32 matrix size for foreground torch.mm work.",
    )
    parser.add_argument(
        "--compute-ms",
        type=_positive_float,
        default=8.0,
        help="Independent target duration used to calibrate foreground compute.",
    )
    parser.add_argument(
        "--compute-iterations",
        type=_positive_int,
        help="Skip calibration and use this many torch.mm calls per sample.",
    )
    parser.add_argument(
        "--num-warmup",
        type=_non_negative_int,
        default=5,
        help="Warmup blocks per scenario.",
    )
    parser.add_argument(
        "--num-runs",
        type=_at_least_two_int,
        default=30,
        help="Paired randomized timing blocks per scenario.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=_positive_int,
        default=2000,
        help="Bootstrap resamples for paired median confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Randomization seed.")
    return parser.parse_args()


def main() -> None:
    """Run the overlap benchmark and write its results."""
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, but received {device}.")
    config = BenchmarkConfig(
        total_mib=tuple(args.total_mib),
        num_tensors=tuple(args.num_tensors),
        matrix_size=args.matrix_size,
        compute_target_ms=args.compute_ms,
        compute_iterations=args.compute_iterations,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    print("D2H transfer / foreground CUDA compute overlap benchmark")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print("PyTorch to(cpu) is a naive reference; compare overlap within each method.")
    summaries, raw_records = _run_benchmarks(config, device)
    _write_csv(summaries, args.output, BenchmarkSummary)
    print(f"\nSummary saved to: {args.output}")
    if args.samples_output:
        _write_csv(raw_records, args.samples_output, RawSampleRecord)
        print(f"Raw samples saved to: {args.samples_output}")


if __name__ == "__main__":
    main()
