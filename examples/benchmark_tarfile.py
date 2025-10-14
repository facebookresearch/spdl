#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Benchmark script for iter_tarfile function.

This script benchmarks the performance of :py:func:`spdl.io.iter_tarfile` against
Python's built-in ``tarfile`` module using multi-threading.
Two types of inputs are tested for  :py:func:`spdl.io.iter_tarfile`.
Byte string and a file-like object returns byte string by chunk.

The benchmark:

1. Creates test tar archives with various numbers of files
2. Runs both implementations with different thread counts
3. Measures queries per second (QPS) for each configuration
4. Plots the results comparing the three implementations

**Result**

The following plot shows the QPS (measured by the number of files processed) of each
functions with different file size.

.. image:: ../../_static/data/example_benchmark_tarfile.png

.. image:: ../../_static/data/example_benchmark_tarfile_2.png

The :py:func:`spdl.io.iter_tarfile` function processes data fastest when the input is a byte
string.
Its performance is consistent across different file sizes.
This is because, when the entire TAR file is loaded into memory as a contiguous array,
the function only needs to read the header and return the address of the corresponding data
(note that :py:func:`~spdl.io.iter_tarfile` returns a memory view when the input is a byte
string).
Since reading the header is very fast, most of the time is spent creating memory view objects
while holding the GIL (Global Interpreter Lock).
As a result, the speed of loading files decreases as more threads are used.

When the input data type is switched from a byte string to a file-like object,
the performance of :py:func:`spdl.io.iter_tarfile` is also affected by the size of
the input data.
This is because data is processed incrementally, and for each file in the TAR archive,
a new byte string object is created.
The implementation tries to request the exact amount of bytes needed, but file-like objects
do not guarantee that they return the requested length,
instead, they return at most the requested number of bytes.
Therefore, many intermediate byte string objects must be created.
As the file size grows, it takes longer to process the data.
Since the GIL must be locked while byte strings are created,
performance degrades as more threads are used.
At some point, the performance becomes similar to Python's built-in ``tarfile`` module,
which is a pure-Python implementation and thus holds the GIL almost entirely.
"""

__all__ = [
    "BenchmarkResult",
    "benchmark",
    "create_test_tar",
    "iter_tarfile_builtin",
    "main",
    "plot_results",
    "process_tar_builtin",
    "process_tar_spdl",
    "process_tar_spdl_filelike",
    "run_benchmark",
]

import argparse
import io
import logging
import os
import tarfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
import spdl.io

_LG = logging.getLogger(__name__)


def iter_tarfile_builtin(tar_data: bytes) -> Iterator[tuple[str, bytes]]:
    """Iterate over TAR file using Python's built-in ``tarfile`` module.

    Args:
        tar_data: TAR archive as bytes.

    Yields:
        Tuple of ``(filename, content)`` for each file in the archive.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                file_obj = tar.extractfile(member)
                if file_obj:
                    content = file_obj.read()
                    yield member.name, content


def process_tar_spdl(tar_data: bytes, convert: bool) -> int:
    """Process TAR archive using :py:func:`spdl.io.iter_tarfile`.

    Args:
        tar_data: TAR archive as bytes.

    Returns:
        Number of files processed.
    """
    count = 0
    if convert:
        for _, content in spdl.io.iter_tarfile(tar_data):
            bytes(content)
            count += 1
        return count
    else:
        for _ in spdl.io.iter_tarfile(tar_data):
            count += 1
        return count


def process_tar_builtin(tar_data: bytes) -> int:
    """Process TAR archive using Python's built-in ``tarfile`` module.

    Args:
        tar_data: TAR archive as bytes.

    Returns:
        Number of files processed.
    """
    count = 0
    for _ in iter_tarfile_builtin(tar_data):
        count += 1
    return count


def process_tar_spdl_filelike(tar_data: bytes) -> int:
    """Process TAR archive using :py:func:`spdl.io.iter_tarfile` with file-like object.

    Args:
        tar_data: TAR archive as bytes.

    Returns:
        Number of files processed.
    """
    count = 0
    file_like = io.BytesIO(tar_data)
    for _ in spdl.io.iter_tarfile(file_like):  # pyre-ignore[6]
        count += 1
    return count


def benchmark(
    func,
    tar_data: bytes,
    num_iterations: int,
    num_threads: int,
    num_runs: int = 5,
) -> tuple[int, float, float, float]:
    """Benchmark function with specified number of threads.

    Runs multiple benchmark iterations and calculates 95% confidence intervals.

    Args:
        func: Function to benchmark (e.g., ``process_tar_spdl`` or ``process_tar_builtin``).
        tar_data: TAR archive as bytes.
        num_iterations: Number of iterations to run per benchmark run.
        num_threads: Number of threads to use.
        num_runs: Number of benchmark runs to perform for confidence interval calculation.
            Defaults to 5.

    Returns:
        Tuple of ``(total_files_processed, qps_mean, qps_lower_ci, qps_upper_ci)``.
    """
    qps_samples = []
    last_total_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        # Warm-up phase: run a few iterations to warm up the executor
        warmup_futures = [exe.submit(func, tar_data) for _ in range(10 * num_threads)]
        for future in as_completed(warmup_futures):
            _ = future.result()

        # Run multiple benchmark iterations
        for _ in range(num_runs):
            t0 = time.monotonic()
            futures = [exe.submit(func, tar_data) for _ in range(num_iterations)]
            total_count = 0
            for future in as_completed(futures):
                total_count += future.result()
            elapsed = time.monotonic() - t0

            qps = num_iterations / elapsed
            qps_samples.append(qps)
            last_total_count = total_count

    # Calculate mean and 95% confidence interval
    qps_mean = sum(qps_samples) / len(qps_samples)
    qps_std = np.std(qps_samples, ddof=1)
    # Using t-distribution critical value for 95% CI
    # For small samples (n=5), t-value ≈ 2.776
    t_value = 2.776 if num_runs == 5 else 2.0
    margin = t_value * qps_std / (num_runs**0.5)
    qps_lower_ci = qps_mean - margin
    qps_upper_ci = qps_mean + margin

    return last_total_count, qps_mean, qps_lower_ci, qps_upper_ci


def _size_str(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024: .2f} kB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024): .2f} MB"
    return f"{n / (1024 * 1024 * 1024): .2f} GB"


@dataclass
class BenchmarkResult:
    """Single benchmark result for a specific configuration."""

    function_name: str
    "Name of the function being benchmarked."
    tar_size: int
    "Size of the TAR archive in bytes."
    file_size: int
    "Size of each file in the TAR archive in bytes."
    num_files: int
    "Number of files in the TAR archive."
    num_threads: int
    "Number of threads used for this benchmark."
    num_iterations: int
    "Number of iterations performed."
    qps_mean: float
    "Mean queries per second (QPS)."
    qps_lower_ci: float
    "Lower bound of 95% confidence interval for QPS."
    qps_upper_ci: float
    "Upper bound of 95% confidence interval for QPS."
    total_files_processed: int
    "Total number of files processed during the benchmark."


def create_test_tar(num_files: int, file_size: int) -> bytes:
    """Create a TAR archive in memory with specified number of files.

    Args:
        num_files: Number of files to include in the archive.
        file_size: Size of each file in bytes.

    Returns:
        TAR archive as bytes.
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for i in range(num_files):
            filename = f"file_{i:06d}.txt"
            content = b"1" * file_size
            info = tarfile.TarInfo(name=filename)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    tar_buffer.seek(0)
    return tar_buffer.getvalue()


def run_benchmark(
    configs: list[tuple[str, Callable[[bytes], int]]],
    num_files: int,
    file_sizes: list[int],
    num_iterations: int,
    thread_counts: list[int],
    num_runs: int = 5,
) -> list[BenchmarkResult]:
    """Run benchmark comparing SPDL and built-in implementations.

    Tests both :py:func:`spdl.io.iter_tarfile` (with bytes and file-like inputs)
    and Python's built-in ``tarfile`` module.

    Args:
        num_files: Number of files in the test TAR archive.
        file_sizes: List of file sizes to test (in bytes).
        num_iterations: Number of iterations for each thread count.
        thread_counts: List of thread counts to test.
        num_runs: Number of runs to perform for confidence interval calculation.
            Defaults to 5.

    Returns:
        List of :py:class:`BenchmarkResult`, one for each configuration tested.
    """

    results: list[BenchmarkResult] = []

    for file_size in file_sizes:
        for func_name, func in configs:
            tar_data = create_test_tar(num_files, file_size)
            _LG.info(
                "TAR size: %s (%d x %s), '%s'",
                _size_str(len(tar_data)),
                num_files,
                _size_str(file_size),
                func_name,
            )

            for num_threads in thread_counts:
                total_count, qps_mean, qps_lower_ci, qps_upper_ci = benchmark(
                    func, tar_data, num_iterations, num_threads, num_runs
                )

                margin = (qps_upper_ci - qps_lower_ci) / 2
                _LG.info(
                    "  Threads: %2d  QPS: %8.2f ± %.2f  (%.2f-%.2f, %d runs, %d files)",
                    num_threads,
                    qps_mean,
                    margin,
                    qps_lower_ci,
                    qps_upper_ci,
                    num_runs,
                    total_count,
                )

                results.append(
                    BenchmarkResult(
                        function_name=func_name,
                        tar_size=len(tar_data),
                        file_size=file_size,
                        num_files=num_files,
                        num_threads=num_threads,
                        num_iterations=num_iterations,
                        qps_mean=qps_mean,
                        qps_lower_ci=qps_lower_ci,
                        qps_upper_ci=qps_upper_ci,
                        total_files_processed=total_count,
                    )
                )

    return results


def plot_results(
    results: list[BenchmarkResult],
    output_path: str,
) -> None:
    """Plot benchmark results with 95% confidence intervals and save to file.

    Creates subplots for each file size tested, showing QPS vs. thread count
    with shaded confidence interval regions.

    Args:
        results: List of :py:class:`BenchmarkResult` containing all benchmark data.
        output_path: Path to save the plot (e.g., ``benchmark_results.png``).
    """
    import matplotlib.pyplot as plt

    # Extract unique file sizes and function names
    file_sizes = sorted({r.file_size for r in results})
    function_names = sorted({r.function_name for r in results})

    # Create subplots: at most 3 columns, multiple rows if needed
    num_sizes = len(file_sizes)
    max_cols = 3
    num_cols = min(num_sizes, max_cols)
    num_rows = (num_sizes + max_cols - 1) // max_cols  # Ceiling division

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))

    # Flatten axes array for easier indexing
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    elif num_rows == 1 or num_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, file_size in enumerate(file_sizes):
        ax = axes[idx]
        first_tar_size = 0
        first_thread_counts = None
        first_num_files = 0

        for func_name in function_names:
            # Filter results for this function and file size
            func_results = [
                r
                for r in results
                if r.function_name == func_name and r.file_size == file_size
            ]

            if not func_results:
                continue

            # Sort by thread count
            func_results.sort(key=lambda r: r.num_threads)

            thread_counts = [r.num_threads for r in func_results]
            qps_means = [r.qps_mean for r in func_results]
            qps_lower_cis = [r.qps_lower_ci for r in func_results]
            qps_upper_cis = [r.qps_upper_ci for r in func_results]

            if first_thread_counts is None:
                first_tar_size = func_results[0].tar_size
                first_thread_counts = thread_counts
                first_num_files = func_results[0].num_files

            ax.plot(
                thread_counts,
                qps_means,
                marker="o",
                label=func_name,
                linewidth=2,
            )

            # Add shaded confidence interval
            ax.fill_between(
                thread_counts,
                qps_lower_cis,
                qps_upper_cis,
                alpha=0.2,
            )

        ax.set_xlabel("Number of Threads", fontsize=11)
        ax.set_ylabel("Queries Per Second (QPS)", fontsize=11)
        ax.set_title(
            f"File Size: {_size_str(first_tar_size)} ({first_num_files} x {_size_str(file_size)})",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, None])
        if first_thread_counts:
            ax.set_xticks(first_thread_counts)
            ax.set_xticklabels(first_thread_counts)

    # Hide any unused subplots
    total_subplots = num_rows * num_cols
    for idx in range(num_sizes, total_subplots):
        axes[idx].set_visible(False)

    fig.suptitle(
        "TAR File Parsing Performance: SPDL vs Python tarfile\n(with 95% Confidence Intervals)",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    _LG.info("Plot saved to: %s", output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark iter_tarfile performance with multi-threading"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=100,
        help="Number of files in the test TAR archive",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations for each thread count",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_tarfile_results.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def _suffix(path: str) -> str:
    p1, p2 = os.path.splitext(path)
    return f"{p1}_2{p2}"


def main() -> None:
    """Main entry point for the benchmark script.

    Parses command-line arguments, runs benchmarks, and generates plots.
    """

    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname).1s]: %(message)s",
    )

    thread_counts = [1, 4, 8, 16, 32]
    file_sizes = [2**8, 2**12, 2**16, 2**20]

    _LG.info("Starting benchmark with configuration:")
    _LG.info("  Number of files: %d", args.num_files)
    _LG.info("  File sizes: %s bytes", file_sizes)
    _LG.info("  Iterations per thread count: %d", args.num_iterations)
    _LG.info("  Thread counts: %s", thread_counts)

    configs: list[tuple[str, Callable[[bytes], int]]] = [
        ("1. Python tarfile", process_tar_builtin),
        ("2. SPDL iter_tarfile (file-like)", process_tar_spdl_filelike),
        (
            "3. SPDL iter_tarfile (bytes w/ convert)",
            partial(process_tar_spdl, convert=True),
        ),
        (
            "4. SPDL iter_tarfile (bytes w/o convert)",
            partial(process_tar_spdl, convert=False),
        ),
    ]

    results = run_benchmark(
        configs,
        num_files=args.num_files,
        file_sizes=file_sizes,
        num_iterations=args.num_iterations,
        thread_counts=thread_counts,
    )

    plot_results(results, args.output)
    # plot another for easier view
    k = configs[-1][0]
    plot_results(
        [r for r in results if r.function_name != k],
        _suffix(args.output),
    )


if __name__ == "__main__":
    main()
