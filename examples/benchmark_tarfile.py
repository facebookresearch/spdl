#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Benchmark script for :py:func:`spdl.io.iter_tarfile` function.

This script benchmarks the performance of :py:func:`~spdl.io.iter_tarfile` against
Python's built-in :py:mod:`tarfile` module using multi-threading.
Two types of inputs are tested for :py:func:`~spdl.io.iter_tarfile`.
Byte string and a file-like object returns byte string by chunk.

The benchmark:

1. Creates test tar archives with various numbers of files
2. Runs both implementations with different thread counts
3. Measures queries per second (QPS) for each configuration
4. Plots the results comparing the three implementations

**Example**

.. code-block:: shell

   $ numactl --membind 0 --cpubind 0 python benchmark_tarfile.py --output results.csv
   # Plot results
   $ python benchmark_tarfile_plot.py --input results.csv --output wav_benchmark_plot.png
   # Plot results without load_wav
   $ python benchmark_tarfile_plot.py --input results.csv --output wav_benchmark_plot_2.png \\
     --filter '4. SPDL iter_tarfile (bytes w/o convert)'

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
    "BenchmarkConfig",
    "BenchmarkResult",
    "create_test_tar",
    "iter_tarfile_builtin",
    "main",
    "process_tar_builtin",
    "process_tar_spdl",
    "process_tar_spdl_filelike",
]

import argparse
import io
import os
import tarfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

import spdl.io

try:
    from examples.benchmark_utils import (  # pyre-ignore[21]
        BenchmarkResult,
        BenchmarkRunner,
        ExecutorType,
        get_default_result_path,
        save_results_to_csv,
    )
except ImportError:
    from spdl.examples.benchmark_utils import (
        BenchmarkResult,
        BenchmarkRunner,
        ExecutorType,
        get_default_result_path,
        save_results_to_csv,
    )


DEFAULT_RESULT_PATH: str = get_default_result_path(__file__)


@dataclass
class BenchmarkConfig:
    """BenchmarkConfig()

    Configuration for a single TAR benchmark run."""

    function_name: str
    """Name of the function being tested"""

    tar_size: int
    """Total size of the TAR archive in bytes"""

    file_size: int
    """Size of each file in the TAR archive in bytes"""

    num_files: int
    """Number of files in the TAR archive"""

    num_threads: int
    """Number of concurrent threads"""

    num_iterations: int
    """Number of iterations per run"""

    total_files_processed: int
    """Total number of files processed across all iterations"""


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


def _size_str(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024: .2f} kB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024): .2f} MB"
    return f"{n / (1024 * 1024 * 1024): .2f} GB"


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


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
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
        type=lambda p: os.path.realpath(p),
        default=DEFAULT_RESULT_PATH,
        help="Output path for the results",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the benchmark script.

    Parses command-line arguments, runs benchmarks, and generates plots.
    """

    args = _parse_args()

    # Define explicit configuration lists
    thread_counts = [1, 4, 8, 16, 32]
    file_sizes = [2**8, 2**12, 2**16, 2**20]

    # Define benchmark function configurations
    # (function_name, function)
    benchmark_functions: list[tuple[str, Callable[[bytes], int]]] = [
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

    print("Starting benchmark with configuration:")
    print(f"  Number of files: {args.num_files}")
    print(f"  File sizes: {file_sizes} bytes")
    print(f"  Iterations per thread count: {args.num_iterations}")
    print(f"  Thread counts: {thread_counts}")

    results: list[BenchmarkResult[BenchmarkConfig]] = []
    num_runs = 5

    for num_threads in thread_counts:
        with BenchmarkRunner(
            executor_type=ExecutorType.THREAD,
            num_workers=num_threads,
            warmup_iterations=10 * num_threads,
        ) as runner:
            for file_size in file_sizes:
                tar_data = create_test_tar(args.num_files, file_size)
                for func_name, func in benchmark_functions:
                    print(
                        f"TAR size: {_size_str(len(tar_data))} "
                        f"({args.num_files} x {_size_str(file_size)}), "
                        f"'{func_name}', {num_threads} threads"
                    )

                    total_files_processed = args.num_files * args.num_iterations

                    config = BenchmarkConfig(
                        function_name=func_name,
                        tar_size=len(tar_data),
                        file_size=file_size,
                        num_files=args.num_files,
                        num_threads=num_threads,
                        num_iterations=args.num_iterations,
                        total_files_processed=total_files_processed,
                    )

                    result, _ = runner.run(
                        config,
                        partial(func, tar_data),
                        args.num_iterations,
                        num_runs=num_runs,
                    )

                    margin = (result.ci_upper - result.ci_lower) / 2
                    print(
                        f"  QPS: {result.qps:8.2f} ± {margin:.2f}  "
                        f"({result.ci_lower:.2f}-{result.ci_upper:.2f}, "
                        f"{num_runs} runs, {total_files_processed} files)"
                    )

                    results.append(result)

    # Save results to CSV
    save_results_to_csv(results, args.output)

    print(
        f"Benchmark complete. To generate plots, run: "
        f"python benchmark_tarfile_plot.py --input {args.output} "
        f"--output {args.output.replace('.csv', '.png')}"
    )


if __name__ == "__main__":
    main()
