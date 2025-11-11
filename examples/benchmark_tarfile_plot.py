#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Plot benchmark results from TAR file parsing benchmarks.

This script reads benchmark results from a CSV file and generates plots
comparing performance across different implementations.

**Example**

.. code-block:: shell

   $ python plot_tar_benchmark.py --input benchmark_tarfile_results.csv --output plot.png
"""

import argparse
import os

import matplotlib.pyplot as plt

try:
    from examples.benchmark_tarfile import (  # pyre-ignore[21]
        BenchmarkConfig,
        DEFAULT_RESULT_PATH,
    )
    from examples.benchmark_utils import load_results_from_csv  # pyre-ignore[21]
except ImportError:
    from spdl.examples.benchmark_tarfile import BenchmarkConfig, DEFAULT_RESULT_PATH
    from spdl.examples.benchmark_utils import load_results_from_csv


def _size_str(n: int) -> str:
    """Convert byte size to human-readable string.

    Args:
        n: Size in bytes.

    Returns:
        Human-readable size string.
    """
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024: .2f} kB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024): .2f} MB"
    return f"{n / (1024 * 1024 * 1024): .2f} GB"


def plot_results(
    csv_file: str,
    output_path: str,
    filter_function: str | None = None,
) -> None:
    """Plot benchmark results with 95% confidence intervals and save to file.

    Creates subplots for each file size tested, showing QPS vs. thread count
    with shaded confidence interval regions.

    Args:
        csv_file: Path to CSV file containing benchmark data
        output_path: Path to save the plot (e.g., ``benchmark_results.png``).
        filter_function: Optional function name to filter out from the plot.
    """
    # Load results from CSV
    results = load_results_from_csv(csv_file, BenchmarkConfig)

    # Filter out specific function if requested
    if filter_function:
        results = [r for r in results if r.config.function_name != filter_function]

    if not results:
        print("No results to plot")
        return

    # Extract Python info from the first result
    python_version = results[0].python_version
    free_threaded = results[0].free_threaded

    # Extract unique file sizes and function names
    file_sizes = sorted({r.config.file_size for r in results})
    function_names = sorted({r.config.function_name for r in results})

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
                if r.config.function_name == func_name
                and r.config.file_size == file_size
            ]

            if not func_results:
                continue

            # Sort by thread count
            func_results.sort(key=lambda r: r.config.num_threads)

            thread_counts = [r.config.num_threads for r in func_results]
            qps_means = [r.qps for r in func_results]
            qps_lower_cis = [r.ci_lower for r in func_results]
            qps_upper_cis = [r.ci_upper for r in func_results]

            if first_thread_counts is None:
                first_tar_size = func_results[0].config.tar_size
                first_thread_counts = thread_counts
                first_num_files = func_results[0].config.num_files

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
        title = (
            f"File Size: {_size_str(first_tar_size)} "
            f"({first_num_files} x {_size_str(file_size)})"
        )
        ax.set_title(title, fontsize=12)
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

    # Create main title with Python info
    abi_info = " (free-threaded)" if free_threaded else ""
    main_title = (
        f"TAR File Parsing Performance: SPDL vs Python tarfile\n"
        f"(with 95% Confidence Intervals) - Python {python_version}{abi_info}"
    )
    fig.suptitle(main_title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Plot iter_tarfile benchmark results from CSV"
    )
    parser.add_argument(
        "--input",
        type=lambda p: os.path.realpath(p),
        default=DEFAULT_RESULT_PATH,
        help="Input CSV file with benchmark results",
    )
    parser.add_argument(
        "--output",
        type=lambda p: os.path.realpath(p),
        help="Output path for the plot",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter out a specific function name from the plot",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the plot script.

    Parses command-line arguments and generates plots from CSV data.
    """
    args = _parse_args()

    print(f"Loading benchmark results from: {args.input}")
    plot_results(
        args.input, args.output or args.input.replace(".csv", ".png"), args.filter
    )


if __name__ == "__main__":
    main()
