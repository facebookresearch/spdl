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
import logging
from dataclasses import dataclass

_LG: logging.Logger = logging.getLogger(__name__)


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
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_file)

    # Extract Python info from the first row (all rows should have same Python info)
    python_version = (
        df["python_version"].iloc[0] if "python_version" in df.columns else "unknown"
    )
    free_threaded = (
        df["free_threaded"].iloc[0] if "free_threaded" in df.columns else False
    )

    results = [
        BenchmarkResult(
            function_name=row["function_name"],
            tar_size=row["tar_size"],
            file_size=row["file_size"],
            num_files=row["num_files"],
            num_threads=row["num_threads"],
            num_iterations=row["num_iterations"],
            qps_mean=row["qps_mean"],
            qps_lower_ci=row["qps_lower_ci"],
            qps_upper_ci=row["qps_upper_ci"],
            total_files_processed=row["total_files_processed"],
        )
        for _, row in df.iterrows()
    ]

    # Filter out specific function if requested
    if filter_function:
        results = [r for r in results if r.function_name != filter_function]

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
    _LG.info("Plot saved to: %s", output_path)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot TAR file benchmark results from CSV"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with benchmark results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_tarfile_plot.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter out a specific function name from the plot",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the plot script.

    Parses command-line arguments and generates plots from CSV data.
    """
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname).1s]: %(message)s",
    )

    _LG.info("Loading benchmark results from: %s", args.input)
    plot_results(args.input, args.output, args.filter)


if __name__ == "__main__":
    main()
