#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plot benchmark results from data format loading benchmarks.

This script reads benchmark results from a CSV file and generates plots
comparing performance across different data formats and implementations.

**Example**

.. code-block:: shell

   $ python benchmark_numpy_plot.py --input results.csv --output plot.png
"""

import argparse
import os

import matplotlib.pyplot as plt

try:
    from examples.benchmark_numpy import (  # pyre-ignore[21]
        BenchmarkConfig,
        DEFAULT_RESULT_PATH,
    )
    from examples.benchmark_utils import load_results_from_csv  # pyre-ignore[21]
except ImportError:
    from spdl.examples.benchmark_numpy import BenchmarkConfig, DEFAULT_RESULT_PATH
    from spdl.examples.benchmark_utils import load_results_from_csv


def plot_results(
    csv_file: str,
    output_path: str,
) -> None:
    """Plot benchmark results and save to file.

    Creates subplots for each executor type (thread vs process), showing QPS vs. worker count
    for different data formats and implementations.

    Args:
        csv_file: Path to CSV file containing benchmark data
        output_path: Path to save the plot (e.g., ``benchmark_results.png``).
    """
    # Load results from CSV
    results = load_results_from_csv(csv_file, BenchmarkConfig)

    if not results:
        print("No results to plot")
        return

    # Extract Python info from the first result
    python_version = results[0].python_version
    free_threaded = results[0].free_threaded

    # Extract unique executor types
    executor_types = sorted({r.executor_type for r in results})

    fig, axes = plt.subplots(
        1, len(executor_types), figsize=(12 * len(executor_types) // 2, 6)
    )

    if len(executor_types) == 1:
        axes = [axes]

    for idx, executor_type in enumerate(executor_types):
        ax = axes[idx]

        # Filter results for this executor type
        executor_results = [r for r in results if r.executor_type == executor_type]

        # Group results by label
        labels_data: dict[str, list[tuple[int, float]]] = {}
        for result in executor_results:
            label = (
                f"{result.config.data_format} "
                f"({result.config.impl}, compressed={result.config.compressed})"
            )
            if label not in labels_data:
                labels_data[label] = []
            labels_data[label].append((result.config.num_workers, result.qps))

        # Sort data by worker count and plot
        for label in sorted(labels_data.keys()):
            data = labels_data[label]
            data.sort(key=lambda x: x[0])
            worker_counts = [x[0] for x in data]
            qps_values = [x[1] for x in data]

            ax.plot(
                worker_counts,
                qps_values,
                marker="o",
                label=label,
                linewidth=2,
            )

        ax.set_xlabel("Number of Workers", fontsize=11)
        ax.set_ylabel("Queries Per Second (QPS)", fontsize=11)
        ax.set_title(f"Executor: {executor_type}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, None])

    abi_info = " (free-threaded)" if free_threaded else ""
    main_title = (
        f"Data Format Loading Performance Benchmark\nPython {python_version}{abi_info}"
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
        description="Plot data format benchmark results from CSV"
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

    return parser.parse_args()


def main() -> None:
    """Main entry point for the plot script.

    Parses command-line arguments and generates plots from CSV data.
    """
    args = _parse_args()

    print(f"Loading benchmark results from: {args.input}")
    plot_results(args.input, args.output or args.input.replace(".csv", ".png"))


if __name__ == "__main__":
    main()
