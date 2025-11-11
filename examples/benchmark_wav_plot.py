#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plot benchmark results from WAV loading benchmarks.

This script reads benchmark results from a CSV file and generates plots
comparing performance across different implementations.

**Example**

.. code-block:: shell

   $ python plot_wav_benchmark.py --input benchmark_results.csv --output plot.png
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

try:
    from examples.benchmark_utils import load_results_from_csv  # pyre-ignore[21]
    from examples.benchmark_wav import (  # pyre-ignore[21]
        BenchmarkConfig,
        DEFAULT_RESULT_PATH,
    )
except ImportError:
    from spdl.examples.benchmark_utils import load_results_from_csv
    from spdl.examples.benchmark_wav import BenchmarkConfig, DEFAULT_RESULT_PATH


def plot_benchmark_results(
    csv_file: str,
    output_file: str = "benchmark_results.png",
    filter_function: str | None = None,
) -> None:
    """Plot benchmark results and save to file.

    Args:
        csv_file: Path to CSV file containing benchmark data
        output_file: Output file path for the saved plot
        filter_function: Optional function name to filter out from the plot
    """
    matplotlib.use("Agg")  # Use non-interactive backend

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

    # Create ABI info string
    abi_info = " (free-threaded)" if free_threaded else ""

    # Group results by label
    labels_data: dict[str, list[tuple[int, float, float, float]]] = {}
    for result in results:
        label = f"{result.config.function_name} ({result.config.duration_seconds}s)"
        if label not in labels_data:
            labels_data[label] = []
        labels_data[label].append(
            (
                result.config.num_threads,
                result.qps,
                result.ci_lower,
                result.ci_upper,
            )
        )

    # Sort data by thread count
    for label in labels_data:
        labels_data[label].sort(key=lambda x: x[0])

    _, ax = plt.subplots(figsize=(12, 6))

    for label in sorted(labels_data.keys()):
        data = labels_data[label]
        thread_counts = [x[0] for x in data]
        qps_values = [x[1] for x in data]
        ci_lower_values = [x[2] for x in data]
        ci_upper_values = [x[3] for x in data]

        line = ax.plot(
            thread_counts,
            qps_values,
            marker="o",
            label=label,
            linewidth=2,
        )

        ax.fill_between(
            thread_counts,
            ci_lower_values,
            ci_upper_values,
            alpha=0.2,
            color=line[0].get_color(),
        )

    ax.set_xlabel("Number of Threads", fontsize=12)
    ax.set_ylabel("QPS (Queries Per Second)", fontsize=12)
    title = f"WAV Loading Performance Benchmark\nPython {python_version}{abi_info}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Function", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot WAV loading benchmark results from CSV"
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
    plot_benchmark_results(
        args.input, args.output or args.input.replace(".csv", ".png"), args.filter
    )


if __name__ == "__main__":
    main()
