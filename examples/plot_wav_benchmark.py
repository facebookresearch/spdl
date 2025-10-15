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
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    matplotlib.use("Agg")  # Use non-interactive backend

    df = pd.read_csv(csv_file)

    # Filter out specific function if requested
    if filter_function:
        df = df[df["function_name"] != filter_function]

    df["label"] = df["function_name"] + " (" + df["duration_seconds"].astype(str) + "s)"

    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(12, 6))
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label].sort_values("num_threads")
        line = ax.plot(
            subset["num_threads"],
            subset["qps"],
            marker="o",
            label=label,
            linewidth=2,
        )

        ax.fill_between(
            subset["num_threads"],
            subset["ci_lower"],
            subset["ci_upper"],
            alpha=0.2,
            color=line[0].get_color(),
        )

    ax.set_xlabel("Number of Threads", fontsize=12)
    ax.set_ylabel("QPS (Queries Per Second)", fontsize=12)
    ax.set_title("WAV Loading Performance Benchmark", fontsize=14, fontweight="bold")
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
        type=str,
        required=True,
        help="Input CSV file with benchmark results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_wav_plot.png",
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
    plot_benchmark_results(args.input, args.output, args.filter)


if __name__ == "__main__":
    main()
