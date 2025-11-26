#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plot benchmark results from video decoding benchmarks.

This script reads benchmark results from a CSV file and generates plots
comparing performance across different video resolutions and concurrency settings.

**Example**

.. code-block:: shell

   $ python benchmark_video_plot.py --input benchmark_results.csv --output plot.png
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

try:
    from examples.benchmark_utils import load_results_from_csv  # pyre-ignore[21]
    from examples.benchmark_video import (  # pyre-ignore[21]
        BenchmarkConfig,
        DEFAULT_RESULT_PATH,
    )
except ImportError:
    from spdl.examples.benchmark_utils import load_results_from_csv
    from spdl.examples.benchmark_video import BenchmarkConfig, DEFAULT_RESULT_PATH


def plot_benchmark_results(
    csv_file: str,
    output_file: str = "benchmark_results.png",
) -> None:
    """Plot benchmark results and save to file.

    Args:
        csv_file: Path to CSV file containing benchmark data
        output_file: Output file path for the saved plot
    """
    matplotlib.use("Agg")

    results = load_results_from_csv(csv_file, BenchmarkConfig)

    if not results:
        print("No results to plot")
        return

    python_version = results[0].python_version
    free_threaded = results[0].free_threaded

    abi_info = " (free-threaded)" if free_threaded else ""

    resolutions = sorted(set(r.config.resolution for r in results))

    fig, axes = plt.subplots(2, len(resolutions), figsize=(6 * len(resolutions), 10))
    if len(resolutions) == 1:
        axes = axes.reshape(2, 1)

    for idx, resolution in enumerate(resolutions):
        ax_qps = axes[0, idx]
        ax_cpu = axes[1, idx]
        resolution_results = [r for r in results if r.config.resolution == resolution]

        decoder_threads_set = sorted(
            set(r.config.decoder_threads for r in resolution_results)
        )

        for decoder_threads in decoder_threads_set:
            thread_results = [
                r
                for r in resolution_results
                if r.config.decoder_threads == decoder_threads
            ]
            thread_results.sort(key=lambda x: x.config.num_workers)

            worker_counts = [r.config.num_workers for r in thread_results]
            qps_values = [r.qps for r in thread_results]
            ci_lower_values = [r.ci_lower for r in thread_results]
            ci_upper_values = [r.ci_upper for r in thread_results]
            cpu_values = [r.cpu_percent for r in thread_results]

            line = ax_qps.plot(
                worker_counts,
                qps_values,
                marker="o",
                label=f"{decoder_threads} decoder threads",
                linewidth=2,
            )

            ax_qps.fill_between(
                worker_counts,
                ci_lower_values,
                ci_upper_values,
                alpha=0.2,
                color=line[0].get_color(),
            )

            ax_cpu.plot(
                worker_counts,
                cpu_values,
                marker="s",
                label=f"{decoder_threads} decoder threads",
                linewidth=2,
                color=line[0].get_color(),
            )

        ax_qps.set_xlabel("Number of Workers", fontsize=11)
        ax_qps.set_ylabel("QPS (Videos Per Second)", fontsize=11)
        ax_qps.set_title(f"{resolution} Resolution", fontsize=12, fontweight="bold")
        ax_qps.legend(title="Decoder Threads", fontsize=9)
        ax_qps.grid(True, alpha=0.3)

        ax_cpu.set_xlabel("Number of Workers", fontsize=11)
        ax_cpu.set_ylabel("CPU Utilization (%)", fontsize=11)
        ax_cpu.set_title(f"{resolution} CPU Usage", fontsize=12, fontweight="bold")
        ax_cpu.legend(title="Decoder Threads", fontsize=9)
        ax_cpu.grid(True, alpha=0.3)

    fig.suptitle(
        f"Video Decoding Performance Benchmark\nPython {python_version}{abi_info}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot video decoding benchmark results from CSV"
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
    """Main entry point for the plotting script."""
    args = _parse_args()

    output_file = args.output
    if output_file is None:
        output_file = args.input.replace(".csv", ".png")

    plot_benchmark_results(args.input, output_file)


if __name__ == "__main__":
    main()
