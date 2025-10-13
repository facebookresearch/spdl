#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example demonstrating the use of :py:func:`spdl.pipeline.profile_pipeline`.

The :py:func:`~spdl.pipeline.profile_pipeline` function allows you to benchmark your pipeline
stages independently across different concurrency levels to identify optimal
performance settings. This is particularly useful when tuning pipeline performance
before deploying to production.

This example shows how to:

1. Create a simple pipeline with multiple processing stages
2. Use :py:func:`~spdl.pipeline.profile_pipeline` to benchmark each stage
3. Analyze the profiling results to identify performance bottlenecks
4. Use a custom callback to process results as they are generated
5. Visualize performance results with a plot

This example generates an figure like the following.

.. image:: ../../_static/data/profile_pipeline_example.png

If a function is time-consuming like networking or performing,
as long as the GIL is released, the performance improves with more threads.
(``"scalable_op"`` and ``"scalable_op2"``).

The function might be constrained by other factors such as CPU resource,
and it can hit the peak performance at some point. (``"scalable_op2"``)

If the function holds the GIL completely, the performance peaks at single
concurrency, and it degrades as more threads are added. (``"op_with_contention"``)
"""

import argparse
import logging
import time
from pathlib import Path

from spdl.pipeline import profile_pipeline, ProfileResult
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig

__all__ = [
    "parse_args",
    "main",
    "scalable_op",
    "scalable_op2",
    "op_with_contention",
    "create_pipeline",
    "print_profile_result",
    "plot_profile_results",
    "run_profiling_example",
]

# pyre-strict

_LG: logging.Logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=500,
        help="Number of inputs to use for profiling each stage",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Path to save the performance plot (e.g., profile_results.png)",
    )
    return parser.parse_args()


def scalable_op(x: int) -> int:
    """Simulate an operation which releases the GIL most of the time.

    Args:
        x: Input integer

    Returns:
        The input value multiplied by 2
    """
    time.sleep(0.01)
    return x * 2


def scalable_op2(x: int) -> int:
    """Simulate an operation which releases the GIL some time.

    Args:
        x: Input integer

    Returns:
        The input value plus 100
    """
    time.sleep(0.003)
    return x + 100


def op_with_contention(x: int) -> int:
    """Simulate an operation holds the GIL.

    Args:
        x: Input integer

    Returns:
        The input value squared
    """
    return x**2


def create_pipeline(num_sources: int = 1000) -> PipelineConfig[int, int]:
    """Create a pipeline configuration with multiple stages.

    Args:
        num_sources: Number of source items to generate

    Returns:
        Pipeline configuration with three processing stages
    """
    return PipelineConfig(
        src=SourceConfig(range(num_sources)),
        pipes=[
            Pipe(scalable_op),
            Pipe(scalable_op2),
            Pipe(op_with_contention),
        ],
        sink=SinkConfig(buffer_size=10),
    )


def print_profile_result(result: ProfileResult) -> None:
    """Print profiling result in a formatted way.

    This is a callback function that will be called after each stage is profiled.

    Args:
        result: Profiling result for a single stage
    """
    _LG.info("=" * 60)
    _LG.info("Stage: %s", result.name)
    _LG.info("-" * 60)

    for stat in result.stats:
        _LG.info(
            "Concurrency %2d: QPS=%8.2f, Occupancy=%5.1f%%",
            stat.concurrency,
            stat.qps,
            stat.occupancy_rate * 100,
        )

    best_stat = max(result.stats, key=lambda s: s.qps)
    _LG.info("-" * 60)
    _LG.info(
        "Best Performance: Concurrency=%d, QPS=%.2f",
        best_stat.concurrency,
        best_stat.qps,
    )
    _LG.info("=" * 60)


def plot_profile_results(
    results: list[ProfileResult], output_path: Path | None = None
) -> None:
    """Plot profiling results showing QPS vs concurrency for each stage.

    Args:
        results: List of profiling results for each pipeline stage
        output_path: Optional path to save the plot. If None, displays the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    all_concurrencies = set()
    for result in results:
        concurrencies = [stat.concurrency for stat in result.stats]
        qps_values = [stat.qps for stat in result.stats]
        plt.plot(concurrencies, qps_values, marker="o", linewidth=2, label=result.name)
        all_concurrencies.update(concurrencies)

    sorted_concurrencies = sorted(all_concurrencies, reverse=True)
    plt.xticks(sorted_concurrencies, [str(c) for c in sorted_concurrencies])

    plt.xlabel("Number of Threads (Concurrency)", fontsize=12)
    plt.ylabel("Throughput (QPS)", fontsize=12)
    plt.title(
        "Pipeline Stage Performance vs Concurrency", fontsize=14, fontweight="bold"
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        _LG.info("Plot saved to: %s", output_path)
    else:
        plt.show()


def run_profiling_example(num_inputs: int = 500) -> list[ProfileResult]:
    """Run the profiling example.

    Args:
        num_inputs: Number of inputs to use for profiling

    Returns:
        List of profiling results for each stage
    """
    _LG.info("Creating pipeline configuration...")
    pipeline_config = create_pipeline(num_sources=num_inputs * 2)

    _LG.info("Starting pipeline profiling with %d inputs...", num_inputs)
    _LG.info("This will benchmark each stage at different concurrency levels.")

    results = profile_pipeline(
        pipeline_config,
        num_inputs=num_inputs,
        callback=print_profile_result,
    )

    _LG.info("Profiling complete!")
    _LG.info("Total stages profiled: %d", len(results))

    return results


def main() -> None:
    """Main entry point demonstrating profile_pipeline usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    _LG.info("Profile Pipeline Example")
    _LG.info("=" * 60)

    results = run_profiling_example(num_inputs=args.num_inputs)

    _LG.info("\nSummary of Best Performance per Stage:")
    _LG.info("=" * 60)
    for result in results:
        best_stat = max(result.stats, key=lambda s: s.qps)
        _LG.info(
            "%-20s: Best at concurrency=%2d (QPS=%.2f)",
            result.name,
            best_stat.concurrency,
            best_stat.qps,
        )

    if args.plot_output or True:
        _LG.info("\nGenerating performance plot...")
        plot_profile_results(results, args.plot_output)


if __name__ == "__main__":
    main()
