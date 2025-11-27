# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""This example measures the performance of video decoding with different concurrency settings.

It benchmarks :py:func:`spdl.io.load_video` across multiple dimensions:

- Various video resolutions (SD: 640x480, HD: 1920x1080, 4K: 3840x2160)
- Different worker thread counts (1, 2, 4, 8) for parallel video processing
- Different decoder thread counts (1, 2, 4) for FFmpeg's internal threading

The benchmark evaluates how throughput changes with:

1. **Worker-level concurrency**: Number of videos processed concurrently (via ``num_workers``)
2. **Decoder-level concurrency**: FFmpeg's internal threading (via ``decoder_options={"threads": "X"}``)

**Example**

.. code-block:: shell

   $ numactl --membind 0 --cpubind 0 python benchmark_video.py --output video_benchmark_results.csv
   # Plot results
   $ python benchmark_video_plot.py --input video_benchmark_results.csv --output video_benchmark_plot.png

**Result**

In many cases, when decoding H264 videos, using 2 threads give a good performance.

.. image:: ../../_static/data/example-benchmark-video.png

"""

__all__ = [
    "BenchmarkConfig",
    "create_video_data",
    "load_video_with_config",
    "main",
]

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass

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


@dataclass(frozen=True)
class BenchmarkConfig:
    """BenchmarkConfig()

    Configuration for a single video decoding benchmark run."""

    resolution: str
    """Video resolution label (e.g., "SD", "HD", "4K")"""

    width: int
    """Video width in pixels"""

    height: int
    """Video height in pixels"""

    duration_seconds: float
    """Duration of the video in seconds"""

    num_workers: int
    """Number of concurrent worker threads"""

    decoder_threads: int
    """Number of FFmpeg decoder threads"""

    iterations: int
    """Number of iterations per run"""

    num_runs: int
    """Number of runs for statistical analysis"""


def create_video_data(
    width: int = 1920,
    height: int = 1080,
    duration_seconds: float = 5.0,
    fps: int = 30,
) -> bytes:
    """Create a mock H.264 video file in memory for benchmarking.

    Args:
        width: Video width in pixels
        height: Video height in pixels
        duration_seconds: Duration of video in seconds
        fps: Frames per second

    Returns:
        Video file as bytes (H.264 encoded in MP4 container)
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
        output_path = tmp_file.name

        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_seconds}:size={width}x{height}:rate={fps}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-y",
            output_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with open(output_path, "rb") as f:
            video_data = f.read()

        return video_data


def load_video_with_config(
    video_data: bytes, decoder_threads: int
) -> spdl.io.CPUBuffer:
    """Load video data using spdl.io.load_video with specified decoder threads.

    Args:
        video_data: Video file data as bytes
        decoder_threads: Number of threads for FFmpeg decoder

    Returns:
        Decoded video frames as CPUBuffer
    """
    decode_config = spdl.io.decode_config(
        decoder_options={"threads": str(decoder_threads)}
    )
    return spdl.io.load_video(video_data, decode_config=decode_config)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the benchmark script.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Benchmark video decoding performance")
    parser.add_argument(
        "--output",
        type=lambda p: os.path.realpath(p),
        default=DEFAULT_RESULT_PATH,
        help="Output file path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run comprehensive benchmark suite for video decoding performance.

    Benchmarks video decoding across different resolutions (SD, HD, 4K),
    worker thread counts (1, 2, 4, 8), and decoder thread counts (1, 2, 4).
    """
    args = _parse_args()

    video_configs = [
        ("SD", 640, 480, 5.0),
        ("HD", 1920, 1080, 5.0),
        ("4K", 3840, 2160, 5.0),
    ]

    worker_counts = [1, 2, 4, 8]
    decoder_thread_counts = [1, 2, 4]

    results: list[BenchmarkResult[BenchmarkConfig]] = []

    for resolution, width, height, duration in video_configs:
        print(f"\nCreating {resolution} video ({width}x{height}, {duration}s)...")
        video_data = create_video_data(
            width=width, height=height, duration_seconds=duration
        )
        print(f"Video size: {len(video_data) / 1024 / 1024:.2f} MB")

        print(f"\n{resolution} ({width}x{height})")
        print("Workers,Decoder Threads,QPS,CI Lower,CI Upper,CPU %")

        for num_workers in worker_counts:
            with BenchmarkRunner(
                executor_type=ExecutorType.THREAD,
                num_workers=num_workers,
            ) as runner:
                for decoder_threads in decoder_thread_counts:
                    config = BenchmarkConfig(
                        resolution=resolution,
                        width=width,
                        height=height,
                        duration_seconds=duration,
                        num_workers=num_workers,
                        decoder_threads=decoder_threads,
                        iterations=num_workers * 2,
                        num_runs=5,
                    )

                    result, output = runner.run(
                        config,
                        lambda data=video_data,
                        threads=decoder_threads: load_video_with_config(data, threads),
                        config.iterations,
                        num_runs=config.num_runs,
                    )

                    results.append(result)

                    print(
                        f"{num_workers},{decoder_threads},"
                        f"{result.qps:.2f},{result.ci_lower:.2f},{result.ci_upper:.2f},"
                        f"{result.cpu_percent:.1f}"
                    )

    save_results_to_csv(results, args.output)
    print(
        f"\nBenchmark complete. To generate plots, run:\n"
        f"python benchmark_video_plot.py --input {args.output} "
        f"--output {args.output.replace('.csv', '.png')}"
    )


if __name__ == "__main__":
    main()
