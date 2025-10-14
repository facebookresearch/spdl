# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""This example measuers the performance of loading WAV audio.

It compares three different approaches for loading WAV files:

- :py:func:`spdl.io.load_wav`: Fast native WAV parser optimized for simple PCM formats
- :py:func:`spdl.io.load_audio`: General-purpose audio loader using FFmpeg backend
- ``soundfile`` (``libsndfile``): Popular third-party audio I/O library

The benchmark suite evaluates performance across multiple dimensions:

- Various audio configurations (sample rates, channels, bit depths, durations)
- Different thread counts (1, 2, 4, 8, 16) to measure parallel scaling
- Statistical analysis with 95% confidence intervals using Student's t-distribution
- Queries per second (QPS) as the primary performance metric

**Example**

.. code-block:: shell

   $ python benchmark_wav.py --plot --output results.png

**Result**

The following plot shows the QPS (measured by the number of files processed) of each
functions with different audio durations.

.. image:: ../../_static/data/example-benchmark-wav.webp


The :py:func:`spdl.io.load_wav` is a lot faster than the others, because all it
does is reinterpret the input byte string as array.
It shows the same performance for audio with longer duration.

And since parsing WAV is instant, the spdl.io.load_wav function spends more time on
creation of NumPy Array.
It needs to acquire the GIL, thus the performance does not scale in multi-threading.
(This performance pattern of this function is pretty same as the
:ref:`spdl.io.load_npz <data-format>`.)

The following is the same plot without ``load_wav``.

.. image:: ../../_static/data/example-benchmark-wav-2.webp

``libsoundfile`` has to process data iteratively (using ``io.BytesIO``) because
it does not support directly loading from byte string, so it takes longer to process
longer audio data.
The performance trend (single thread being the fastest) suggests that
it does not release the GIL majority of the time.

The :py:func:`spdl.io.load_audio` function (the generic FFmpeg-based implementation) does
a lot of work so its overall performance is not as good,
but it scales in multi-threading as it releases the GIL almost entirely.
"""

__all__ = [
    "BenchmarkResult",
    "BenchmarkConfig",
    "create_wav_data",
    "load_sf",
    "load_spdl_audio",
    "load_spdl_wav",
    "benchmark",
    "run_benchmark_suite",
    "plot_benchmark_results",
    "main",
]

import argparse
import io
import os.path
import time
from collections.abc import Callable
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.io.wavfile
import scipy.stats
import soundfile as sf
import spdl.io
from numpy.typing import NDArray


def create_wav_data(
    sample_rate: int = 44100,
    num_channels: int = 2,
    bits_per_sample: int = 16,
    duration_seconds: float = 1.0,
) -> tuple[bytes, NDArray]:
    """Create a WAV file in memory for benchmarking.

    Args:
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample (16 or 32)
        duration_seconds: Duration of audio in seconds

    Returns:
        Tuple of (WAV file as bytes, audio samples array)
    """
    num_samples = int(sample_rate * duration_seconds)

    dtype_map = {
        16: np.int16,
        32: np.int32,
    }
    dtype = dtype_map[bits_per_sample]
    max_amplitude = 32767 if bits_per_sample == 16 else 2147483647

    t = np.linspace(0, duration_seconds, num_samples)
    frequencies = 440.0 + np.arange(num_channels) * 110.0
    sine_waves = np.sin(2 * np.pi * frequencies[:, np.newaxis] * t)
    samples = (sine_waves.T * max_amplitude).astype(dtype)

    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, sample_rate, samples)
    wav_data = wav_buffer.getvalue()

    return wav_data, samples


def load_sf(wav_data: bytes) -> NDArray:
    """Load WAV data using soundfile library.

    Args:
        wav_data: WAV file data as bytes

    Returns:
        Audio samples array as int16 numpy array
    """
    audio_file = io.BytesIO(wav_data)
    data, _ = sf.read(audio_file, dtype="int16")
    return data


def load_spdl_audio(wav_data: bytes) -> NDArray:
    """Load WAV data using :py:func:`spdl.io.load_audio` function.

    Args:
        wav_data: WAV file data as bytes

    Returns:
        Audio samples array as numpy array
    """
    return spdl.io.to_numpy(spdl.io.load_audio(wav_data, filter_desc=None))


def load_spdl_wav(wav_data: bytes) -> NDArray:
    """Load WAV data using :py:func:`spdl.io.load_wav` function.

    Args:
        wav_data: WAV file data as bytes

    Returns:
        Audio samples array as numpy array
    """
    return spdl.io.to_numpy(spdl.io.load_wav(wav_data))


@dataclass(frozen=True)
class BenchmarkResult:
    """Results from a single benchmark run."""

    duration: float
    qps: float
    ci_lower: float
    ci_upper: float
    num_threads: int
    function_name: str
    duration_seconds: float


def benchmark(
    name: str,
    func: Callable[[], NDArray],
    iterations: int,
    num_threads: int,
    num_sets: int,
    duration_seconds: float,
) -> tuple[BenchmarkResult, NDArray]:
    """Benchmark a function using multiple threads and calculate statistics.

    Executes a warmup phase followed by multiple benchmark sets to compute
    performance metrics including mean queries per second (QPS) and 95%
    confidence intervals using Student's t-distribution.

    Args:
        name: Descriptive name for the benchmark (used in results)
        func: Callable function to benchmark (takes no args, returns NDArray)
        iterations: Total number of function calls per benchmark set
        num_threads: Number of concurrent threads for parallel execution
        num_sets: Number of independent benchmark sets for confidence interval
        duration_seconds: Duration of audio file being processed (for metadata)

    Returns:
        Tuple containing:
            - BenchmarkResult with timing statistics, QPS, confidence intervals
            - Output NDArray from the last function execution
    """

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Warmup
        futures = [executor.submit(func) for _ in range(num_threads * 30)]
        for future in as_completed(futures):
            output = future.result()

        # Run multiple sets for confidence interval
        qps_samples = []
        for _ in range(num_sets):
            t0 = time.perf_counter()
            futures = [executor.submit(func) for _ in range(iterations)]
            for future in as_completed(futures):
                output = future.result()
            elapsed = time.perf_counter() - t0
            qps_samples.append(iterations / elapsed)

    # Calculate mean and 95% confidence interval
    qps_mean = np.mean(qps_samples)
    qps_std = np.std(qps_samples, ddof=1)
    confidence_level = 0.95
    degrees_freedom = num_sets - 1
    confidence_interval = scipy.stats.t.interval(
        confidence_level,
        degrees_freedom,
        loc=qps_mean,
        scale=qps_std / np.sqrt(num_sets),
    )

    duration = 1.0 / qps_mean
    result = BenchmarkResult(
        duration=duration,
        qps=qps_mean,
        ci_lower=float(confidence_interval[0]),
        ci_upper=float(confidence_interval[1]),
        num_threads=num_threads,
        function_name=name,
        duration_seconds=duration_seconds,
    )
    return result, output  # pyre-ignore[61]


def run_benchmark_suite(
    wav_data: bytes,
    ref: NDArray,
    num_threads: int,
    duration_seconds: float,
) -> tuple[BenchmarkResult, BenchmarkResult, BenchmarkResult]:
    """Run benchmarks for both libraries with given parameters.

    Args:
        wav_data: WAV file data as bytes
        ref: Reference audio array for validation
        num_threads: Number of threads (use 1 for single-threaded)
        duration_seconds: Duration of audio in seconds

    Returns:
        Tuple of (spdl_wav_result, spdl_audio_result, soundfile_result)
    """
    # load_wav is fast but the performance is unstable, so we need to run more
    iterations = 100 * num_threads
    num_sets = 100

    spdl_wav_result, output = benchmark(
        name="spdl.io.load_wav",
        func=lambda: load_spdl_wav(wav_data),
        iterations=iterations,
        num_threads=num_threads,
        num_sets=num_sets,
        duration_seconds=duration_seconds,
    )
    np.testing.assert_array_equal(output, ref)

    # others are slow but the performance is stable.
    iterations = 10 * num_threads
    num_sets = 5

    spdl_audio_result, output = benchmark(
        name="spdl.io.load_audio",
        func=lambda: load_spdl_audio(wav_data),
        iterations=iterations,
        num_threads=num_threads,
        num_sets=num_sets,
        duration_seconds=duration_seconds,
    )
    np.testing.assert_array_equal(output, ref)
    soundfile_result, output = benchmark(
        name="soundfile",
        func=lambda: load_sf(wav_data),
        iterations=iterations,
        num_threads=num_threads,
        num_sets=num_sets,
        duration_seconds=duration_seconds,
    )
    if output.ndim == 1:
        output = output[:, None]
    np.testing.assert_array_equal(output, ref)

    return spdl_wav_result, spdl_audio_result, soundfile_result


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for audio file parameters used in benchmarking.

    Attributes:
        sample_rate: Audio sample rate in Hz (e.g., 44100 for CD quality)
        num_channels: Number of audio channels (1=mono, 2=stereo, etc.)
        bits_per_sample: Bit depth per sample (16 or 32)
        duration_seconds: Duration of the audio file in seconds
    """

    sample_rate: int
    num_channels: int
    bits_per_sample: int
    duration_seconds: float


def plot_benchmark_results(
    results: list[BenchmarkResult], output_file: str = "benchmark_results.png"
) -> None:
    """Plot benchmark results and save to file.

    Args:
        results: List of BenchmarkResult objects containing benchmark data
        output_file: Output file path for the saved plot
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    matplotlib.use("Agg")  # Use non-interactive backend

    data = [
        {
            "num_threads": r.num_threads,
            "qps": r.qps,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "function": r.function_name,
            "duration": f"{r.duration_seconds}s",
        }
        for r in results
    ]
    df = pd.DataFrame(data)

    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(12, 6))
    df["label"] = df["function"] + " (" + df["duration"] + ")"
    for label in df["label"].unique():
        subset = df[df["label"] == label].sort_values("num_threads")
        line = ax.plot(
            subset["num_threads"],
            subset["qps"],
            marker="o",
            label=label,
            linewidth=2,
        )

        # Add confidence interval as shaded region
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
    """Parse command line arguments for the benchmark script.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Benchmark WAV loading performance")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save a plot of the benchmark results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.png",
        help="Output file path for the plot (default: benchmark_results.png)",
    )
    return parser.parse_args()


def _suffix(path: str) -> str:
    p1, p2 = os.path.splitext(path)
    return f"{p1}_2{p2}"


def main() -> None:
    """Run comprehensive benchmark suite for WAV loading performance.

    Benchmarks multiple configurations of audio files with different durations,
    comparing spdl.io.load_wav, spdl.io.load_audio, and soundfile libraries
    across various thread counts (1, 2, 4, 8, 16).
    """
    args = _parse_args()

    benchmark_configs = [
        # (sample_rate, num_channels, bits_per_sample, duration_seconds, iterations)
        # BenchmarkConfig(8000, 1, 16, 1.0),  # Low quality mono
        # BenchmarkConfig(16000, 1, 16, 1.0),  # Speech quality mono
        # BenchmarkConfig(48000, 2, 16, 1.0),  # High quality stereo
        # BenchmarkConfig(48000, 8, 16, 1.0),  # Multi-channel audio
        BenchmarkConfig(44100, 2, 16, 1.0),  # CD quality stereo
        BenchmarkConfig(44100, 2, 16, 10.0),  #
        BenchmarkConfig(44100, 2, 16, 60.0),  #
        # (44100, 2, 24, 1.0, 100),  # 24-bit audio
    ]

    results: list[BenchmarkResult] = []

    for cfg in benchmark_configs:
        print(cfg)
        wav_data, ref = create_wav_data(
            sample_rate=cfg.sample_rate,
            num_channels=cfg.num_channels,
            bits_per_sample=cfg.bits_per_sample,
            duration_seconds=cfg.duration_seconds,
        )
        print(
            f"Threads,"
            f"SPDL WAV QPS ({cfg.duration_seconds} sec),CI Lower, CI Upper,"
            f"SPDL Audio QPS ({cfg.duration_seconds} sec),CI Lower, CI Upper,"
            f"soundfile QPS ({cfg.duration_seconds} sec),CI Lower, CI Upper"
        )
        for num_threads in [1, 2, 4, 8, 16]:
            spdl_wav_result, spdl_audio_result, soundfile_result = run_benchmark_suite(
                wav_data,
                ref,
                num_threads=num_threads,
                duration_seconds=cfg.duration_seconds,
            )
            results.extend([spdl_wav_result, spdl_audio_result, soundfile_result])
            print(
                f"{num_threads},"
                f"{spdl_wav_result.qps:.2f},{spdl_wav_result.ci_lower:.2f},{spdl_wav_result.ci_upper:.2f},"
                f"{spdl_audio_result.qps:.2f},{spdl_audio_result.ci_lower:.2f},{spdl_audio_result.ci_upper:.2f},"
                f"{soundfile_result.qps:.2f},{soundfile_result.ci_lower:.2f},{soundfile_result.ci_upper:.2f}"
            )

    if args.plot:
        plot_benchmark_results(results, args.output)
        k = "spdl.io.load_wav"
        plot_benchmark_results(
            [r for r in results if r.function_name != k],
            _suffix(args.output),
        )


if __name__ == "__main__":
    main()
