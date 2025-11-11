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

   $ numactl --membind 0 --cpubind 0 python benchmark_wav.py --output wav_benchmark_results.csv
   # Plot results
   $ python benchmark_wav_plot.py --input wav_benchmark_results.csv --output wav_benchmark_plot.png
   # Plot results without load_wav
   $ python benchmark_wav_plot.py --input wav_benchmark_results.csv --output wav_benchmark_plot_2.png --filter '3. spdl.io.load_wav'

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
:ref:`spdl.io.load_npz <example-data-formats>`.)

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
    "BenchmarkConfig",
    "create_wav_data",
    "load_sf",
    "load_spdl_audio",
    "load_spdl_wav",
    "main",
]

import argparse
import io
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import spdl.io
from numpy.typing import NDArray

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
    """Configuration for a single WAV benchmark run.

    Combines both audio file parameters and benchmark execution parameters.

    Attributes:
        function_name: Name of the function being tested
        function: The actual function to benchmark
        sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bit depth per sample (16 or 32)
        duration_seconds: Duration of the audio file in seconds
        num_threads: Number of concurrent threads
        iterations: Number of iterations per run
        num_runs: Number of runs for statistical analysis
    """

    function_name: str
    function: Callable[[bytes], NDArray]
    sample_rate: int
    num_channels: int
    bits_per_sample: int
    duration_seconds: float
    num_threads: int
    iterations: int
    num_runs: int


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


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the benchmark script.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Benchmark WAV loading performance")
    parser.add_argument(
        "--output",
        type=lambda p: os.path.realpath(p),
        default=DEFAULT_RESULT_PATH,
        help="Output file path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run comprehensive benchmark suite for WAV loading performance.

    Benchmarks multiple configurations of audio files with different durations,
    comparing spdl.io.load_wav, spdl.io.load_audio, and soundfile libraries
    across various thread counts (1, 2, 4, 8, 16).
    """
    args = _parse_args()

    # Define audio configurations to test
    audio_configs = [
        # (sample_rate, num_channels, bits_per_sample, duration_seconds)
        # (8000, 1, 16, 1.0),  # Low quality mono
        # (16000, 1, 16, 1.0),  # Speech quality mono
        # (48000, 2, 16, 1.0),  # High quality stereo
        # (48000, 8, 16, 1.0),  # Multi-channel audio
        (44100, 2, 16, 1.0),  # CD quality stereo
        (44100, 2, 16, 10.0),  #
        (44100, 2, 16, 60.0),  #
        # (44100, 2, 24, 1.0),  # 24-bit audio
    ]

    thread_counts = [1, 2, 4, 8, 16]

    # Define benchmark function configurations
    # (function_name, function, iterations_multiplier, num_runs)
    benchmark_functions = [
        ("3. spdl.io.load_wav", load_spdl_wav, 100, 100),  # Fast but unstable
        ("2. spdl.io.load_audio", load_spdl_audio, 10, 5),  # Slower but stable
        ("1. soundfile", load_sf, 10, 5),  # Slower but stable
    ]

    results: list[BenchmarkResult[BenchmarkConfig]] = []

    for sample_rate, num_channels, bits_per_sample, duration_seconds in audio_configs:
        # Create WAV data for this audio configuration
        wav_data, ref = create_wav_data(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            duration_seconds=duration_seconds,
        )

        print(
            f"\n{sample_rate}Hz, {num_channels}ch, {bits_per_sample}bit, {duration_seconds}s"
        )
        print(
            f"Threads,"
            f"SPDL WAV QPS ({duration_seconds} sec),CI Lower,CI Upper,"
            f"SPDL Audio QPS ({duration_seconds} sec),CI Lower,CI Upper,"
            f"soundfile QPS ({duration_seconds} sec),CI Lower,CI Upper"
        )

        for num_threads in thread_counts:
            thread_results: list[BenchmarkResult[BenchmarkConfig]] = []

            with BenchmarkRunner(
                executor_type=ExecutorType.THREAD,
                num_workers=num_threads,
            ) as runner:
                for (
                    function_name,
                    function,
                    iterations_multiplier,
                    num_runs,
                ) in benchmark_functions:
                    config = BenchmarkConfig(
                        function_name=function_name,
                        function=function,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        bits_per_sample=bits_per_sample,
                        duration_seconds=duration_seconds,
                        num_threads=num_threads,
                        iterations=iterations_multiplier * num_threads,
                        num_runs=num_runs,
                    )

                    result, output = runner.run(
                        config,
                        lambda fn=function, data=wav_data: fn(data),
                        config.iterations,
                        num_runs=config.num_runs,
                    )

                    output_to_validate = output
                    if output_to_validate.ndim == 1:
                        output_to_validate = output_to_validate[:, None]
                    np.testing.assert_array_equal(output_to_validate, ref)

                    thread_results.append(result)
                    results.append(result)

            # Print results for this thread count (all 3 benchmarks)
            spdl_wav_result = thread_results[0]
            spdl_audio_result = thread_results[1]
            soundfile_result = thread_results[2]
            print(
                f"{num_threads},"
                f"{spdl_wav_result.qps:.2f},{spdl_wav_result.ci_lower:.2f},{spdl_wav_result.ci_upper:.2f},"
                f"{spdl_audio_result.qps:.2f},{spdl_audio_result.ci_lower:.2f},{spdl_audio_result.ci_upper:.2f},"
                f"{soundfile_result.qps:.2f},{soundfile_result.ci_lower:.2f},{soundfile_result.ci_upper:.2f}"
            )

    save_results_to_csv(results, args.output)
    print(
        f"\nBenchmark complete. To generate plots, run:\n"
        f"python benchmark_wav_plot.py --input {args.output} "
        f"--output {args.output.replace('.csv', '.png')}"
    )


if __name__ == "__main__":
    main()
