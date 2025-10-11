# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Benchmark script comparing WAV loading performance between libsoundfile and spdl.io.load_wav."""

import io
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

    # Generate audio samples with sine wave pattern
    dtype_map = {
        16: np.int16,
        32: np.int32,
    }
    dtype = dtype_map[bits_per_sample]

    # Create audio samples with a simple sine wave pattern
    samples = np.zeros((num_samples, num_channels), dtype=dtype)
    for channel_idx in range(num_channels):
        frequency = 440.0 + (channel_idx * 110.0)  # A4 and harmonics
        t = np.linspace(0, duration_seconds, num_samples)
        wave = np.sin(2 * np.pi * frequency * t)

        if bits_per_sample == 16:
            samples[:, channel_idx] = (wave * 32767).astype(dtype)
        elif bits_per_sample == 32:
            samples[:, channel_idx] = (wave * 2147483647).astype(dtype)

    # Use scipy to write WAV file to memory buffer
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, sample_rate, samples)
    wav_data = wav_buffer.getvalue()

    return wav_data, samples


def load_sf(wav_data: bytes) -> NDArray:
    """Load WAV data using soundfile."""

    audio_file = io.BytesIO(wav_data)
    data, _ = sf.read(audio_file, dtype="int16")
    return data


def load_sp(wav_data: bytes) -> NDArray:
    return spdl.io.to_numpy(spdl.io.load_audio(wav_data, filter_desc=None))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    duration: float
    qps: float
    ci_lower: float
    ci_upper: float


def benchmark(
    func: Callable[[], NDArray],
    iterations: int,
    num_threads: int,
    num_sets: int = 5,
) -> tuple[BenchmarkResult, NDArray]:
    """Benchmark a function using multiple threads.

    Args:
        name: Name of the benchmark
        func: Function to benchmark
        iterations: Total number of iterations across all threads per set
        num_threads: Number of concurrent threads
        num_sets: Number of benchmark sets to run for confidence interval calculation

    Returns:
        BenchmarkResult with timing statistics and confidence intervals
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
    )
    return result, output  # pyre-ignore[61]


def run_benchmark_suite(
    wav_data: bytes,
    ref: NDArray,
    iterations: int,
    num_threads: int,
    num_sets: int = 5,
) -> tuple[BenchmarkResult, BenchmarkResult, BenchmarkResult]:
    """Run benchmarks for both libraries with given parameters.

    Args:
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample
        duration_seconds: Duration of audio in seconds
        iterations: Number of iterations for benchmarking per set
        num_threads: Number of threads (use 1 for single-threaded)
        num_sets: Number of benchmark sets to run for confidence interval calculation

    Returns:
        Tuple of (spdl_wav_result, spdl_audio_result, soundfile_result)
    """

    spdl_wav_result, output = benchmark(
        func=lambda: spdl.io.load_wav(wav_data),
        iterations=iterations,
        num_threads=num_threads,
        num_sets=num_sets,
    )
    np.testing.assert_array_equal(output, ref)
    spdl_audio_result, output = benchmark(
        func=lambda: load_sp(wav_data),
        iterations=iterations,
        num_threads=num_threads,
        num_sets=num_sets,
    )
    np.testing.assert_array_equal(output, ref)
    soundfile_result, output = benchmark(
        func=lambda: load_sf(wav_data),
        iterations=iterations,
        num_threads=num_threads,
        num_sets=num_sets,
    )
    if output.ndim == 1:
        output = output[:, None]
    np.testing.assert_array_equal(output, ref)

    return spdl_wav_result, spdl_audio_result, soundfile_result


@dataclass(frozen=True)
class Config:
    sample_rate: int
    num_channels: int
    bits_per_sample: int
    duration_seconds: float


def main() -> None:
    """Run comprehensive benchmark suite."""

    benchmark_configs = [
        # (sample_rate, num_channels, bits_per_sample, duration_seconds, iterations)
        # Config(8000, 1, 16, 1.0),  # Low quality mono
        # Config(16000, 1, 16, 1.0),  # Speech quality mono
        # Config(48000, 2, 16, 1.0),  # High quality stereo
        # Config(48000, 8, 16, 1.0),  # Multi-channel audio
        Config(44100, 2, 16, 1.0),  # CD quality stereo
        Config(44100, 2, 16, 10.0),  #
        Config(44100, 2, 16, 60.0),  #
        # (44100, 2, 24, 1.0, 100),  # 24-bit audio
    ]

    for cfg in benchmark_configs:
        print(f"Benchmarking: {cfg}")
        wav_data, ref = create_wav_data(
            sample_rate=cfg.sample_rate,
            num_channels=cfg.num_channels,
            bits_per_sample=cfg.bits_per_sample,
            duration_seconds=cfg.duration_seconds,
        )
        print(
            f"Threads,"
            f"SPDL WAV QPS ({cfg.duration_seconds} sec),SPDL WAV CI,"
            f"SPDL Audio QPS ({cfg.duration_seconds} sec),SPDL Audio CI,"
            f"soundfile QPS ({cfg.duration_seconds} sec),soundfile CI"
        )
        results = []
        for num_threads in [1, 2, 4, 8, 16]:
            spdl_wav_result, spdl_audio_result, soundfile_result = run_benchmark_suite(
                wav_data,
                ref,
                iterations=100 * num_threads,
                num_threads=num_threads,
                num_sets=5,
            )
            results.append(
                (num_threads, spdl_wav_result, spdl_audio_result, soundfile_result)
            )
            print(
                f"{num_threads},"
                f"{spdl_wav_result.qps:.2f},[{spdl_wav_result.ci_lower:.2f}-{spdl_wav_result.ci_upper:.2f}],"
                f"{spdl_audio_result.qps:.2f},[{spdl_audio_result.ci_lower:.2f}-{spdl_audio_result.ci_upper:.2f}],"
                f"{soundfile_result.qps:.2f},[{soundfile_result.ci_lower:.2f}-{soundfile_result.ci_upper:.2f}]"
            )


if __name__ == "__main__":
    main()
