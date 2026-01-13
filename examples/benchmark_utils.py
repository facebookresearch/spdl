#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common utilities for benchmark scripts.

This module provides a standardized framework for running benchmarks with:

- Configurable executor types (
  :py:class:`~concurrent.futures.ThreadPoolExecutor`,
  :py:class:`~concurrent.futures.ProcessPoolExecutor`,
  :py:class:`~concurrent.futures.InterpreterPoolExecutor`)
- Warmup phase to exclude executor initialization overhead
- Statistical analysis with confidence intervals
- CSV export functionality
- Python version and free-threaded ABI detection

.. seealso::

   - :doc:`./benchmark_tarfile`
   - :doc:`./benchmark_wav`
   - :doc:`./benchmark_numpy`

"""

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "ExecutorType",
    "get_default_result_path",
    "load_results_from_csv",
    "save_results_to_csv",
]

import csv
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import (
    as_completed,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from typing import Any, Generic, TypeVar

import numpy as np
import psutil
import scipy.stats

T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


@dataclass
class BenchmarkResult(Generic[ConfigT]):
    """BenchmarkResult()

    Generic benchmark result containing configuration and performance metrics.

    This class holds both the benchmark-specific configuration and the
    common performance statistics. It is parameterized by the config type,
    which allows each benchmark script to define its own configuration dataclass.
    """

    config: ConfigT
    """Benchmark-specific configuration (e.g., data format, file size, etc.)"""

    executor_type: str
    """Type of executor used (thread, process, or interpreter)"""

    qps: float
    """Queries per second (mean)"""

    ci_lower: float
    """Lower bound of 95% confidence interval for QPS"""

    ci_upper: float
    """Upper bound of 95% confidence interval for QPS"""

    date: str
    """When benchmark was run. ISO 8601 format."""

    python_version: str
    """Python version used for the benchmark"""

    free_threaded: bool
    """Whether Python is running with free-threaded ABI."""

    cpu_percent: float
    """Average CPU utilization percentage during benchmark execution."""


class ExecutorType(Enum):
    """ExecutorType()

    Supported executor types for concurrent execution."""

    THREAD = "thread"
    """Use :py:class:`~concurrent.futures.ThreadPoolExecutor`."""

    PROCESS = "process"
    """Use :py:class:`~concurrent.futures.ProcessPoolExecutor`."""

    INTERPRETER = "interpreter"
    """Use :py:class:`~concurrent.futures.InterpreterPoolExecutor`.

    Requires Python 3.14+.
    """


def _get_python_info() -> tuple[str, bool]:
    """Get Python version and free-threaded ABI information.

    Returns:
        Tuple of (``python_version``, ``is_free_threaded``)
    """
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    try:
        is_free_threaded = not sys._is_gil_enabled()  # pyre-ignore[16]
    except AttributeError:
        is_free_threaded = False
    return python_version, is_free_threaded


def _create_executor(executor_type: ExecutorType, max_workers: int) -> Executor:
    """Create an executor of the specified type.

    Args:
        executor_type: Type of executor to create
        max_workers: Maximum number of workers

    Returns:
        Executor instance

    Raises:
        ValueError: If ``executor_type`` is not supported
    """
    match executor_type:
        case ExecutorType.THREAD:
            return ThreadPoolExecutor(max_workers=max_workers)
        case ExecutorType.PROCESS:
            return ProcessPoolExecutor(max_workers=max_workers)
        case ExecutorType.INTERPRETER:
            from concurrent.futures import InterpreterPoolExecutor  # pyre-ignore[21]

            return InterpreterPoolExecutor(max_workers=max_workers)
        case _:
            raise ValueError(f"Unsupported executor type: {executor_type}")


def _verify_workers(executor: Executor, expected_workers: int) -> None:
    """Verify that the executor has created the expected number of workers.

    Args:
        executor: The executor to verify
        expected_workers: Expected number of workers

    Raises:
        RuntimeError: If the number of workers doesn't match expected
    """
    match executor:
        case ThreadPoolExecutor():
            actual_workers = len(executor._threads)
        case ProcessPoolExecutor():
            actual_workers = len(executor._processes)
        case _:
            raise ValueError(f"Unexpected executor type {type(executor)}")

    if actual_workers != expected_workers:
        raise RuntimeError(
            f"Expected {expected_workers} workers, but executor has {actual_workers}"
        )


def _warmup_executor(
    executor: Executor, func: Callable[[], T], num_iterations: int
) -> T:
    """Warmup the executor by running the function multiple times.

    Args:
        executor: The executor to warmup
        func: Function to run for warmup
        num_iterations: Number of warmup iterations

    Returns:
        Output from the last warmup iteration
    """
    futures = [executor.submit(func) for _ in range(num_iterations)]
    last_output: T | None = None
    for future in as_completed(futures):
        last_output = future.result()
    return last_output  # pyre-ignore[7]


class BenchmarkRunner:
    """Runner for executing benchmarks with configurable executors.

    This class provides a standardized way to run benchmarks with:

    - Warmup phase to exclude executor initialization overhead
    - Multiple runs for statistical confidence intervals
    - Support for different executor types

    The executor is initialized and warmed up in the constructor to exclude
    initialization overhead from benchmark measurements.

    Args:
        executor_type: Type of executor to use
            (``"thread"``, ``"process"``, or ``"interpreter"``)
        num_workers: Number of concurrent workers
        warmup_iterations: Number of warmup iterations (default: ``2 * num_workers``)
    """

    def __init__(
        self,
        executor_type: ExecutorType,
        num_workers: int,
        warmup_iterations: int | None = None,
    ) -> None:
        self._executor_type: ExecutorType = executor_type

        warmup_iters = (
            warmup_iterations if warmup_iterations is not None else 2 * num_workers
        )

        self._executor: Executor = _create_executor(executor_type, num_workers)

        _warmup_executor(self._executor, partial(time.sleep, 1), warmup_iters)
        _verify_workers(self._executor, num_workers)

    @property
    def executor_type(self) -> ExecutorType:
        """Get the executor type."""
        return self._executor_type

    def __enter__(self) -> "BenchmarkRunner":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and shutdown executor."""
        self._executor.shutdown(wait=True)

    def _run_iterations(
        self,
        func: Callable[[], T],
        iterations: int,
        num_runs: int,
    ) -> tuple[list[float], list[float], T]:
        """Run benchmark iterations and collect QPS and CPU utilization samples.

        Args:
            func: Function to benchmark (takes no arguments)
            iterations: Number of iterations per run
            num_runs: Number of benchmark runs

        Returns:
            Tuple of (list of QPS samples, list of CPU percent samples, last function output)
        """
        qps_samples: list[float] = []
        cpu_samples: list[float] = []
        last_output: T | None = None

        process = psutil.Process()

        for _ in range(num_runs):
            process.cpu_percent()
            t0 = time.perf_counter()
            futures = [self._executor.submit(func) for _ in range(iterations)]
            for future in as_completed(futures):
                last_output = future.result()
            elapsed = time.perf_counter() - t0
            cpu_percent = process.cpu_percent()
            qps_samples.append(iterations / elapsed)
            cpu_samples.append(cpu_percent / iterations)

        return qps_samples, cpu_samples, last_output  # pyre-ignore[7]

    def run(
        self,
        config: ConfigT,
        func: Callable[[], T],
        iterations: int,
        num_runs: int = 5,
        confidence_level: float = 0.95,
    ) -> tuple[BenchmarkResult[ConfigT], T]:
        """Run benchmark and return results with configuration.

        Args:
            config: Benchmark-specific configuration
            func: Function to benchmark (takes no arguments)
            iterations: Number of iterations per run
            num_runs: Number of benchmark runs for confidence interval calculation
                (default: ``5``)
            confidence_level: Confidence level for interval calculation (default: ``0.95``)

        Returns:
            Tuple of (``BenchmarkResult``, last output from function)
        """
        qps_samples, cpu_samples, last_output = self._run_iterations(
            func, iterations, num_runs
        )

        qps_mean = np.mean(qps_samples)
        qps_std = np.std(qps_samples, ddof=1)
        degrees_freedom = num_runs - 1
        confidence_interval = scipy.stats.t.interval(
            confidence_level,
            degrees_freedom,
            loc=qps_mean,
            scale=qps_std / np.sqrt(num_runs),
        )

        cpu_mean = np.mean(cpu_samples)

        python_version, free_threaded = _get_python_info()
        date = datetime.now(timezone.utc).isoformat()

        result = BenchmarkResult(
            config=config,
            executor_type=self.executor_type.value,
            qps=float(qps_mean),
            ci_lower=float(confidence_interval[0]),
            ci_upper=float(confidence_interval[1]),
            date=date,
            python_version=python_version,
            free_threaded=free_threaded,
            cpu_percent=float(cpu_mean),
        )

        return result, last_output


def get_default_result_path(path: str, ext: str = ".csv") -> str:
    """Get the default result path with Python version appended."""
    base, _ = os.path.splitext(os.path.realpath(path))
    dirname = os.path.join(os.path.dirname(base), "data")
    filename = os.path.basename(base)
    python_version, free_threaded = _get_python_info()
    version_suffix = (
        f"_{'.'.join(python_version.split('.')[:2])}{'t' if free_threaded else ''}"
    )
    return os.path.join(dirname, f"{filename}{version_suffix}{ext}")


def save_results_to_csv(
    results: list[BenchmarkResult[Any]],
    output_file: str,
) -> None:
    """Save benchmark results to a CSV file.

    Flattens the nested BenchmarkResult structure (config + performance metrics)
    into a flat CSV format. Each row contains both the benchmark configuration
    fields and the performance metrics.

    Args:
        results: List of BenchmarkResult instances
        output_file: Output file path for the CSV file
    """
    if not results:
        raise ValueError("No results to save")

    flattened_results = []
    for result in results:
        config_dict = asdict(result.config)
        # convert bool to int for slight readability improvement of raw CSV file
        config_dict = {
            k: (int(v) if isinstance(v, bool) else v) for k, v in config_dict.items()
        }
        flattened = {
            "date": result.date,
            "python_version": result.python_version,
            "free_threaded": int(result.free_threaded),
            **config_dict,
            "executor_type": result.executor_type,
            "qps": result.qps,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "cpu_percent": result.cpu_percent,
        }
        flattened_results.append(flattened)

    # Get all field names from the first result
    fieldnames = list(flattened_results[0].keys())

    output_path = os.path.realpath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_path, "w", newline="") as csvfile:
        # Write generated marker as first line
        # Note: Splitting the marker so as to avoid linter consider this file as generated file
        csvfile.write("# @")
        csvfile.write("generated\n")

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result_dict in flattened_results:
            writer.writerow(result_dict)

    print(f"Results saved to {output_file}")


def load_results_from_csv(
    input_file: str,
    config_type: type[ConfigT],
) -> list[BenchmarkResult[ConfigT]]:
    """Load benchmark results from a CSV file.

    Reconstructs BenchmarkResult objects from the flattened CSV format created
    by :py:func:`save_results_to_csv`.
    Each row in the CSV is parsed into a :py:class:`BenchmarkResult`
    with the appropriate config type.

    Args:
        input_file: Input CSV file path
        config_type: The dataclass type to use for the config field

    Returns:
        List of BenchmarkResult instances with parsed config objects

    Raises:
        FileNotFoundError: If input_file does not exist
        ValueError: If CSV format is invalid or ``config_type`` is not a dataclass
    """
    if not hasattr(config_type, "__dataclass_fields__"):
        raise ValueError(f"config_type must be a dataclass, got {config_type}")
    fields: dict[str, Any] = config_type.__dataclass_fields__  # pyre-ignore[16]

    # Normalize input path and resolve symbolic links
    input_file = os.path.realpath(input_file)

    # Get the field names from the config dataclass
    config_fields = set(fields.keys())

    # Performance metric fields that are part of BenchmarkResult
    result_fields = {
        "executor_type",
        "qps",
        "ci_lower",
        "ci_upper",
        "date",
        "python_version",
        "free_threaded",
        "cpu_percent",
    }

    results: list[BenchmarkResult[ConfigT]] = []

    TRUES = ("true", "1", "yes")

    with open(input_file, newline="") as csvfile:
        reader = csv.DictReader((v for v in csvfile if not v.strip().startswith("#")))

        for row in reader:
            # Split row into config fields and result fields
            config_dict = {}
            result_dict = {}

            for key, value in row.items():
                if key in config_fields:
                    config_dict[key] = value
                elif key in result_fields:
                    result_dict[key] = value
                else:
                    # Unknown field - could be from config or result
                    # Try to infer based on whether it matches a config field name
                    config_dict[key] = value

            # Convert string values to appropriate types for config
            typed_config_dict = {}
            for field_name, field_info in fields.items():
                if field_name not in config_dict:
                    continue

                value = config_dict[field_name]
                field_type = field_info.type

                # Handle type conversions
                if field_type is int or field_type == "int":
                    typed_config_dict[field_name] = int(value)
                elif field_type is float or field_type == "float":
                    typed_config_dict[field_name] = float(value)
                elif field_type is bool or field_type == "bool":
                    typed_config_dict[field_name] = value.lower() in TRUES
                else:
                    # Keep as string or use the value as-is
                    typed_config_dict[field_name] = value

            result = BenchmarkResult(
                config=config_type(**typed_config_dict),
                executor_type=result_dict["executor_type"],
                qps=float(result_dict["qps"]),
                ci_lower=float(result_dict["ci_lower"]),
                ci_upper=float(result_dict["ci_upper"]),
                date=result_dict["date"],
                python_version=result_dict["python_version"],
                free_threaded=result_dict["free_threaded"].lower()
                in ("true", "1", "yes"),
                cpu_percent=float(result_dict.get("cpu_percent", 0.0)),
            )

            results.append(result)

    return results
