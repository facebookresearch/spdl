#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This example benchmarks the speed of loading data in different formats.

See `Case Studies / Data Format <../case_studies/data_format.html>`_ for
the detail of how data format and the loading function affects
the performance of the training pipeline.
"""

__all__ = [
    "main",
    "get_mock_data",
    "load_npy",
    "load_npy_spdl",
    "load_npz",
    "load_npz_spdl",
    "load_torch",
    "BenchmarkConfig",
]

# pyre-strict

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from io import BytesIO

import numpy as np
import spdl.io
import torch
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


def load_npy(items: list[bytes]) -> list[NDArray]:
    """Load arrays from serialized NPY binary strings using :py:func:`numpy.load`."""
    return [np.load(BytesIO(item), allow_pickle=False) for item in items]


def load_npy_spdl(items: list[bytes]) -> list[NDArray]:
    """Load arrays from serialized NPY binary strings using :py:func:`spdl.io.load_npy`."""
    return [spdl.io.load_npy(item) for item in items]


def load_npz(item: bytes) -> list[NDArray]:
    """Load arrays from a serialized NPZ binary string using :py:func:`numpy.load`."""
    data = np.load(BytesIO(item))
    return list(data.values())


def load_npz_spdl(item: bytes) -> list[NDArray]:
    """Load arrays from serialized NPZ binary strings using :py:func:`spdl.io.load_npz`."""
    data = spdl.io.load_npz(item)
    return list(data.values())


def load_torch(item: bytes) -> list[NDArray]:
    """Load arrays from a serialized PyTorch state dict."""
    return list(torch.load(BytesIO(item)).values())


def _get_load_fn(
    data_format: str, impl: str
) -> Callable[[list[bytes]], list[NDArray]] | Callable[[bytes], list[NDArray]]:
    match data_format:
        case "torch":
            return load_torch
        case "npy":
            if impl == "spdl":
                return load_npy_spdl
            return load_npy
        case "npz":
            if impl == "spdl":
                return load_npz_spdl
            return load_npz
        case _:
            raise ValueError(f"Unexpected data format: {data_format}")


def _dump_np(arr: NDArray | dict[str, NDArray], compressed: bool = False) -> bytes:
    with BytesIO() as buf:
        if isinstance(arr, dict):
            if compressed:
                np.savez_compressed(buf, allow_pickle=False, **arr)
            else:
                np.savez(buf, allow_pickle=False, **arr)
        else:
            np.save(buf, arr, allow_pickle=False)
        buf.seek(0)
        return buf.read()


def _dump_torch(arr: dict[str, NDArray]) -> bytes:
    with BytesIO() as buf:
        torch.save({k: torch.from_numpy(v) for k, v in arr.items()}, buf)
        buf.seek(0)
        return buf.read()


def get_mock_data(format: str, compressed: bool = False) -> tuple[bytes, bytes] | bytes:
    """Generate a single sample in the given format.

    The mock data resemboles an RGB image and its segmentation labels.

    Args:
        format: One of ``"npz"``, ``"npy"`` or ``"torch"``.
        compressed: If ``True``, NPZ file is compressed.
            (i.e. :py:func:`numpy.savez_compressed` is used.)

    Returns:
        Serialized mock arrays. If ``"npy"`` then arrays are serialized
        separately. Otherwise arrays are bundled together.
    """
    img = np.random.randint(256, size=(3, 640, 480), dtype=np.uint8)
    lbl = np.random.randint(256, size=(640, 480), dtype=np.uint8)

    match format:
        case "npz":
            return _dump_np({"img": img, "lbl": lbl}, compressed=compressed)
        case "npy":
            return _dump_np(img), _dump_np(lbl)
        case "torch":
            return _dump_torch({"img": img, "lbl": lbl})
        case _:
            raise ValueError(f"Unexpected `format`: {format}")


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    data_format: str
    compressed: bool
    impl: str
    num_workers: int


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark data format loading performance"
    )
    parser.add_argument(
        "--output",
        type=lambda p: os.path.realpath(p),
        default=DEFAULT_RESULT_PATH,
        help="Output path for the results",
    )
    return parser.parse_args()


def main() -> None:
    """The entrypoint from CLI."""
    args = _parse_args()

    # Define explicit configuration lists
    worker_counts = [32, 16, 8, 4, 2, 1]
    executor_types = [ExecutorType.PROCESS, ExecutorType.THREAD]

    # Define benchmark configurations
    # (data_format, compressed, impl)
    data_configs = [
        ("torch", False, "torch"),
        ("npy", False, "np"),
        ("npy", False, "spdl"),
        ("npz", False, "np"),
        ("npz", True, "np"),
        ("npz", False, "spdl"),
        ("npz", True, "spdl"),
    ]

    results: list[BenchmarkResult[BenchmarkConfig]] = []
    iterations = 1000
    num_runs = 5

    for num_workers in worker_counts:
        for executor_type in executor_types:
            with BenchmarkRunner(
                executor_type=executor_type,
                num_workers=num_workers,
                warmup_iterations=30 * num_workers,
            ) as runner:
                for data_format, compressed, impl in data_configs:
                    data = get_mock_data(data_format, compressed)

                    load_fn = _get_load_fn(data_format, impl)

                    result, _ = runner.run(
                        BenchmarkConfig(
                            data_format=data_format,
                            compressed=compressed,
                            impl=impl,
                            num_workers=num_workers,
                        ),
                        partial(load_fn, data),
                        iterations,
                        num_runs=num_runs,
                    )

                    results.append(result)
                    print(
                        f"{data_format},{compressed},{impl},{executor_type.value},{num_workers},{result.qps:.1f}"
                    )

    save_results_to_csv(results, args.output)
    plot_output = args.output.replace(".csv", ".png")
    print(
        f"\nBenchmark complete. To generate plots, run:\n"
        f"python benchmark_numpy_plot.py --input {args.output} --output {plot_output}"
    )


if __name__ == "__main__":
    main()
