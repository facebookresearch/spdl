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
    "get_pipeline",
    "load_npy",
    "load_npy_spdl",
    "load_torch",
    "run_pipeline",
    "DataSource",
]

import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from io import BytesIO
from typing import Generic, TypeVar

import numpy as np
import spdl.io
import torch
from numpy.typing import NDArray
from spdl.pipeline import Pipeline, PipelineBuilder

# pyre-strict

T = TypeVar("T")


def get_pipeline(
    src: Iterable[T],
    load_fn: Callable[[T], list[NDArray]],
    num_workers: int,
    mode: str,
) -> Pipeline:
    """Build a pipeline to iterate the source with the load function in different parallelism.

    Args:
        src: The data source.
        load_fn: The function that loads NumPy NDArray from the source byte string.
        num_workers: The number of worker threads or processes.
        mode: The mode of parallelism. The valid values are ``"mt"`` (multi-threading)
            and ``"mp"`` (multi-processing).

    Returns:
        The resulting pipeline.
    """
    match mode:
        case "mt":
            executor = ThreadPoolExecutor(num_workers)
        case "mp":
            executor = ProcessPoolExecutor(num_workers)
        case _:
            raise ValueError(f'The `mode` must be either "mt" or "mp". Found: {mode}')

    return (
        PipelineBuilder()
        .add_source(src)
        .pipe(load_fn, concurrency=num_workers, executor=executor)
        .add_sink(buffer_size=1)
        .build(num_threads=1)
    )


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


class DataSource(Generic[T]):
    """Keep yielding the same data given times.

    Args:
        data: Data to be yielded.
        repeat: The number of yields.
    """

    def __init__(self, data: T, repeat: int) -> None:
        self.data = data
        self.repeat = repeat

    def __iter__(self) -> Iterator[T]:
        for _ in range(self.repeat):
            yield self.data


def run_pipeline(pipeline: Pipeline[...]) -> tuple[int, float]:
    """Run the pipeline and measure the time."""
    t0 = time.monotonic()
    with pipeline.auto_stop():
        num_items = 0
        for _ in pipeline:
            num_items += 1
    elapsed = time.monotonic() - t0
    return num_items, elapsed


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


def main() -> None:
    """The entrypoint from CLI."""
    configs = [
        ("torch", False, "torch"),
        ("npy", False, "np"),
        ("npy", False, "spdl"),
        ("npz", False, "np"),
        ("npz", True, "np"),
        ("npz", False, "spdl"),
        ("npz", True, "spdl"),
    ]
    for data_format, compressed, impl in configs:
        src = DataSource(get_mock_data(data_format, compressed), repeat=1000)
        load_fn = _get_load_fn(data_format, impl)
        for mode in ["mp", "mt"]:
            for num_workers in [32, 16, 8, 4, 2, 1]:
                pipeline = get_pipeline(
                    src,  # pyre-ignore: [6]
                    load_fn,
                    num_workers,
                    mode,
                )
                num_items, elapsed = run_pipeline(pipeline)
                qps = num_items / elapsed
                print(
                    f"{data_format},{compressed},{impl},{mode},{num_workers},{qps:.1f}"
                )


if __name__ == "__main__":
    main()
