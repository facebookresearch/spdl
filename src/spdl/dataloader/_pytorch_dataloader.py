# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

__all__ = ["PyTorchDataLoader", "get_pytorch_dataloader"]

import logging
import multiprocessing as mp
import os
import pickle
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from types import ModuleType
from typing import cast, Sized, TYPE_CHECKING, TypeVar

from spdl._internal import import_utils, log_api_usage_once
from spdl.pipeline import Pipeline, PipelineBuilder

if TYPE_CHECKING:
    import torch
else:
    torch: ModuleType = import_utils.lazy_import("torch")


K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

_LG: logging.Logger = logging.getLogger(__name__)


################################################################################
# ProcessExecutor
################################################################################

_DATASET: "torch.utils.data.dataset.Dataset[T]" = None  # pyre-ignore: [15]
_COLLATE_FN: Callable = None  # pyre-ignore: [15]


def _get_item(index: K) -> ...:
    global _DATASET, _COLLATE_FN
    return _COLLATE_FN(_DATASET[index])


def _get_items(indices: list[K]) -> ...:
    global _DATASET, _COLLATE_FN
    if hasattr(_DATASET, "__getitems__"):
        return _COLLATE_FN(_DATASET.__getitems__(indices))  # pyre-ignore: [16]
    return _COLLATE_FN([_DATASET[index] for index in indices])


def _init_dataset(name: str, collate_fn: Callable) -> None:
    _LG.info("[%s] Initializing dataset.", os.getpid())
    shmem = SharedMemory(name=name)
    global _DATASET, _COLLATE_FN
    _DATASET = pickle.loads(shmem.buf)
    _COLLATE_FN = collate_fn


def _get_executor(
    name: str,
    collate_fn: Callable[[list[T]], U],
    num_workers: int,
    mp_ctx: mp.context.BaseContext,
) -> ProcessPoolExecutor:
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_ctx,
        initializer=_init_dataset,
        initargs=(name, collate_fn),
    )
    return executor


def _serialize_dataset(dataset: "torch.utils.data.dataset.Dataset[T]") -> SharedMemory:
    _LG.info("Serializing dataset.")
    t0 = time.monotonic()
    data = pickle.dumps(dataset)
    shmem = SharedMemory(create=True, size=len(data))
    shmem.buf[:] = data
    elapsed = time.monotonic() - t0
    _LG.info(
        "Written dataset into shared memory %s (%s MB) in %.2f seconds",
        shmem.name,
        f"{len(data) // 1_000_000:_d}",
        elapsed,
    )
    return shmem


class PyTorchDataLoader(Iterable[V]):
    """PyTorchDataLoader()

    A PyTorch-style data loader that works on map-style dataset.
    Use :py:func:`get_pytorch_dataloader` to instantiate this class.
    You can use this class as almost drop-in replacement of PyTorch's DataLoader class.

    The architecture of data loader is different in following ways:

    - Only the dataset and the collate function are copied to the worker process.
      (Sampler and Generator are not copied)
    - The dataset is copied to worker processed via shared memory.
    - Sampler is executed in the main process and the resulting indices are passed to the
      worker processes.
    - Worker processes share the same input/output queues.
      (PyTorch creates a set of i/o queues for each worker process.)

    Due to the way Dataset is defined, this class still has to copy the dataset to
    each worker process. So the memory consumption is not reduced.
    However, fast initialization and reduced inter-process communication makes
    this implementation faster than PyTorch DataLoader.

    :ivar: dataset: The source dataset.
    """

    def __init__(
        self,
        *,
        dataset: "torch.utils.data.dataset.Dataset[T]",
        shmem: SharedMemory,  # to keep the reference alive
        sampler: "torch.utils.data.sampler.Sampler[K]",
        fetch_fn: Callable[[K], U],
        collate_fn: Callable[[list[U]], V],
        transfer_fn: Callable[[V], V],
        mp_ctx: mp.context.BaseContext,
        num_workers: int,
        timeout: float | None,
        buffer_size: int,
        output_order: str = "completion",
    ) -> None:
        log_api_usage_once("spdl.dataloader.PyTorchDataLoader")

        self.dataset = dataset  # For external access.
        self._shmem: SharedMemory = shmem
        self._sampler = sampler
        self._fetch_fn = fetch_fn
        self._collate_fn = collate_fn
        self._transfer_fn = transfer_fn
        self._mp_ctx = mp_ctx
        self._num_workers = num_workers
        self._buffer_size = buffer_size
        self._timeout = timeout
        self._output_order = output_order

    def __len__(self) -> int:
        """Returns the number of samples/batches this data loader returns."""
        return len(cast(Sized, self._sampler))

    def _get_pipeline(self) -> tuple[ProcessPoolExecutor, Pipeline]:
        executor = _get_executor(
            self._shmem.name, self._collate_fn, self._num_workers, self._mp_ctx
        )
        builder = (
            PipelineBuilder()
            .add_source(self._sampler)
            .pipe(
                self._fetch_fn,
                executor=executor,
                output_order=self._output_order,
                concurrency=self._num_workers,
            )
        )
        if self._transfer_fn:
            builder.pipe(
                self._transfer_fn,
                output_order=self._output_order,
            )

        pipeline = builder.add_sink(self._buffer_size).build(num_threads=1)
        return executor, pipeline

    def __iter__(self) -> Iterator[V]:
        """Iterate on the dataset and yields samples/batches."""
        executor, pipeline = self._get_pipeline()
        with executor, pipeline.auto_stop():
            for item in pipeline.get_iterator(timeout=self._timeout):
                yield item


################################################################################
# resolve sampler, fetch and collate
################################################################################


def _get_sampler(
    dataset: "torch.utils.data.dataset.Dataset[T]",
    shuffle: bool,
    generator: "torch.Generator | None",
) -> "torch.utils.data.sampler.Sampler[int]":
    from torch.utils.data.sampler import (
        RandomSampler,
        SequentialSampler,
    )

    assert hasattr(dataset, "__len__")
    ds = cast(Sized, dataset)
    return RandomSampler(ds, generator=generator) if shuffle else SequentialSampler(ds)


def _resolve_sampler(
    dataset: "torch.utils.data.dataset.Dataset[T]",
    batch_size: int | None = 1,
    shuffle: bool = False,
    sampler: "torch.utils.data.sampler.Sampler[K] | None" = None,
    batch_sampler: "torch.utils.data.sampler.Sampler[list[K]] | None" = None,
    collate_fn: Callable[[list[T]], U] | None = None,
    drop_last: bool = False,
    generator: "torch.Generator | None" = None,
) -> "tuple[torch.utils.data.sampler.Sampler[K], Callable[[K], U], Callable[[list[T]], U]]":
    from torch.utils.data.dataloader import default_collate, default_convert
    from torch.utils.data.sampler import BatchSampler

    if all(s is not None for s in [sampler, batch_sampler]):
        raise ValueError("`sampler` and `batch_sampler` are mutually exclusive.")

    if all(o is not None for o in [batch_size, batch_sampler]):
        raise ValueError("`batch_size` and `batch_sampler` are mutually exclusive.")

    if any(s is not None for s in [sampler, batch_sampler]) and shuffle:
        raise ValueError(
            "`shuffle` must be False when `batch_sampler` or `sampler` is provided."
        )

    if batch_sampler is not None and drop_last:
        raise ValueError("`drop_last` must be False when `batch_sampler` is provided.")

    if batch_size is None and drop_last:
        raise ValueError("`drop_last` must be False when `batch_size` is None.")

    if batch_sampler is not None:
        _sampler = batch_sampler
        _fetch_fn = _get_items
        _collate_fn = collate_fn or default_collate
    elif batch_size is not None:
        _sampler = BatchSampler(
            sampler or _get_sampler(dataset, shuffle, generator),  # pyre-ignore: [6]
            batch_size,
            drop_last,
        )
        _fetch_fn = _get_items
        _collate_fn = collate_fn or default_collate
    elif sampler is not None:
        _sampler = sampler
        _fetch_fn = _get_item
        _collate_fn = collate_fn or default_convert
    else:
        _sampler = _get_sampler(dataset, shuffle, generator)
        _fetch_fn = _get_item
        _collate_fn = collate_fn or default_convert

    return _sampler, _fetch_fn, _collate_fn


################################################################################
# get_pytorch_dataloader
################################################################################


def get_pytorch_dataloader(
    dataset: "torch.utils.data.dataset.Dataset[T]",
    batch_size: int | None = 1,
    shuffle: bool = False,
    sampler: "torch.utils.data.sampler.Sampler[K] | None" = None,
    batch_sampler: "torch.utils.data.sampler.Sampler[list[K]] | None" = None,
    num_workers: int = 1,
    collate_fn: Callable[[list[T]], U] | None = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float | None = None,
    worker_init_fn: None = None,
    multiprocessing_context: str | mp.context.BaseContext | None = None,
    generator: "torch.Generator | None" = None,
    *,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory_device: str | None = None,
    in_order: bool = False,
) -> PyTorchDataLoader[U]:
    from torch.utils.data.dataloader import IterableDataset

    if isinstance(dataset, IterableDataset):
        raise ValueError("IterableDataset is not supported.")

    if worker_init_fn is not None:
        raise ValueError("`worker_init_fn` is not supported.")

    if pin_memory_device is not None:
        raise ValueError("`pin_memory_device` is not supported.")

    if persistent_workers:
        raise ValueError("`persistent_workers` is not supported.")

    if timeout is not None and timeout < 0:
        raise ValueError(f"`timeout` must be positive. Found: {timeout}.")

    if num_workers < 0:
        raise ValueError(f"`num_workers` must be greater than 0. Found: {num_workers}")
    elif num_workers == 0:
        warnings.warn(
            "`num_workers` is 0. Setting `num_workers` to 1 for single process dataloading.",
            stacklevel=2,
        )
        num_workers = 1

    buffer_size = prefetch_factor * num_workers

    _sampler, _fetch_fn, _collate_fn = _resolve_sampler(
        dataset,
        batch_size,
        shuffle,
        sampler,
        batch_sampler,
        collate_fn,
        drop_last,
        generator,
    )

    from torch.utils.data._utils.pin_memory import pin_memory as pin_memory_fn

    if pin_memory and not torch.cuda.is_available():
        _LG.warning(
            "'pin_memory' argument is set as true but no accelerator is found, "
            "then device pinned memory won't be used."
        )

    transfer_fn = (
        pin_memory_fn if pin_memory and torch.accelerator.is_available() else None
    )

    mp_ctx = (
        multiprocessing_context
        if isinstance(multiprocessing_context, mp.context.BaseContext)
        else mp.get_context(multiprocessing_context)
    )
    _LG.info("Using multiprocessing context: %s", mp_ctx.get_start_method())
    shmem = _serialize_dataset(dataset)

    return PyTorchDataLoader(
        dataset=dataset,
        shmem=shmem,
        sampler=_sampler,
        fetch_fn=_fetch_fn,  # pyre-ignore: [6]
        collate_fn=_collate_fn,
        transfer_fn=transfer_fn,
        mp_ctx=mp_ctx,
        num_workers=num_workers,
        timeout=timeout,
        buffer_size=buffer_size,
        output_order="input" if in_order else "completion",
    )
