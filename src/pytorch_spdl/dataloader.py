"""This module implements interfaces compatible to PyTorch."""

# pyre-unsafe

import logging
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import torch
from torch.utils.data import (
    BatchSampler,
    Dataset,
    default_collate,
    IterableDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

__all__ = ["DataLoader"]


T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

_LG = logging.getLogger(__name__)


class _Loader:
    """Base structure to be used in place of sampler/batch_sampler.

    It applies the given function to the result of generator. Generator
    can yield either single item or batch of items. It does not matter for
    this class as long as the process function is able to handle it.

    It can be written as simple generator expression, but PyTorch expect
    sampler/batch_sampler to have `__len__` method, so it is in the shape of
    class.

    `num_workers` determines the number of threads to be used for processing.
    `num_buffer` determines how many  pforcess functions should be executed
    concurrently.

    To use the thread resource properly, `num_buffer` must be larger or equal
    to `num_workers`. Otherwise, the thread pool will be starved.
    """

    def __init__(self, sample_generator, func, num_buffer, num_workers):
        self.sample_generator = sample_generator
        self.func = func
        self.num_buffer = num_buffer
        self.num_workers = num_workers

    def __iter__(self):
        sentinel = object()

        def _wrap(item):
            try:
                return self.func(item)
            except Exception as e:
                _LG.error("Failed to process: %s", e)
                return sentinel

        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while True:
                if len(futures) >= self.num_buffer:
                    if (result := futures.pop(0).result()) is not sentinel:
                        yield result

                try:
                    samples = next(self.sample_generator)
                except StopIteration:
                    break
                else:
                    futures.append(executor.submit(_wrap, samples))

            while futures:
                if (result := futures.pop(0).result()) is not sentinel:
                    yield result


def _batch_indexer(dataset, batch_sampler):
    for idx in batch_sampler:
        if hasattr(dataset, "__getitems__"):
            yield dataset.__getitems__(idx)
        else:
            yield [dataset[i] for i in idx]


class _SizedBatchSampler(_Loader):
    """Base structure to be used in place of batch_sampler.

    It can be written as simple generator expression, but PyTorch expect
    sampler/batch_sampler to have `__len__` method, so it is in the shape of
    class.
    """

    def __init__(self, dataset, batch_sampler, load_fn, num_buffer, num_workers):
        super().__init__(
            _batch_indexer(dataset, batch_sampler), load_fn, num_buffer, num_workers
        )
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)


def _indexer(dataset, sampler):
    for i in sampler:
        yield dataset[i]


class _SizedSampler(_Loader):
    """Index-based sampling without batch."""

    def __init__(self, dataset, sampler, load_fn, num_buffer, num_workers):
        super().__init__(_indexer(dataset, sampler), load_fn, num_buffer, num_workers)
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)


def _wrap(dataset):
    for item in dataset:
        yield [item]


class _Iterator(_Loader):
    """Like a regular iterator, but has __len__."""

    def __init__(self, dataset, load_fn, num_buffer, num_workers):
        super().__init__(_wrap(dataset), load_fn, num_buffer, num_workers)
        self.dataset = dataset

    def __iter__(self):
        for item in super().__iter__():
            yield item[0]

    def __len__(self):
        return len(self.dataset)


def _batched(iterable, batch_size, drop_last):
    batch = []
    for item in iterable:
        batch.append(item)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch and not drop_last:
        yield batch


def _get_num_items(num_items, batch_size, drop_last):
    num_batches = num_items // batch_size
    if not drop_last and (num_items % batch_size > 0):
        num_batches += 1
    return num_batches


class _BatchIterator(_Loader):
    """Iterate dataset sequentially and batch."""

    def __init__(
        self, dataset, batch_size, drop_last, load_fn, num_buffer, num_workers
    ):
        super().__init__(
            _batched(dataset, batch_size, drop_last), load_fn, num_buffer, num_workers
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        return _get_num_items(len(self.dataset), self.batch_size, self.drop_last)


def _get_sampler(num_samples, shuffle=False, generator=None):
    class _Sized:
        def __init__(self, n):
            self.n = n

        def __len__(self) -> int:
            return self.n

    dummy = _Sized(num_samples)
    if shuffle:
        return RandomSampler(dummy, generator=generator)
    return SequentialSampler(dummy)


def _get_batch_sampler(
    num_samples: int,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    generator,
    sampler: Iterable[int] | None,
):
    return BatchSampler(
        sampler or _get_sampler(num_samples, shuffle, generator), batch_size, drop_last
    )


class _DummyDataSet:
    """We perform load and preprocessing in [batch] sampler, but PyTorch DataLoader

    expects that dataset to be queried from the output of the sampler, so this dataset
    just returns the query key as-is.
    """

    def __getitem__(self, key):
        return key

    def __getitems__(self, key):
        return key


def DataLoader(
    dataset: Dataset[T_co],
    batch_size: int | None = 1,
    shuffle: bool | None = None,
    sampler: Sampler | Iterable | None = None,
    batch_sampler: Sampler[list] | Iterable[list] | None = None,
    num_workers: int = 4,
    collate_fn=None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: int | float | None = None,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    *,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
) -> torch.utils.data.DataLoader[T_co]:
    """A factory function to create PyTorch DataLoader but with multi-threading.

    The resulting dataloader can be used as a drop-in replacement of PyTorch
    DataLoader.

    The DataLoader creates a batch sampler that wraps the given arguments.
    Inside of the batch sampler, it spawns a thread pool and runs collate function
    in the pool.

    To achieve high performance, `dataset` and `collate_fn` should work with such
    settings. Therefore, making the following changes are recommended.

    - `dataset` should return the idicator of the sample, and it should not perform
      CPU-intensive computations like decoding images and videos.
    - `collate_fn` should perform batch loading and preprocessing in thread-safe,
      GIL-free manner.

    SPDL provides load/preprocessing functions for audio/video/image that are
    thread-safe and releases GIL.

    For NLP, OpenAI's `tiktoken` is known to work. However, do not use methods
    suffixed with `_batch` as these methods individually spawn thread pool.
    Instead, use the single decode/encode method, so that it uses the thread pool
    controlled by the DataLoader instance.

    Arguments for this function are mostly same as [torch.utils.data.DataLoader][],
    except that the following.

    - `collate_fn` is used regarless of loading batch of samples or single samples.
      When loading single samples (both `batch_sample` and `batch_size` are `None`),
      `collate_fn` should accept single item.
    - `timeout`, `worker_init_fn`, `multiprocessing_context`, `persistent_worker`
      and `pin_memory_device` are not used.

    (`pin_memory_device` is supported by [spdl.io.transfer_buffer][] function.)
    """
    # Note:
    #
    # DataLoader has the following 6 mode of operations, determined by the arguments
    #
    # For IterableDataset
    # (`shuffle`, `sampler` and `batch_sampler` are not compatible)
    #
    # supports len?, batch_size, drop_last,  --> Mode of Operation
    #      N            None         -           (Sequential) Single iteration
    #      Y            None         -           (Sequential) Single iteration with len support
    #      N            Given       Y/N          (Sequential) Batch iteration
    #      Y            Given       Y/N          (Sequential) Batch iteration with len support
    #
    # For Map-style dataset (`len` must be provided.)
    #
    # batch_size, drop_last, shuffle, sampler, batch_sampler --> Mode of Operation
    #     None        -        Y/N       -          -            Single sampling
    #     None        -         -      Given        -            Single sampling
    #     Given      Y/N       Y/N       -          -            Batch sampling
    #     Given      Y/N        -      Given        -            Batch sampling
    #       -         -         -        -        Given          Batch sampling
    #

    if isinstance(dataset, IterableDataset):
        if any(arg is not None for arg in [shuffle, sampler, batch_sampler]):
            warnings.warn(
                "`dataset` is an instance of IterableDataset. "
                "`shuffle`, `sampler` and `batch_sampler` are ignored.",
                stacklevel=2,
            )
    else:
        if sampler is not None and batch_sampler is not None:
            raise ValueError(
                "`sampler` and `batch_sampler` cannot be specified at the same time."
            )
        if sampler is not None:
            if shuffle:
                warnings.warn(
                    "`sampler` is provided. `shuffle` will be ignored.", stacklevel=2
                )
        if batch_sampler is not None:
            if shuffle:
                warnings.warn(
                    "`batch_sampler` is provided. `shuffle` will be ignored.",
                    stacklevel=2,
                )
            if batch_size != 1:
                warnings.warn(
                    "`batch_sampler` is provided, `batch_size` will be ignored.",
                    stacklevel=2,
                )

    if num_workers <= 0:
        raise ValueError("`num_workers` must be greater than 0.")

    if prefetch_factor <= 0:
        raise ValueError("`prefetch_factor` must be greater than 0.")

    if pin_memory:
        warnings.warn(
            "In SPDL, `pin_memory` is supported via convert function. "
            "See `spdl.io.convert_frames` for the detail.",
            stacklevel=2,
        )

    if timeout is not None:
        warnings.warn(
            "`timeout` is not supported. It is ignored.",
            stacklevel=2,
        )

    if worker_init_fn is not None:
        warnings.warn(
            "`worker_init_fn` is not supported. It is ignored.",
            stacklevel=2,
        )
    if multiprocessing_context is not None:
        warnings.warn(
            "`multiprocessing_context` is not supported. It is ignored.",
            stacklevel=2,
        )
    if persistent_workers:
        warnings.warn(
            "`persistent_workers` is not supported. It is ignored.",
            stacklevel=2,
        )

    collate_fn = collate_fn or default_collate
    num_buffer = prefetch_factor * num_workers
    ds = _DummyDataSet()

    if isinstance(dataset, IterableDataset):
        if batch_size is not None:
            # Batch iterate mode
            _batch_loader = _BatchIterator(
                dataset, batch_size, drop_last, collate_fn, num_buffer, num_workers
            )
            return torch.utils.data.DataLoader(
                ds, batch_sampler=_batch_loader, collate_fn=lambda x: x, num_workers=0
            )
        else:
            # Single iterate mode
            _loader = _Iterator(dataset, collate_fn, num_buffer, num_workers)
            return torch.utils.data.DataLoader(
                ds, sampler=_loader, batch_size=None, num_workers=0
            )
    else:
        num_samples = len(dataset)
        if batch_size is not None or batch_sampler is not None:
            # Batch sampling mode
            _batch_sampler = batch_sampler or BatchSampler(
                sampler or _get_sampler(num_samples, shuffle, generator),
                batch_size,
                drop_last,
            )

            _loader = _SizedBatchSampler(
                dataset,
                _batch_sampler,
                collate_fn,
                num_buffer,
                num_workers,
            )

            return torch.utils.data.DataLoader(
                ds, batch_sampler=_loader, collate_fn=lambda x: x, num_workers=0
            )
        else:
            # Single sampling mode
            _sampler = _get_sampler(num_samples, shuffle, generator)
            _loader = _SizedSampler(
                dataset, _sampler, collate_fn, num_buffer, num_workers
            )
            return torch.utils.data.DataLoader(
                ds, sampler=_loader, batch_size=None, num_workers=0
            )
