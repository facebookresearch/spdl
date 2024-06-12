import asyncio
import warnings
from collections.abc import Iterable
from typing import TypeVar

import torch
from spdl.dataloader import BackgroundGenerator
from spdl.utils import iter_batch
from torch import Tensor

from torch.utils.data import (
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


def _get_sampler(dataset, shuffle, generator):
    if shuffle:
        return RandomSampler(dataset, generator=generator)
    return SequentialSampler(dataset)


async def _aiter(gen, func, buffer_size):
    task_buffer = asyncio.Queue(buffer_size)
    while True:
        if task_buffer.full():
            # TOOD: add error handling
            yield await task_buffer.get_nowait()

        try:
            item = next(gen)
        except StopIteration:
            break
        else:
            task_buffer.put_nowait(asyncio.create_task(func(item)))
    while not task_buffer.empty():
        yield await task_buffer.get_nowait()


class _BackgroundIterator(IterableDataset):
    def __init__(
        self,
        iterable,
        collate_fn,
        num_workers: int,
        buffer_size: int = 2,
        timeout: int | float | None = 30,
    ) -> None:
        self.iterable = iterable
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.timeout = timeout

    def __iter__(self):
        gen = _aiter(
            self.iterable,
            self.collate_fn,
            self.buffer_size,
        )
        bgg = BackgroundGenerator(
            gen,
            num_workers=self.num_workers,
            queue_size=1,
            timeout=self.timeout,
        )
        return iter(bgg)


async def _default_collate(val) -> Tensor:
    return default_collate(val)


def _get_num_items(dataset, batch_size, drop_last):
    num_items = len(dataset) if hasattr(dataset, "__len__") else None
    if num_items is not None and batch_size is not None:
        if drop_last:
            num_items = num_items // batch_size
        else:
            num_items = num_items // batch_size + (1 if num_items % batch_size else 0)
    return num_items


class _DataLoader(torch.utils.data.DataLoader):
    """Wrap DataLoader to override __len__"""
    def __init__(self, ds, num_items):
        super().__init__(
            ds,
            batch_size=None,
            num_workers=0,
        )
        self.num_items = num_items

    def __len__(self):
        if self.num_items is not None:
            return self.num_items
        return super().__len__()


T_co = TypeVar("T_co", covariant=True)


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
    pin_memory_device: str = ""
) -> torch.utils.data.DataLoader[T_co]:
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
                "`samplers` and `batch_sampler` cannot be specified at the same time."
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

    if timeout == 0:
        warnings.warn(
            "`timeout=0` is treated as immediate timeout. "
            "To disable timeout, provide `None`.",
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

    if worker_init_fn is not None:
        warnings.warn(
            "`worker_init_fn` is not supported. It will be ignored.",
            stacklevel=2,
        )
    if multiprocessing_context is not None:
        warnings.warn(
            "`multiprocessing_context` is ignored.",
            stacklevel=2,
        )
    if persistent_workers:
        warnings.warn(
            "`persistent_workers` is not supported.",
            stacklevel=2,
        )

    collate_fn = collate_fn or _default_collate
    num_items = _get_num_items(dataset, batch_size, drop_last)

    if isinstance(dataset, IterableDataset):
        iter_dataset = (
            dataset
            if batch_size is None
            else iter_batch(dataset, batch_size, drop_last)
        )

    else:
        _batch_sampler = batch_sampler or iter_batch(
            sampler or _get_sampler(dataset, shuffle, generator),
            batch_size=batch_size,
            drop_last=drop_last,
        )

        def _gen():
            for batch_idx in _batch_sampler:
                yield [dataset[j] for j in batch_idx]

        iter_dataset = _gen()

    bg_iterator = _BackgroundIterator(
        iter_dataset,
        collate_fn,
        num_workers,
        buffer_size=prefetch_factor,
        timeout=timeout,
    )

    return _DataLoader(bg_iterator, num_items)
