import asyncio
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import TypeVar

from spdl._internal import import_utils
from spdl.dataloader import BackgroundGenerator
from spdl.utils import iter_batch
from torch.utils.data import RandomSampler, SequentialSampler


torch = import_utils.lazy_import("torch")

T = TypeVar("T")
U = TypeVar("U")


def _get_sampler(dataset, shuffle, generator):
    if shuffle:
        return RandomSampler(dataset, generator=generator)
    return SequentialSampler(dataset)


async def _sample(dataset, batch_sampler, buffer_size, collate_fn):
    task_buffer = asyncio.Queue(buffer_size)

    for idx in batch_sampler:
        task = asyncio.create_task(collate_fn([dataset[j] for j in idx]))
        task_buffer.put_nowait(task)

        if task_buffer.full():
            yield await task_buffer.get_nowait()

    while not task_buffer.empty():
        yield await task_buffer.get_nowait()


class PyTorchStyleDataLoader:
    def __init__(
        self,
        dataset: Mapping[int, T],
        batch_size: int | None = 1,
        shuffle: bool = False,
        sampler: Iterable[int] | None = None,
        batch_sampler: Iterable[list[int]] | None = None,
        num_workers: int = 4,
        collate_fn: Callable[[list[T]], U] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 30,
        generator=None,
        *,
        prefetch_factor: int = 4,
        pin_memory_device: str | None = None,
    ):
        if sampler is not None and batch_sampler is not None:
            raise ValueError(
                "`samplers` and `batch_sampler` cannot be specified at the same time."
            )
        if sampler is not None:
            if shuffle:
                raise ValueError(
                    "When `sampler` is provided, `shuffle` must be `False`."
                )
        if batch_sampler is not None:
            if shuffle:
                raise ValueError(
                    "When `batch_sampler` is provided, `shuffle` must be `False`."
                )
            if batch_size != 1:
                raise ValueError(
                    "When `batch_sampler` is provided, `batch_size` must be `None`."
                )
        if num_workers <= 0:
            raise ValueError("`num_workers` must be greater than 0.")
        if pin_memory:
            warnings.warn("`pin_memory` is not supported.")
        if prefetch_factor <= 0:
            raise ValueError("`prefetch_factor` must be greater than 0.")

        self.dataset = dataset

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.generator = generator
        self.prefetch_factor = prefetch_factor

    def __iter__(self):
        batch_sampler = self.batch_sampler or iter_batch(
            self.sampler or _get_sampler(self.dataset, self.shuffle, self.generator),
            self.batch_size,
            self.drop_last,
        )

        gen = _sample(self.dataset, batch_sampler, self.num_workers, self.collate_fn)

        bgg = BackgroundGenerator(
            gen,
            num_workers=self.num_workers,
            queue_size=self.prefetch_factor,
            timeout=self.timeout,
        )

        return iter(bgg)