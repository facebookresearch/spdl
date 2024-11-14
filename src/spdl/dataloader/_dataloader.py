# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["DataLoader", "MapIterator"]

from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from typing import Generic, TypeAlias, TypeVar

from spdl.pipeline import Pipeline, PipelineBuilder

# pyre-strict

Source = TypeVar("Source")
Output = TypeVar("Output")

T = TypeVar("T")
U = TypeVar("U")

K = TypeVar("K")
V = TypeVar("V")


Callables: TypeAlias = (
    Callable[[T], U]
    | Callable[[T], Iterable[U]]
    | Callable[[T], Awaitable[U]]
    | Callable[[T], AsyncIterable[U]]
)

Functions: TypeAlias = Callable[[T], U] | Callable[[T], Awaitable[U]]


class DataLoader(Generic[Source, Output]):
    """A data preprocessing pipeline composed of source, preprocessing and aggregation.

    It generates source items, preprocess them concurrently, and aggergates them and store
    the result in buffer.

    .. seealso::

       - :py:class:`~spdl.pipeline.Pipeline`

    .. code-block::

       ┌────────┐
       │ Source │
       └─┬──────┘
         │
       ┌─▼─────────────────────┐
       │ Preprocessing         │─┐
       │                       │ │─┐
       │ fn: Source -> T       │ │ │
       │                       │ │ │
       └──┬────────────────────┘ │ │
         │└──┬───────────────────┘ │
         │   └─────────────────────┘
         │
       ┌─▼─────────────────────┐
       │ Aggregate             │
       │                       │
       │ fn: list[T] -> Output │
       │                       │
       └─┬─────────────────────┘
         │
       ┌─▼──────┐
       │ Buffer │
       └────────┘

    Args:
        src: Data source. An object impelements :py:class:`~collections.abc.Iterable` interface
            or :py:class:`~collections.abc.AsyncIterable` interface.
            Typically, a generator that yields file paths or URLs.
            To iterate over an object that implements :py:class:`~collections.abc.Mapping`,
            use :py:class:`MapIterator`.

        preprocessor: A [async] function or [async] generator that process the individual data
            source. Often times it loads data into array format.

        aggregator: A [async] function that takes a set of preprocessed data and
            returns one item. Typically, batching and GPU transfer.

        batch_size: The number of items to aggregate before it's passed to the aggregator.

        drop_last: If ``True`` and the number of source items are not divisible by
            ``batch_size``, then drop the reminder.

        buffer_size: The number of aggregated items to buffer.

        num_threads: The number of worker threads.

        timeout: The timeout until the next item becomes available.
            Default behavior is to wait indefinitely.
    """

    def __init__(
        self,
        src: Iterable[Source] | AsyncIterable[Source],
        *,
        preprocessor: Callables[[Source], T] | None = None,
        batch_size: int | None = None,
        drop_last: bool = False,
        aggregator: Functions[[list[T]], Output] | None = None,
        buffer_size: int = 3,
        num_threads: int = 8,
        timeout: float | None = None,
        output_order: str = "completion",
    ) -> None:
        self.src = src
        self.preprocessor = preprocessor
        self.aggregator = aggregator

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.num_threads = num_threads
        self.timeout = timeout
        self.output_order = output_order

    def _get_pipeline(self) -> Pipeline:
        builder = PipelineBuilder()
        builder.add_source(self.src)

        if self.preprocessor:
            builder.pipe(
                self.preprocessor,
                concurrency=self.num_threads,
                output_order=self.output_order,
            )

        if self.batch_size:
            builder.aggregate(self.batch_size, drop_last=self.drop_last)

        if self.aggregator:
            builder.pipe(self.aggregator)

        builder.add_sink(self.buffer_size)

        return builder.build(num_threads=self.num_threads)

    def __iter__(self) -> Iterable[Output]:
        pipeline = self._get_pipeline()

        with pipeline.auto_stop():
            for item in pipeline.get_iterator(timeout=self.timeout):
                yield item


class MapIterator(Iterable[V]):
    """Combine Mapping object and iterable to iterate over mapped objects

    Args:
        mapping: Object implements :py:class:`~collections.abc.Mapping` interface.
        sampler: **Optional** Generator that yields key for the mapping.
            Used to specify the iteratoin order over the mapping and/or to sample
            from a subset of the mapping.

    Example:
        >>> mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        >>> for item in MapIterator(mapping):
        ...    print(item)
        ...
        a
        b
        c
        d
        e
        >>> sampler = range(4, -2, -1)
        >>> for item in MapIterator(mapping, sampler):
        ...    print(item)
        ...
        e
        c
        a
    """

    def __init__(
        self,
        mapping: Mapping[K, V],
        sampler: Iterable[K] | None = None,
    ) -> None:
        self.mapping = mapping
        self.sampler = sampler

    def __iter__(self) -> Iterator[V]:
        for key in self.sampler or self.mapping:
            yield self.mapping[key]
