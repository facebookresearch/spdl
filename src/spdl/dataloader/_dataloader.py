# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["DataLoader", "MapIterator", "MergeIterator"]

import random
import sys
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
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
            To iterate over an object that implements :py:class:`~collections.abc.Mapping`
            protocol, and optionally with sampling, use :py:class:`MapIterator`.

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

        output_order: If 'completion' (default), items processed by the preprocessor are
            passed to the aggregator in order of completion. If 'input', then they are passed
            to the aggregator in the order of the source input.

    :ivar src: The source object provided in the constructor.

    Exapmles:
        >>> import spdl.io
        >>> from spdl.io import ImageFrames
        >>>
        >>> import torch
        >>> from torch import Tensor
        >>>
        >>> ##################################################################
        >>> # Source
        >>> ##################################################################
        >>> def source(root_dir: str) -> Iterable[str]:
        ...     # Iterate the directory and find images.
        ...     yield from glob.iglob(f"{root_dir}/**/*.JPEG", recursive=True)
        >>>
        >>>
        >>> ##################################################################
        >>> # Preprocessor
        >>> ##################################################################
        >>> width, height, batch_size = 224, 224, 32
        >>>
        >>> # Filter description that scales the image and convert to RGB
        >>> filter_desc = spdl.io.get_filter_desc(
        ...     scale_width=width,
        ...     scale_height=height,
        ...     pix_fmt="rgb24"
        ... )
        >>>
        >>> def decode_image(path: str) -> ImageFrames:
        ...     # Decode image and resize
        ...     packets = spdl.io.demux_image(path)
        ...     return spdl.io.decode_packets(packets, filter_desc=filter_desc)
        ...
        >>>
        >>> ##################################################################
        >>> # Aggregator (batch + device transfer)
        >>> ##################################################################
        >>> cuda_device_index = 0
        >>> size = width * height * batch_size * 3
        >>> storage = spdl.io.cpu_storage(size, pin_memory=True)
        >>> stream = torch.cuda.Stream(device=cuda_device_index)
        >>>
        >>> cuda_config = spdl.io.cuda_config(
        ...     device_index=cuda_device_index,
        ...     stream=stream.cuda_stream,
        ... )
        >>>
        >>> def batchify(data: list[ImageFrames]) -> Tensor:
        ...     # Merge the decoded frames into the pre-allocated pinned-memory.
        ...     cpu_buffer = spdl.io.convert_frames(data, storage=storage)
        ...     # Send to CUDA in a separate stream.
        ...     cuda_buffer = spdl.io.transfer_buffer(cpu_buffer, cuda_config=cuda_config)
        ...     # Cast to Torch Tensor type.
        ...     return spdl.io.to_torch(cuda_buffer)
        ...
        >>>
        >>> dataloader = DataLoader(
        ...     src=source(root_dir),
        ...     preprocessor=decode_image,
        ...     batch_size=batch_size,
        ...     aggregator=batchify,
        ... )
        >>>
        >>> for batch in dataloader:
        ...     ...
        >>>

    .. seealso::

       - :py:class:`spdl.pipeline.Pipeline`: The abstraction used for executing the logics.
       - :py:func:`spdl.io.demux_image`, :py:func:`spdl.io.decode_packets`: Decoding image.
       - :py:func:`spdl.io.cpu_storage`: Allocate page-locked memory.
       - :py:func:`spdl.io.convert_frames`: Merging the decoded frames into pre-allocated memory
         without creating intermediate arrays.
       - :py:func:`spdl.io.transfer_buffer`: Sending the data to GPU.
       - :py:func:`spdl.io.to_torch`, :py:func:`spdl.io.to_numba`, :py:func:`spdl.io.to_jax`: Casting
         the memroy buffer to array type.

    """

    def __init__(
        self,
        src: Iterable[Source] | AsyncIterable[Source],
        *,
        # Pre-processing
        preprocessor: Callables[[Source], T] | None = None,
        # Aggregation
        batch_size: int | None = None,
        drop_last: bool = False,
        aggregator: Functions[[list[T]], Output] | None = None,
        # Buffering
        buffer_size: int = 3,
        # Execution config
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
        """Run the data loading pipeline in background.

        Yields:
            The items processed by processor and aggregator.
        """
        pipeline = self._get_pipeline()

        with pipeline.auto_stop():
            for item in pipeline.get_iterator(timeout=self.timeout):
                yield item


class MapIterator(Iterable[V]):
    """Combine Mapping object and iterable to iterate over mapped objects

    Args:
        mapping: Object implements :py:class:`~collections.abc.Mapping` interface.
        sampler: **Optional** Generator that yields key for the mapping.
            Used to specify the iteration order over the mapping and/or to sample
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


_FIRST_EXHAUSTION = 0


def _ordered_iter(iterators: list[Iterator[T]], stop_after: float) -> Iterable[T]:
    num_items = 0
    while iterators:
        remove = []

        for i, iterator in enumerate(iterators):
            try:
                yield next(iterator)
            except StopIteration:
                if stop_after == _FIRST_EXHAUSTION:
                    return
                # Insert in reversed order beacause we use this for popping from list
                remove.insert(0, i)
                continue

            num_items += 1
            if stop_after > 0 and num_items >= stop_after:
                return

        if remove:
            for i in remove:
                iterators.pop(i)


def _stocastic_iter(
    iterators: list[Iterator[T]],
    weights: Sequence[float],
    stop_after: float,
    seed: int,
) -> Iterable[T]:
    # These are all checked in MergeIterator constructor
    assert len(iterators) == len(weights)
    assert all(w >= sys.float_info.epsilon for w in weights)

    population = list(range(len(iterators)))
    rng = random.Random(seed)
    num_items = 0

    not_exhausted = [True for _ in range(len(iterators))]
    while any(not_exhausted):
        for i in rng.choices(population, weights, k=100):
            try:
                yield next(iterators[i])
            except StopIteration:
                not_exhausted[i] = False
                if stop_after == _FIRST_EXHAUSTION:
                    return
                continue

            num_items += 1
            if stop_after > 0 and num_items >= stop_after:
                return


class MergeIterator(Iterable[T]):
    """Iterate over given iterables and yield one item from each iterator.


    Args:
        iterables: The source iterables
        probabilities: The probability to choose the next iterable.
            If not provided, the given iterables are visited in the given order
            repeatedly.
        stop_after: Determines the stop criteria or the behavior when one of
            the input iterables gets exhausted,
            Available values are;

            - ``0``: The iteration stops when one of the iterator is exhausted.
            - ``n > 0``: The iteration stops when the specified number of items
              are yielded or all the input iterables are exhausted.
            - ``-1``: The iteration continues until all the input iterables are
              exhausted.
        seed: Used to seed the random generator when probabilities is provided.

    Example:

        >>> iterables = [
        ...     [0, 1, 2],
        ...     [10, 11, 12],
        ...     [20, 21, 22],
        ... ]
        >>>
        >>> print(list(MergeIterator(iterables)))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>>
        >>> # By default, it stops after one iterable gets exhausted.
        >>> iterables = [
        ...     [0, 1, 2],
        ...     [10, 11],
        ...     [20, 21, 22],
        ... ]
        >>>
        >>> print(list(MergeIterator(iterables)))
        [0, 10, 20, 1, 11, 21, 2]  # 22 is not included
        >>>
        >>> # Stop after yielding the given number of items
        >>> print(list(MergeIterator(iterables, stop_after=5)))
        [0, 10, 20, 1, 11]
        >>>
        >>> # stop_after>1 ignores the exhaustion.
        >>> print(list(MergeIterator(iterables, stop_after=9)))
        [0, 10, 20, 1, 11, 21, 2, 22]
        >>>
        >>> # Providing weights will pick up the iterable stocastically.
        >>> print(list(MergeIterator(iterables, stop_after=9, weights=[1, 1, 1])))
        [0, 1, 10, 11, 20, 2, 21, 22]
    """

    def __init__(
        self,
        iterables: Sequence[Iterable[T]],
        *,
        weights: Sequence[float] | None = None,
        stop_after: int = _FIRST_EXHAUSTION,
        seed: int = 0,
    ) -> None:
        if not iterables:
            raise ValueError("iterables cannot be empty.")

        if weights is not None:
            if len(weights) != len(iterables):
                raise ValueError(
                    f"The number of probabilities ({len(weights)}) and "
                    f"iterables ({len(iterables)}) must match."
                )

            # If any of them is 0 or negative, then there is something wrong with
            # user logic, so we raise an exception.
            if any(w < sys.float_info.epsilon for w in weights):
                raise ValueError("Weights must be non-zero and positive.")

        if not stop_after >= -1:
            msg = (
                f"`stop_after` must be greater than or equal to -1. Found: {stop_after}"
            )
            raise ValueError(msg)

        self.iterables = iterables
        self.weights = weights
        self.stop_after = stop_after
        self.seed = seed

    def __iter__(self) -> Iterator[T]:
        iterators = [iter(ite) for ite in self.iterables]

        if self.weights is None:
            yield from _ordered_iter(iterators, self.stop_after)
        else:
            yield from _stocastic_iter(
                iterators, self.weights, self.stop_after, self.seed
            )
