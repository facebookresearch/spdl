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
