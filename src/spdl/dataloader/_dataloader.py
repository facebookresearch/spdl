# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["DataLoader"]

from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
)
from typing import Generic, TypeAlias, TypeVar

from spdl._internal import log_api_usage_once
from spdl.pipeline import Pipeline, PipelineBuilder

# pyre-strict

Source = TypeVar("Source")
Output = TypeVar("Output")

T = TypeVar("T")
U = TypeVar("U")


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
       │ Aggregate             │─┐
       │                       │ │─┐
       │ fn: list[T] -> Output │ │ │
       │                       │ │ │
       └──┬────────────────────┘ │ │
         │└──┬───────────────────┘ │
         │   └─────────────────────┘
         │
       ┌─▼──────────────────────┐
       │ Transfer               │
       │                        │  * No concurrency, as GPUs do not support
       │ fn: Output -> Output   │    transferring multiple data concurrently.
       │                        │
       └─┬──────────────────────┘
         │
       ┌─▼──────┐
       │ Buffer │
       └────────┘

    Args:
        src: Data source. An object implements :py:class:`~collections.abc.Iterable` interface
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

        transfer_fn: A function applied to the output of aggregator function.
            It is intended for transferring data to GPU devices.
            Since GPU device transfer does not support concurrent transferring,
            this function is executed in a single thread.

        buffer_size: The number of aggregated items to buffer.

        num_threads: The number of worker threads.

        timeout: The timeout until the next item becomes available.
            Default behavior is to wait indefinitely.

        output_order: If 'completion' (default), items processed by the preprocessor are
            passed to the aggregator in order of completion. If 'input', then they are passed
            to the aggregator in the order of the source input.

    Examples:
        >>> import spdl.io
        >>> from spdl.io import CPUBuffer, CUDABuffer, ImageFrames
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
        >>> # Aggregator
        >>> ##################################################################
        >>> size = width * height * batch_size * 3
        >>> storage = spdl.io.cpu_storage(size, pin_memory=True)
        >>>
        >>> def batchify(data: list[ImageFrames]) -> Tensor:
        ...     # Merge the decoded frames into the pre-allocated pinned-memory.
        ...     return spdl.io.convert_frames(data, storage=storage)
        ...
        >>>
        >>> ##################################################################
        >>> # Transfer
        >>> ##################################################################
        >>> cuda_device_index = 0
        >>> stream = torch.cuda.Stream(device=cuda_device_index)
        >>> cuda_config = spdl.io.cuda_config(
        ...     device_index=cuda_device_index,
        ...     stream=stream.cuda_stream,
        ... )
        >>>
        >>> def transfer(cpu_buffer: CPUBuffer) -> CUDABuffer:
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
        ...     transfer_fn=transfer,
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
         the memory buffer to array type.

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
        # Device transfer
        transfer_fn: Callable[[Output], Output] | None = None,
        # Buffering
        buffer_size: int = 3,
        # Execution config
        num_threads: int = 8,
        timeout: float | None = None,
        output_order: str = "completion",
    ) -> None:
        log_api_usage_once("spdl.dataloader.DataLoader")

        self._src = src
        self._preprocessor = preprocessor
        self._aggregator = aggregator
        self._transfer_fn = transfer_fn

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._buffer_size = buffer_size
        self._num_threads = num_threads
        self._timeout = timeout
        self._output_order = output_order

    def _get_pipeline(self) -> Pipeline:
        builder = PipelineBuilder().add_source(self._src)
        if self._preprocessor:
            builder.pipe(
                self._preprocessor,
                concurrency=self._num_threads,
                output_order=self._output_order,
            )

        if self._batch_size:
            builder.aggregate(self._batch_size, drop_last=self._drop_last)

        if self._aggregator:
            builder.pipe(
                self._aggregator,
                concurrency=self._num_threads,
                output_order=self._output_order,
            )

        # Transfer runs in the default thread pool (with num_threads=1)
        # because GPU data transfer cannot be parallelized.
        # Note: this thread pool is also used by aggregate and disaggregate.
        if self._transfer_fn is not None:
            builder.pipe(self._transfer_fn)

        builder.add_sink(self._buffer_size)

        return builder.build(num_threads=self._num_threads)

    def __iter__(self) -> Iterable[Output]:
        """Run the data loading pipeline in background.

        Yields:
            The items processed by processor and aggregator.
        """
        pipeline = self._get_pipeline()

        with pipeline.auto_stop():
            for item in pipeline.get_iterator(timeout=self._timeout):
                yield item
