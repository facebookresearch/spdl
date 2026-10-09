# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "transfer_tensor",
    "transfer_tensor_d2h",
    "transfer_tensor_h2d",
]

import logging
import os
import threading
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields, is_dataclass
from functools import partial
from types import ModuleType
from typing import Any, cast, TYPE_CHECKING, TypeVar

from ._internal import import_utils

if TYPE_CHECKING:
    import torch
    from torch import device as TDevice, Tensor

else:
    torch: ModuleType = import_utils.lazy_import("torch")

_LG: logging.Logger = logging.getLogger(__name__)


T = TypeVar("T")
S = TypeVar("S")


def _recursive_apply(fn: Callable[[T], T], obj: T) -> T:
    """Recursively apply the given function to the given (container) object.

    Args:
        fn: The function to apply.
        obj: The object to which the function is applied.

    Returns:
        The result of applying the function.
    """
    Class = type(obj)
    match obj:
        case list():
            return Class(_recursive_apply(fn, v) for v in obj)
        case tuple():
            if hasattr(obj, "_asdict") and hasattr(obj, "_fields"):  # namedtuple
                # pyrefly: ignore [bad-unpacking]
                return Class(**_recursive_apply(fn, obj._asdict()))
            return Class(_recursive_apply(fn, v) for v in obj)
        case defaultdict():
            return Class(
                obj.default_factory,
                {k: _recursive_apply(fn, v) for k, v in obj.items()},
            )
        case Mapping():
            return Class({k: _recursive_apply(fn, v) for k, v in obj.items()})
        case obj if is_dataclass(obj) and not isinstance(obj, type):
            new_obj = Class(
                **{
                    field.name: _recursive_apply(fn, getattr(obj, field.name))
                    for field in fields(obj)
                    if field.init
                }
            )
            for field in fields(obj):
                if not field.init:
                    val = _recursive_apply(fn, getattr(obj, field.name))
                    setattr(new_obj, field.name, val)
            return new_obj

        case _:
            return fn(obj)  # pyre-ignore: [6]


def _transfer(obj: T, device: "TDevice", pinned_memory_cache: "set[Tensor]") -> T:
    if isinstance(obj, torch.Tensor) and obj.is_cpu:
        pinned = obj.pin_memory()
        pinned_memory_cache.add(pinned)
        # pyrefly: ignore [bad-assignment]
        obj = pinned.to(device, non_blocking=True)
    return obj


class _DataTransfer:
    def __init__(self, device: "TDevice", num_caches: int) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device)
        self._batch_cache: list[Any] = [None for _ in range(num_caches)]

    def _transfer(self, obj: T, pinned_memory_cache: "set[Tensor]") -> T:
        return _transfer(obj, self._device, pinned_memory_cache)

    def __call__(self, batch: T) -> T:
        pinned_memory_cache = set()
        fn = partial(self._transfer, pinned_memory_cache=pinned_memory_cache)
        with torch.cuda.stream(self._stream):
            batch = _recursive_apply(fn, batch)
        self._stream.synchronize()
        self._batch_cache.append(batch)
        self._batch_cache.pop(0)
        return batch


_THREAD_LOCAL = threading.local()


def _get_transfer_func(num_caches: int) -> _DataTransfer:
    if not hasattr(_THREAD_LOCAL, "transfer"):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
        device = torch.device(f"cuda:{local_rank}")
        _LG.info("Creating transfer stream on %s", device)
        _THREAD_LOCAL.transfer = _DataTransfer(  # pyre-ignore: [16]
            device, num_caches=num_caches
        )
    return _THREAD_LOCAL.transfer


def transfer_tensor_h2d(batch: T, /, *, num_caches: int = 4) -> T:
    """Transfer PyTorch tensors from CPU to CUDA in a dedicated stream.

    .. versionadded:: 0.7.0

    This function wraps calls to :py:meth:`torch.Tensor.pin_memory` and
    :py:meth:`torch.Tensor.to`, and executes them in a dedicated CUDA stream.

    When called in a background thread, the data transfer overlaps with
    the GPU computation happening in the foreground thread (such as training
    and inference).

    .. seealso::

       :ref:`pipeline-parallelism-custom-mt` - An intended way to use
       this function in :py:class:`~spdl.pipeline.Pipeline`.

    .. image:: ../../_static/data/parallelism_transfer.png

    Concretely, it performs the following operations.

    1. If a dedicated CUDA stream local to the calling thread is not found
       in thread-local storage, creates and caches one.
       (The target device is determined by the ``"LOCAL_RANK"`` environment
       variable.)
    2. Activates the CUDA stream.
    3. Traverses the given object recursively and transfers CPU tensors to CUDA.
       Data is first copied to page-locked memory by calling ``pin_memory``,
       then transferred to CUDA asynchronously with
       ``.to(non_blocking=True)``.
    4. Synchronizes the stream, to ensure that all the data transfers are
       completed.

    Args:
        batch: A :py:class:`torch.Tensor` or a composition of tensors
            with container types such as ``list``, ``tuple``, ``dict``
            and ``dataclass``.

        num_caches: Number of batch caches to maintain the reference to.
            This parameter helps mitigate race conditions when using
            multi-threading with multiple CUDA streams.

            See :doc:`../notes/pytorch_cuda_race_condition`
            for details on the rationale behind this parameter.

    Returns:
        An object of the same type as the input, but the PyTorch
        tensors are transferred to CUDA device.

    Example:
        .. code-block:: python

           from concurrent.futures import ThreadPoolExecutor

           from spdl.io import transfer_tensor_h2d

           with ThreadPoolExecutor(max_workers=1) as executor:
               future = executor.submit(transfer_tensor_h2d, cpu_batch)
               cuda_batch = future.result()
    """
    transfer = _get_transfer_func(num_caches)
    return transfer(batch)


def transfer_tensor(batch: T, /, *, num_caches: int = 4) -> T:
    """Transfer PyTorch tensors from CPU to CUDA in a dedicated stream.

    .. deprecated:: 0.7.0
       Use :py:func:`spdl.io.transfer_tensor_h2d` instead.

    Args:
        batch: A :py:class:`torch.Tensor` or a composition of tensors with
            container types such as ``list``, ``tuple``, ``dict`` and
            ``dataclass``.
        num_caches: Number of recent output batches retained to mitigate
            allocator races across CUDA streams.

    Returns:
        An object of the same type as the input, with CPU tensors transferred
        to CUDA.
    """
    warnings.warn(
        "transfer_tensor is deprecated; use transfer_tensor_h2d instead.",
        FutureWarning,
        stacklevel=2,
    )
    return transfer_tensor_h2d(batch, num_caches=num_caches)


###############################################################################
# Device-to-host transfer
###############################################################################


def _iter_leaves(obj: Any) -> Iterator[Any]:
    """Iterate over leaves without rebuilding the input containers."""
    match obj:
        case list() | tuple():
            for value in obj:
                yield from _iter_leaves(value)
        case Mapping():
            for value in obj.values():
                yield from _iter_leaves(value)
        case value if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                if field.init:
                    yield from _iter_leaves(getattr(value, field.name))
            for field in fields(value):
                if not field.init:
                    yield from _iter_leaves(getattr(value, field.name))
        case _:
            yield obj


def _gather_tensors(batch: T, device: "TDevice") -> "list[Tensor]":
    """Gather tensors from a batch.

    Args:
        batch: A Tensor or a composition of tensors with container types.
        device: The device to which the tensors are transferred.

    Returns:
        A list of all tensors in the batch.
    """
    return [
        obj
        for obj in _iter_leaves(batch)
        if isinstance(obj, torch.Tensor) and obj.device == device
    ]


def _get_tensor_offsets(tensors: "list[Tensor]") -> tuple[list[int], int]:
    """Return byte-aligned offsets and the total packed-buffer size."""
    offsets = []
    size = 0
    for tensor in tensors:
        alignment = tensor.element_size()
        size = (size + alignment - 1) // alignment * alignment
        offsets.append(size)
        size += tensor.nbytes
    return offsets, size


def _get_pinned_memory(size: int) -> "Tensor":
    """Allocate page-locked memory or fetch a cache.

    Args:
        size: Minimum size in bytes required.

    Returns:
        Pinned memory tensor (uint8).
    """
    if (
        not hasattr(_THREAD_LOCAL, "d2h_pinned_memory")
        or _THREAD_LOCAL.d2h_pinned_memory.numel() < size
    ):
        _THREAD_LOCAL.d2h_pinned_memory = torch.empty(
            size, dtype=torch.uint8
        ).pin_memory()
    return _THREAD_LOCAL.d2h_pinned_memory[:size]


def _transfer_with_autograd(
    batch: T,
    source_device: "TDevice",
    destination_device: "TDevice",
) -> T:
    """Transfer tensors individually so that PyTorch preserves autograd edges."""

    def _copy(obj: S) -> S:
        if isinstance(obj, torch.Tensor) and obj.device == source_device:
            return cast(S, obj.to(destination_device))
        return obj

    return _recursive_apply(_copy, batch)


def _rebuild_batch(
    batch: T,
    source_device: "TDevice",
    buffer: "Tensor",
    offsets: list[int],
) -> T:
    """Rebuild a batch with tensors represented by views into a packed buffer.

    Args:
        batch: Original batch structure.
        source_device: Device identifying tensors to replace.
        buffer: Packed destination buffer.
        offsets: Byte-aligned offset for each replaced tensor.

    Returns:
        New batch with tensors that share ``buffer`` storage but have independent
        TensorImpls and version counters.
    """
    index = 0

    def _rebuild(obj: S) -> S:
        if isinstance(obj, torch.Tensor) and obj.device == source_device:
            nonlocal index
            offset = offsets[index]
            index += 1
            strides = []
            stride = 1
            for dimension in reversed(obj.shape):
                strides.append(stride)
                stride *= max(dimension, 1)
            view = torch.empty(0, dtype=obj.dtype, device=buffer.device)
            return cast(
                S,
                view.set_(
                    buffer.untyped_storage(),
                    offset // obj.element_size(),
                    obj.shape,
                    tuple(reversed(strides)),
                ),
            )
        return obj

    return _recursive_apply(_rebuild, batch)


class _AsyncD2HTransfer:
    """Async Device to Host transfer handler."""

    def __init__(self, device: "TDevice", stream: "torch.cuda.Stream") -> None:
        self._device = device
        self._stream = stream

    def __call__(self, batch: T) -> T:
        """Transfer batch from GPU to CPU asynchronously.

        Args:
            batch: A Tensor or composition of tensors to transfer.

        Returns:
            Batch with tensors on CPU.
        """
        tensors = _gather_tensors(batch, self._device)
        if not tensors:
            return batch

        if any(tensor.requires_grad for tensor in tensors):
            return _transfer_with_autograd(
                batch,
                self._device,
                torch.device("cpu"),
            )

        offsets, size = _get_tensor_offsets(tensors)
        if size > 0:
            pinned = _get_pinned_memory(size)
            producer_stream = torch.cuda.current_stream(self._device)
            if producer_stream != self._stream:
                self._stream.wait_stream(producer_stream)
            with torch.cuda.stream(self._stream):
                for tensor, offset in zip(tensors, offsets, strict=True):
                    if tensor.nbytes > 0:
                        destination = (
                            pinned[offset : offset + tensor.nbytes]
                            .view(tensor.dtype)
                            .view(tensor.shape)
                        )
                        destination.copy_(tensor, non_blocking=True)
            self._stream.synchronize()
            cpu_buffer = pinned.clone()
        else:
            cpu_buffer = torch.empty(0, dtype=torch.uint8)

        return _rebuild_batch(
            batch,
            self._device,
            cpu_buffer,
            offsets,
        )


def _normalize_cuda_device(device: "TDevice | str | None") -> "TDevice":
    """Resolve a CUDA device to one with an explicit index."""
    if device is None:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    else:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"Expected a CUDA device, but received {device}.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

    device_count = torch.cuda.device_count()
    if device.index is None or not 0 <= device.index < device_count:
        raise RuntimeError(
            f"CUDA device index {device.index} is invalid for {device_count} devices."
        )
    return device


def _get_d2h_transfer(
    device: "TDevice",
    stream: "torch.cuda.Stream | None" = None,
) -> _AsyncD2HTransfer:
    """Get thread-local D2H transfer handler.

    Args:
        device: CUDA device to use for the transfer.
        stream: Optional CUDA stream to use. If None, uses a cached handler
            with a thread-local stream.

    Returns:
        D2H transfer handler.
    """
    if stream is not None:
        return _AsyncD2HTransfer(device, stream)

    if not hasattr(_THREAD_LOCAL, "d2h_transfer"):
        _THREAD_LOCAL.d2h_transfer = {}
    transfers: dict[Any, _AsyncD2HTransfer] = _THREAD_LOCAL.d2h_transfer
    if device not in transfers:
        _LG.info("Creating D2H transfer handler on %s", device)
        stream = torch.cuda.Stream(device)
        transfers[device] = _AsyncD2HTransfer(device, stream)

    return transfers[device]


def transfer_tensor_d2h(
    batch: T,
    /,
    *,
    device: "TDevice | str | None" = None,
    stream: "torch.cuda.Stream | None" = None,
) -> T:
    """Transfer PyTorch CUDA tensors to CPU through a dedicated stream.

    .. versionadded:: 0.7.0

    This function performs efficient GPU to CPU data transfer using
    page-locked (pinned) memory and a dedicated CUDA stream. The page-locked
    memory is cached and reused across calls.

    The transfer process:
    1. Gathers all tensors from the batch.
    2. Allocates (or reuses cached) page-locked memory.
    3. Asynchronously transfers data from GPU to page-locked memory.
    4. Copies data from page-locked memory to new CPU tensors.
    5. Rebuilds the batch structure with CPU tensors.

    The copy stream waits for work already submitted to the caller's current
    stream, and this function waits for the copy stream before returning. If a
    tensor was produced on another non-current stream, the caller must first
    establish an ordering dependency with the current stream. When called from
    a background CPU thread, the transfer can overlap with later GPU work
    submitted independently by a foreground thread. It is intended for
    offloading nested results before CPU post-processing or serialization.

    If any transferred tensor requires gradients, the function uses a
    synchronous per-tensor transfer on the caller's current stream to preserve
    autograd.

    Example:
        .. code-block:: python

           import torch
           from spdl.io import transfer_tensor_d2h

           batch = {"scores": torch.randn(32, device="cuda:0")}
           cpu_batch = transfer_tensor_d2h(batch, device="cuda:0")

    Args:
        batch: A :py:class:`torch.Tensor` or a composition of tensors
            with container types such as ``list``, ``tuple``, ``dict``
            and ``dataclass``.

        device: **Optional** CUDA device to transfer data from.

            If ``None`` the source device is determined by the ``LOCAL_RANK``
            environment variable. If not set, ``cuda:0`` is used.

        stream: **Optional** Custom CUDA stream to use for the transfer.
            If ``None``, a stream is created from the ``device`` argument,
            and cached to a thread-local storage for future reuse.

            When stream is not ``None``, the ``device`` argument must be provided.
            The stream must be on the same device.

    Returns:
        An object of the same type as the input, but the PyTorch CUDA tensors
        on the specified device are transferred to CPU.

        If there is no PyTorch tensor in the input, the input is returned as-is.

        If there is no CUDA device available, the input is returned as-is.

    Raises:
        ValueError: If ``device`` is not a CUDA device, or a custom stream does
            not match the requested device.
        RuntimeError: If the resolved CUDA device index is unavailable.
    """
    if stream is not None and device is None:
        raise ValueError("device must be provided when stream is not None")

    if not torch.cuda.is_available():
        return batch

    device = _normalize_cuda_device(device)
    if stream is not None and stream.device != device:
        raise ValueError(
            f"The transfer stream is on {stream.device}, not the requested {device}."
        )
    transfer = _get_d2h_transfer(device, stream)
    return transfer(batch)
