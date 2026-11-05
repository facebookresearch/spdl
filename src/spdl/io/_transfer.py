# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "transfer_tensor_d2h",
    "transfer_tensor_h2d",
]

import logging
import os
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
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


@dataclass(eq=False)
class _IncompleteH2D:
    sources: "set[Tensor]"
    destinations: "set[Tensor]"
    stream: "torch.cuda.Stream"


_FAILED_H2D: list[_IncompleteH2D] = []
_FAILED_H2D_LOCK = threading.Lock()


class _DataTransfer:
    def __init__(self, device: "TDevice", num_caches: int) -> None:
        if num_caches < 0:
            raise ValueError("num_caches must be non-negative")
        self._device = device
        self._stream = torch.cuda.Stream(device)
        self._batch_cache: list[Any] = [None for _ in range(num_caches)]
        self._pending: _IncompleteH2D | None = None

    def _transfer(self, obj: T, pinned_memory_cache: "set[Tensor]") -> T:
        return _transfer(obj, self._device, pinned_memory_cache)

    def _drain_pending(self) -> None:
        pending = self._pending
        if pending is not None:
            self._stream.synchronize()
            with _FAILED_H2D_LOCK:
                _FAILED_H2D[:] = [
                    entry for entry in _FAILED_H2D if entry is not pending
                ]
            self._pending = None

    def _hold_incomplete(
        self, sources: "set[Tensor]", destinations: "set[Tensor]"
    ) -> None:
        pending = _IncompleteH2D(sources, destinations, self._stream)
        with _FAILED_H2D_LOCK:
            _FAILED_H2D.append(pending)
        self._pending = pending

    def __call__(self, batch: T) -> T:
        self._drain_pending()
        sources: set[Tensor] = set()
        destinations: set[Tensor] = set()

        def transfer(obj: S) -> S:
            result = self._transfer(obj, sources)
            if isinstance(result, torch.Tensor) and result.device == self._device:
                destinations.add(result)
            return result

        try:
            with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
                result = _recursive_apply(transfer, batch)
        except BaseException as traversal_error:
            try:
                self._stream.synchronize()
            except BaseException as sync_error:  # noqa: B036 - Primary error wins.
                self._hold_incomplete(sources, destinations)
                traversal_error.add_note(
                    "H2D synchronization failed; "
                    f"storage owners retained: {sync_error!r}"
                )
            raise
        try:
            self._stream.synchronize()
        except BaseException:
            self._hold_incomplete(sources, destinations)
            raise
        if destinations:
            default_stream = torch.cuda.default_stream(self._device)
            for destination in destinations:
                destination.record_stream(default_stream)
        self._batch_cache.append(result)
        self._batch_cache.pop(0)
        return result


_THREAD_LOCAL = threading.local()


def _get_transfer_func(num_caches: int) -> _DataTransfer:
    if num_caches < 0:
        raise ValueError("num_caches must be non-negative")
    device = _normalize_cuda_device(None)
    if (
        not hasattr(_THREAD_LOCAL, "transfer")
        or _THREAD_LOCAL.transfer._device != device
    ):
        _LG.info("Creating transfer stream on %s", device)
        _THREAD_LOCAL.transfer = _DataTransfer(device, num_caches=num_caches)
    return _THREAD_LOCAL.transfer


def transfer_tensor_h2d(batch: T, /, *, num_caches: int = 4) -> T:
    """Transfer PyTorch tensors from CPU to CUDA in a dedicated stream.

    .. versionadded:: 0.8.0

    .. versionchanged:: 0.9.0
       Recursive failures drain submitted copies before unwinding. Failed drains
       retain source/destination owners until completion is proved. Returned GPU
       destinations are recorded on the selected device's default consumer stream;
       callers using other streams retain normal readiness/recording responsibilities.

    This function wraps calls to :py:meth:`torch.Tensor.pin_memory` and
    :py:meth:`torch.Tensor.to`, and executes them in a dedicated CUDA stream.

    When called in a background thread, the data transfer overlaps with
    the GPU computation happening in the foreground thread (such as training
    and inference).

    .. seealso::

       :ref:`pipeline-parallelism-custom-mt` - An intended way to use
       this function in :py:class:`~spdl.pipeline.Pipeline`.

       :py:mod:`benchmark_pinned_arena` - Matched public pool/ring pinning
       comparisons through subprocess iteration and synchronized H2D wall time,
       with archival earlier measurements.

    .. image:: ../../_static/data/parallelism_transfer.png

    Concretely, it performs the following operations.

    1. If a dedicated CUDA stream local to the calling thread is not found
       in thread-local storage, creates and caches one.
       (The target device is determined by the ``"LOCAL_RANK"`` environment
       variable.)
    2. Activates the CUDA stream.
    3. Traverses the given object recursively and transfers CPU tensors to CUDA.
       ``pin_memory`` reuses source storage when Torch recognizes it as pinned;
       otherwise it creates a page-locked copy. CUDA transfer is submitted with
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


###############################################################################
# Device-to-host transfer
###############################################################################


class _AsyncD2HTransfer:
    """Async Device to Host transfer handler."""

    def __init__(self, device: "TDevice", stream: "torch.cuda.Stream") -> None:
        self._device = device
        self._stream = stream

    def __call__(self, batch: T, producer_event: "torch.cuda.Event") -> T:
        """Transfer batch from GPU to CPU asynchronously.

        Args:
            batch: A Tensor or composition of tensors to transfer.
            producer_event: Event marking when the batch is ready to copy.

        Returns:
            Batch with tensors on CPU.
        """
        transferred = False

        def _copy(obj: S) -> S:
            nonlocal transferred
            if not isinstance(obj, torch.Tensor) or obj.device != self._device:
                return obj
            if not transferred:
                self._stream.wait_event(producer_event)
            transferred = True
            return cast(S, obj.to("cpu", non_blocking=True))

        try:
            with torch.cuda.stream(self._stream):
                result = _recursive_apply(_copy, batch)
        except BaseException as traversal_error:
            if transferred:
                try:
                    self._stream.synchronize()
                except Exception as synchronize_error:
                    raise synchronize_error from traversal_error
            raise
        if not transferred:
            return batch

        self._stream.synchronize()
        return result


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
    producer_event: "torch.cuda.Event | None" = None,
) -> T:
    """Transfer PyTorch CUDA tensors to CPU through a dedicated stream.

    .. versionadded:: 0.8.0

    This function transfers tensors directly to CPU with
    :py:meth:`torch.Tensor.to` on a dedicated CUDA stream. Non-blocking D2H
    transfers produce page-locked (pinned) CPU tensors.

    The transfer process:
    1. Traverses the batch recursively.
    2. Asynchronously transfers each matching tensor directly to CPU.
    3. Synchronizes the transfer stream before returning the rebuilt batch.

    The returned CPU tensors remain page-locked. Retaining many transferred
    batches therefore retains the corresponding amount of pinned host memory.

    By default, the function records an event on the calling thread's current
    stream and makes the copy stream wait for that event. When dispatching the
    function to another CPU thread, record the event on the producer thread
    before dispatch and pass it as ``producer_event``. This captures the exact
    readiness boundary without waiting for later foreground work. The function
    waits for the copy stream before returning.

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

        producer_event: **Optional** Event marking when the input tensors are
            ready. If ``None``, an event is recorded on the calling thread's
            current stream. Callers that dispatch this function to another CPU
            thread must record and pass an event from the producer thread.

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
    if producer_event is None:
        producer_event = torch.cuda.Event()
        producer_event.record(torch.cuda.current_stream(device))
    transfer = _get_d2h_transfer(device, stream)
    return transfer(batch, producer_event)
