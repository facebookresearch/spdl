# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "transfer_tensor",
]

import logging
import os
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

from ._internal import import_utils

if TYPE_CHECKING:
    import torch
    from torch import device as TDevice
else:
    torch: ModuleType = import_utils.lazy_import("torch")
    TDevice = object

_LG: logging.Logger = logging.getLogger(__name__)


T = TypeVar("T")
S = TypeVar("S")

# pyre-strict


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


def _transfer(obj: T, device: TDevice) -> T:
    if isinstance(obj, torch.Tensor):
        obj = obj.pin_memory().to(device, non_blocking=True)
    return obj


class _DataTransfer:
    def __init__(self, device: TDevice) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device)

    def _transfer(self, obj: T) -> T:
        return _transfer(obj, self._device)

    def __call__(self, batch: T) -> T:
        with torch.cuda.stream(self._stream):
            batch = _recursive_apply(self._transfer, batch)
        self._stream.synchronize()
        return batch


_THREAD_LOCAL = threading.local()


def _get_trancfer_func() -> _DataTransfer:
    if not hasattr(_THREAD_LOCAL, "transfer"):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
        device = torch.device(f"cuda:{local_rank}")
        _LG.info("Creating transfer stream on %s", device)
        _THREAD_LOCAL.transfer = _DataTransfer(device)  # pyre-ignore: [16]
    return _THREAD_LOCAL.transfer


def transfer_tensor(batch: T, /) -> T:
    """Transfers PyTorch CPU Tensors to CUDA in a dedicated stream.

    This function wraps calls to :py:meth:`torch.Tensor.pin_memory` and
    :py:meth:`torch.Tensor.to`, and execute them in a dedicated CUDA stream.

    When called in a background thread, the data transfer overlaps with
    the GPU computation happening in the foreground thread (such as training
    and inference).

    .. seealso::

       :ref:`pipeline-parallelism-custom-mt` - An intended way to use
       this function in :py:class:`~spdl.pipeline.Pipeline`.

    .. image:: ../../_static/data/parallelism_transfer.png

    Concretely, it performs the following operations.

    1. If a dedicated CUDA stream local to the calling thread is not found
       in a thread-local storage, creates and stashes one.
       (The target device is deetrmined by ``"LOCAL_RANK"`` environment
       variable.)
    2. Activates the CUDA stream.
    3. Traverses the given object recursively, and transfer tensors to GPU.
       Data are first copied to page-locked memory by calling ``pin_memory``
       method, then the data is transferred to the GPU in asynchronous manner.
       (i.e. ``.to(non_blocking=True)``)
    4. Synchronizes the stream, to ensure that all the data transfers are
       completed.

    Args:
        batch: A :py:class:`Torch.Tensor` or a composition of tensors
            with container types such as ``list``, ``tuple``, ``dict``
            and ``dataclass``.

    Returns:
        An object of the same type as the input, but the PyTorch
        tensors are transferred to CUDA device.
    """
    transfer = _get_trancfer_func()
    return transfer(batch)
