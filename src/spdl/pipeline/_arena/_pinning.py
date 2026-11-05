# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Consumer-local CUDA registration and owning pinned copy-out buffers."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from types import ModuleType
from typing import cast

from spdl._internal.import_utils import lazy_import

__all__ = [
    "_HostRegistration",
    "_allocate_pinned",
    "_check_host_budget",
    "_local_device",
]

torch: ModuleType = lazy_import("torch")


def _local_device() -> object:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    count = torch.cuda.device_count()
    if not 0 <= rank < count:
        raise RuntimeError(f"CUDA device index {rank} is invalid for {count} devices.")
    if torch.cuda._is_in_bad_fork():
        raise RuntimeError(
            "Pinned arenas cannot initialize CUDA after fork; use spawn."
        )
    return torch.device("cuda", rank)


def _check_host_budget(nbytes: int, *, shared: bool) -> None:
    """Conservatively check Linux tmpfs and available host/cgroup-v2 memory."""
    if shared and Path("/dev/shm").is_dir():
        stats = os.statvfs("/dev/shm")
        free = stats.f_bavail * stats.f_frsize
        if nbytes > free:
            raise BufferError(
                f"Arena requires {nbytes} bytes but tmpfs has {free} free."
            )
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                if nbytes > available:
                    raise BufferError(
                        f"Pinning requires {nbytes} bytes but host has {available} available."
                    )
                break
    groups = Path("/proc/self/cgroup")
    if groups.is_file():
        for line in groups.read_text().splitlines():
            if line.startswith("0::"):
                root = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                limit, current = root / "memory.max", root / "memory.current"
                if limit.is_file() and current.is_file():
                    maximum = limit.read_text().strip()
                    if maximum != "max":
                        headroom = int(maximum) - int(current.read_text())
                        if nbytes > headroom:
                            raise BufferError(
                                f"Pinning needs {nbytes} bytes; "
                                f"cgroup headroom is {headroom}."
                            )
                break


class _HostRegistration:
    """One full shared mapping registered on the consumer's selected device."""

    def __init__(self, view: "memoryview[bytes]", device: object) -> None:
        self._view: "memoryview[bytes] | None" = memoryview(view)
        self.address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
        self.nbytes = view.nbytes
        self._device = device
        self._pid = os.getpid()
        self._state = "unregistered"

    def register(self) -> None:
        if self._pid != os.getpid():
            raise RuntimeError(
                "CUDA registration was inherited across fork; use spawn."
            )
        if self._state == "registered":
            return
        if self._state == "uncertain":
            raise RuntimeError(
                "Previous registration completion is uncertain; mapping retained."
            )
        self._state = "uncertain"
        with torch.cuda.device(self._device):
            error = torch.cuda.cudart().cudaHostRegister(self.address, self.nbytes, 0)
        if int(error):
            self._state = "unregistered"
            raise RuntimeError(
                f"cudaHostRegister failed with error {int(error)} for {self.nbytes} bytes."
            )
        self._state = "registered"

    def close(self) -> None:
        view = self._view
        if view is None:
            return
        if self._pid != os.getpid():
            raise RuntimeError(
                "Cannot unregister a CUDA mapping inherited across fork."
            )
        if self._state != "unregistered":
            with torch.cuda.device(self._device):
                error = torch.cuda.cudart().cudaHostUnregister(self.address)
                if not int(error):
                    self._state = "unregistered"
            if int(error):
                raise RuntimeError(
                    f"cudaHostUnregister failed with error {int(error)}; mapping retained."
                )
        view.release()
        self._view = None


def _allocate_pinned(nbytes: int) -> "memoryview[bytes]":
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    import importlib

    try:
        buffer = importlib.import_module("spdl.io.lib._buffer")
    except ModuleNotFoundError as error:
        if error.name not in (
            "spdl.io",
            "spdl.io.lib",
            "spdl.io.lib._buffer",
        ):
            raise
        raise RuntimeError(
            "Pinned arena storage requires optional SPDL IO buffer support; "
            "install spdl.io with its native buffer helper."
        ) from error
    device = _local_device()
    with torch.cuda.device(device):
        storage = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    return cast(
        "memoryview[bytes]", buffer._export_owned(storage.data_ptr(), nbytes, storage)
    )
