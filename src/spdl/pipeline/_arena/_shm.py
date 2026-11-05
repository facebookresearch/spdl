# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Non-owning shared-memory attachments without resource-tracker mutation."""

from __future__ import annotations

import inspect
import mmap
import os
import sys
from collections.abc import Callable
from multiprocessing import shared_memory
from types import ModuleType
from typing import cast

__all__ = ["_attach"]


class _AttachedSharedMemory(shared_memory.SharedMemory):
    """Protect the stdlib's constructor error cleanup from unlinking an owner."""

    def __init__(self, name: str) -> None:
        self._initializing = True
        try:
            initialize = cast(Callable[..., None], super().__init__)
            initialize(name=name, track=False)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:  # noqa: B036 - Primary error wins.
                error.add_note(
                    f"Non-owning attachment cleanup failed: {cleanup_error!r}"
                )
            raise
        finally:
            self._initializing = False

    def unlink(self) -> None:
        if not self._initializing:
            raise RuntimeError(
                "Non-owning shared-memory attachments must not unlink names"
            )

    def __reduce__(self):
        return _attach, (self.name,)


class _UntrackedSharedMemory(shared_memory.SharedMemory):
    """CPython POSIX fallback for versions without the track keyword."""

    def __init__(self, name: str) -> None:
        self._name = "/" + name.lstrip("/")
        self._fd = -1
        self._mmap: mmap.mmap | None = None
        self._buf: memoryview | None = None
        self._track = False
        try:
            posix = cast(ModuleType, shared_memory)._posixshmem
            self._fd = posix.shm_open(self._name, os.O_RDWR, mode=0o600)
            self._size = os.fstat(self._fd).st_size
            self._mmap = mmap.mmap(self._fd, self._size)
            self._buf = memoryview(self._mmap)
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(
                    f"Non-owning attachment cleanup failed: {cleanup_error!r}"
                )
            raise

    def unlink(self) -> None:
        raise RuntimeError("Non-owning shared-memory attachments must not unlink names")


def _attach(name: str) -> shared_memory.SharedMemory:
    """Attach without registering or unregistering the owner's shared name.

    Related multiprocessing processes share a tracker *set*, not per-handle
    refcounts. Register-then-unregister in a consumer can remove the owner's
    entry or race a second unregister. Use the stdlib's track=False capability
    when present (including backports); older CPython POSIX runtimes attach
    directly with shm_open/mmap. No process-global tracker monkeypatch is used.

    Args:
        name: Existing shared-memory name.

    Returns:
        A non-owning mapping. Only the arena owner removes the shared name.
    """
    if sys.platform == "win32":
        return shared_memory.SharedMemory(name=name)
    if "track" in inspect.signature(shared_memory.SharedMemory).parameters:
        return _AttachedSharedMemory(name)
    return _UntrackedSharedMemory(name)
