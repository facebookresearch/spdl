# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared-memory helpers used by the arena backends."""

from __future__ import annotations

import sys
from multiprocessing import resource_tracker, shared_memory

__all__ = ["_attach"]


def _attach(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing shared-memory segment by name, without owning it.

    A non-owning process must not let its ``resource_tracker`` unlink the segment
    when it exits (CPython bpo-39959); only the creating process unlinks. So after
    attaching, the segment is unregistered from this process's tracker.

    Args:
        name: The name of the shared-memory segment to attach to.

    Returns:
        A :py:class:`multiprocessing.shared_memory.SharedMemory` mapping the
        named segment, detached from this process's resource tracker.
    """
    shm = shared_memory.SharedMemory(name=name)
    # The ``resource_tracker`` only tracks shared memory on POSIX; Windows
    # refcounts the mapping handle in the OS and never registers the segment.
    # Worse, calling ``unregister`` on Windows when the tracker is not already
    # running forces it to launch, and that launch imports ``_posixsubprocess``
    # (a POSIX-only module), raising ``ModuleNotFoundError`` and killing the
    # process. So only unregister where the tracker actually owns the segment.
    if sys.platform != "win32":
        try:
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except (KeyError, AttributeError):
            pass
    return shm
