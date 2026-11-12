# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Retain unfinished execution-region pool cleanup in its owning process."""

from __future__ import annotations

import os
import threading
from typing import Protocol

__all__ = [
    "_retain_pool_cleanup",
    "_release_pool_cleanup",
    "get_pending_pipeline_pool_cleanup_count",
    "retry_pending_pipeline_pool_cleanup",
]


class _PoolCleanup(Protocol):
    def shutdown(self) -> None: ...


_LOCK = threading.RLock()
_PENDING: dict[tuple[int, int], _PoolCleanup] = {}


def _reset_lock_after_fork() -> None:
    global _LOCK
    _LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)


def _retain_pool_cleanup(pool: _PoolCleanup) -> None:
    pid = os.getpid()
    owner = getattr(pool, "_owner_pid", pid)
    if isinstance(owner, int) and owner != pid:
        raise RuntimeError("Execution-region cleanup belongs to another process")
    with _LOCK:
        _PENDING[(pid, id(pool))] = pool


def _release_pool_cleanup(pool: _PoolCleanup) -> None:
    with _LOCK:
        removed = _PENDING.pop((os.getpid(), id(pool)), None)
    # Release the last root outside the registry lock: finalizers may re-enter.
    del removed


def _snapshot() -> list[_PoolCleanup]:
    pid = os.getpid()
    with _LOCK:
        snapshot = _PENDING.copy()
    return [pool for (owner, _), pool in snapshot.items() if owner == pid]


def get_pending_pipeline_pool_cleanup_count() -> int:
    """Count execution-region cleanup owners whose retirement is incomplete.

    These are SPDL-owned process/subinterpreter pools created by execution
    regions, including orphaned interpreters from failed worker startup. Failed
    shutdown or construction rollback retains their worker and queue owners,
    even if the original pipeline is dropped. This does not
    include caller-owned executors, standalone iterator workers or every kind
    of process pool.

    Returns:
        Number of retained cleanup owners in this process. An in-progress shutdown also
        counts as pending; a zero count is not proof that all application
        threads/processes have exited.

    .. versionadded:: 0.9.0
    """
    return len(_snapshot())


def retry_pending_pipeline_pool_cleanup() -> None:
    """Attempt each retained execution-region pool's cleanup once.

    Use this after a failed pipeline stop or construction rollback, once the
    cause of failure is resolved. Worker joins/escalation keep their usual
    per-worker bounds; this function is synchronous, not a background retry
    loop, and total time grows with the number of pending workers. Pools stay
    rooted until their workers retire and required queue closes succeed.
    Do not call from a signal handler or a pool's own shutdown callback.

    Newly registered pools are not added to this call's snapshot. Entries
    inherited by a forked child are never retried on behalf of their parent.
    This is in-process recovery, not recovery after the owning process exits.

    Raises:
        BaseExceptionGroup: Cleanup errors or interruptions from owners still
            pending after this attempt. All owners in the snapshot are attempted
            independently; unfinished resources remain retained for a later try.

    Example::

        from spdl.pipeline.config import (
            get_pending_pipeline_pool_cleanup_count,
            retry_pending_pipeline_pool_cleanup,
        )

        if get_pending_pipeline_pool_cleanup_count():
            retry_pending_pipeline_pool_cleanup()

    .. versionadded:: 0.9.0
    """
    errors: list[BaseException] = []
    for pool in _snapshot():
        try:
            pool.shutdown()
        except BaseException as error:  # noqa: B036 - attempt every retained owner
            errors.append(error)
    if errors:
        raise BaseExceptionGroup(
            "Execution-region pool cleanup is still pending", errors
        )
