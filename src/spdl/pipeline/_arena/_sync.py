# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conservative process liveness and non-acknowledging capacity wakes."""

from __future__ import annotations

import os
from multiprocessing.synchronize import Semaphore

__all__ = ["_process_exists", "_signal_space"]


def _process_exists(pid: int) -> bool:
    if not pid:
        return False
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _signal_space(signal: Semaphore) -> None:
    """Coalesce one pending wake while the caller owns the allocator lock.

    The semaphore starts empty. Borrowing and restoring an available token, or
    queuing one when empty, never waits for a receiver acknowledgement. Consumers
    take tokens outside the allocator lock and recheck the real capacity/state;
    successful admission or cancellation hands off another wake when required.
    No CPython Condition waiter counters are read, reset or repaired.
    """
    signal.acquire(False)
    signal.release()
