# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Single shared-memory ring buffer — the first-cut arena backend.

A :py:class:`SharedMemoryRingBuffer` is a fixed-size contiguous shared-memory
block that carries large binary payloads from a worker process to a parent
process. It is the simplest object accepted by the ``arena`` argument of
:py:func:`spdl.pipeline.iterate_in_subprocess`.

The block starts with a small control header holding two absolute byte
cursors — ``head`` (advanced by the writer when it publishes a unit) and
``tail`` (advanced by the reader when it releases a unit) — followed by the
payload arena. Free space is ``capacity - (head - tail)``.

Writes that cross the physical end of the arena are split into two copies, so
the wrap-around is invisible to callers. A "unit" (one pipeline item, which may
contain several binaries) is written at successive offsets and published once,
and released once, so reservation/return happen in bulk per unit.

Backpressure: ``write_binary`` waits on a process-shared
:py:class:`multiprocessing.Condition` until the consumer releases enough space,
up to ``acquire_timeout`` seconds, then raises. Set ``acquire_timeout=0`` for
the legacy non-blocking behavior (raise immediately when full).
"""

from __future__ import annotations

import multiprocessing
import struct
from multiprocessing import shared_memory
from multiprocessing.synchronize import Condition as _Condition
from typing import TypedDict

from ._shm import _attach

__all__ = ["SharedMemoryRingBuffer"]


# Default time ``write_binary`` waits for the consumer to free enough space
# before giving up and raising. Generous, so it only fires on a genuinely
# stalled or dead consumer rather than normal slow-consumer backpressure.
_DEFAULT_ACQUIRE_TIMEOUT: float = 60.0


class _RingState(TypedDict):
    """Picklable state of a :py:class:`SharedMemoryRingBuffer`."""

    shm_name: str
    capacity: int
    acquire_timeout: float
    space_cv: _Condition


# Control header: head, tail (absolute, monotonic byte counters), shutdown
# (uint8 flag set on teardown to wake any blocked producer cleanly).
_HEADER: struct.Struct = struct.Struct("<QQB")
_HEADER_SIZE: int = _HEADER.size  # 17


class SharedMemoryRingBuffer:
    """**[Experimental]** A single shared-memory ring buffer for cross-process payloads.

    Construct it in the parent process and pass it to
    :py:func:`spdl.pipeline.iterate_in_subprocess` via the ``arena`` argument;
    ownership transfers to that call, which closes and unlinks it at teardown.

    .. versionadded:: 0.5.0

    .. versionchanged:: 0.6.0
       The producer now blocks on a process-shared
       :py:class:`multiprocessing.Condition` when the arena is full and resumes
       once the consumer releases enough space, throttling the producer instead
       of failing it. Pass ``acquire_timeout=0`` to restore the legacy
       raise-immediately behavior.

    .. versionadded:: 0.6.0
       The ``acquire_timeout`` argument.

    Args:
        capacity: Size of the payload arena in bytes. Must be at least as large
            as the biggest single pipeline unit (the sum of the binaries
            offloaded from one item). Size it to the in-flight high-water mark,
            roughly ``(buffer_size + 2) * max_unit_bytes``.
        acquire_timeout: Seconds ``write_binary`` waits for the consumer to free
            enough space before raising :py:exc:`BufferError`. Guards against a
            stalled/dead consumer; set to ``0`` for non-blocking behavior (raise
            immediately when the in-progress unit would not fit).
    """

    def __init__(
        self,
        capacity: int,
        *,
        acquire_timeout: float = _DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if acquire_timeout < 0:
            raise ValueError("acquire_timeout must be non-negative")
        self._capacity: int = capacity
        self._acquire_timeout: float = acquire_timeout
        shm = shared_memory.SharedMemory(create=True, size=_HEADER_SIZE + capacity)
        self._shm_name: str = shm.name
        self._shm: shared_memory.SharedMemory | None = shm
        # pyrefly: ignore [bad-assignment]
        self._buf: "memoryview[bytes] | None" = shm.buf
        # pyrefly: ignore [bad-argument-type]
        _HEADER.pack_into(shm.buf, 0, 0, 0, 0)
        # Process-shared condition variable used by the producer to block on
        # ``tail`` advances by the consumer (or on a shutdown wakeup).
        self._space_cv: _Condition = multiprocessing.Condition()

    # -- pickling: carry only the spec; attach lazily, once, per process --

    def __getstate__(self) -> _RingState:
        return {
            "shm_name": self._shm_name,
            "capacity": self._capacity,
            "acquire_timeout": self._acquire_timeout,
            "space_cv": self._space_cv,
        }

    def __setstate__(self, state: _RingState) -> None:
        self._shm_name = state["shm_name"]
        self._capacity = state["capacity"]
        self._acquire_timeout = state["acquire_timeout"]
        self._space_cv = state["space_cv"]
        self._shm = None
        self._buf = None

    @property
    def capacity(self) -> int:
        """Size of the payload arena in bytes."""
        return self._capacity

    @property
    def name(self) -> str:
        """Name of the underlying shared-memory segment."""
        return self._shm_name

    def _ensure(self) -> "memoryview[bytes]":
        if self._buf is None:
            self._shm = _attach(self._shm_name)
            # pyrefly: ignore [bad-assignment]
            self._buf = self._shm.buf
        # pyrefly: ignore [bad-return]
        return self._buf

    @property
    def head(self) -> int:
        """Absolute byte count published by the writer."""
        return int(_HEADER.unpack_from(self._ensure(), 0)[0])

    @head.setter
    def head(self, value: int) -> None:
        struct.pack_into("<Q", self._ensure(), 0, value)

    @property
    def tail(self) -> int:
        """Absolute byte count released by the reader."""
        return int(_HEADER.unpack_from(self._ensure(), 0)[1])

    @tail.setter
    def tail(self, value: int) -> None:
        struct.pack_into("<Q", self._ensure(), 8, value)
        # Wake any producer blocked in ``write_binary`` waiting for free space.
        # Doing this in the setter (rather than only at the end_unit call site)
        # also wakes the producer when tests or other consumers advance ``tail``
        # directly.
        with self._space_cv:
            self._space_cv.notify_all()

    @property
    def _shutdown(self) -> bool:
        return bool(_HEADER.unpack_from(self._ensure(), 0)[2])

    @_shutdown.setter
    def _shutdown(self, value: bool) -> None:
        struct.pack_into("<B", self._ensure(), 16, 1 if value else 0)

    def shutdown_arena(self) -> None:
        """Wake any producer blocked in :py:meth:`_RingWriter.write_binary`.

        Sets a sticky shutdown flag in the control header and broadcasts on the
        space condition variable so a producer that is currently waiting for
        free space exits the wait promptly with a :py:exc:`BufferError`. Safe
        to call multiple times and from either process.
        """
        try:
            self._shutdown = True
        except Exception:  # noqa: BLE001 — buf may already be released
            return
        with self._space_cv:
            self._space_cv.notify_all()

    def open_writer(self) -> _RingWriter:
        """Return the writer endpoint (call in the worker process)."""
        return _RingWriter(self, self._ensure())

    def open_reader(self) -> _RingReader:
        """Return the reader endpoint (call in the parent process)."""
        return _RingReader(self, self._ensure())

    def close(self) -> None:
        """Release this process's mapping of the segment."""
        # First wake any blocked producer so it exits its wait before we tear
        # down the buffer it observes.
        self.shutdown_arena()
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def unlink(self) -> None:
        """Remove the segment from the system (call once, by the owner)."""
        shm = self._shm
        temporary = shm is None
        if shm is None:
            # Already closed in this process: attach a temporary handle just to
            # unlink, using _attach so it is not (re)registered with the
            # resource_tracker, and close it afterwards so its mapping is freed.
            try:
                shm = _attach(self._shm_name)
            except FileNotFoundError:
                return
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        finally:
            if temporary:
                shm.close()


class _RingWriter:
    """Writer endpoint. Used as the last step in the worker process."""

    def __init__(self, buf: SharedMemoryRingBuffer, mv: "memoryview[bytes]") -> None:
        self._b = buf
        self._mv = mv
        self._cap: int = buf.capacity
        self._acquire_timeout: float = buf._acquire_timeout
        self._data_off: int = _HEADER_SIZE
        self._ustart: int = 0
        self._wpos: int = 0

    def reset(self) -> None:
        """Reclaim all space at an iteration boundary.

        Safe to set both cursors here because the parent is quiescent (waiting
        for ``ITERATION_STARTED``) and has not started reading the new
        iteration; any residual bytes from an abandoned iteration are discarded.
        """
        self._b.head = 0
        # ``tail`` setter notifies the space cv, which is the right behavior
        # here: a fresh iteration has the entire arena free.
        self._b.tail = 0
        self._ustart = self._wpos = 0

    def begin_unit(self) -> None:
        """Start a new unit at the current published head."""
        self._ustart = self._wpos = self._b.head

    def write_binary(
        self, data: "bytes | bytearray | memoryview[bytes]"
    ) -> tuple[int, int]:
        """Copy one binary into the arena; return its ``(offset, nbytes)``.

        Blocks (up to ``acquire_timeout``) on the arena's process-shared
        condition variable until the consumer has released enough space for the
        write, then proceeds. With ``acquire_timeout=0`` the call instead raises
        immediately when the in-progress unit would overflow the free space.

        Raises:
            BufferError: when ``n`` is larger than the entire ring (no amount of
                consumer reclaim can ever satisfy it), when the in-progress unit
                accumulates more bytes than the ring can ever hold (committing
                it would never reclaim those bytes), when the consumer does not
                free enough space within ``acquire_timeout``, when the arena is
                shut down, or when ``acquire_timeout=0`` and the write would not
                currently fit.
        """
        mv = memoryview(data).cast("B")
        n = mv.nbytes
        # Hard guards: no amount of consumer progress will satisfy these.
        if n > self._cap:
            raise BufferError(
                f"binary of {n} bytes exceeds ring capacity {self._cap}; "
                "increase SharedMemoryRingBuffer capacity"
            )
        in_unit = self._wpos - self._ustart
        if in_unit + n > self._cap:
            raise BufferError(
                f"unit would exceed ring capacity ({self._cap}): {in_unit} "
                f"already written + {n} new > capacity. Increase "
                "SharedMemoryRingBuffer capacity."
            )
        b = self._b
        free = self._cap - (self._wpos - b.tail)
        if n > free:
            if self._acquire_timeout == 0:
                raise BufferError(
                    f"shared-memory arena full: need {n} more bytes but only "
                    f"{free} free and acquire_timeout=0; increase "
                    "SharedMemoryRingBuffer capacity or acquire_timeout"
                )
            with b._space_cv:
                # Re-evaluate ``tail`` on each wakeup; the cv predicate also
                # exits if the arena is being shut down.
                def _ready() -> bool:
                    if b._shutdown:
                        return True
                    return self._cap - (self._wpos - b.tail) >= n

                if not b._space_cv.wait_for(_ready, timeout=self._acquire_timeout):
                    raise BufferError(
                        f"shared-memory arena full: need {n} more bytes after "
                        f"waiting {self._acquire_timeout:g}s — the consumer is "
                        "releasing units too slowly or has stalled. Increase "
                        "SharedMemoryRingBuffer capacity or acquire_timeout, "
                        "or check the consumer."
                    )
                if b._shutdown:
                    raise BufferError(
                        "shared-memory ring shut down while waiting for free space"
                    )
        buf = self._mv
        d = self._data_off
        idx = self._wpos % self._cap
        end_run = self._cap - idx
        if n <= end_run:
            buf[d + idx : d + idx + n] = mv
        else:
            buf[d + idx : d + self._cap] = mv[:end_run]
            buf[d : d + (n - end_run)] = mv[end_run:]
        self._wpos += n
        return idx, n

    def commit_unit(self) -> int:
        """Publish the unit (advance head); return the unit's byte span."""
        self._b.head = self._wpos
        return self._wpos - self._ustart

    def abort_unit(self) -> None:
        """Discard an in-progress unit without publishing it."""
        self._wpos = self._ustart


class _RingReader:
    """Reader endpoint. Used as the first step in the parent process."""

    # ``read_binary`` copies each payload out into a fresh private buffer (the ring
    # reuses its space immediately), so restored values never alias shared memory.
    zero_copy: bool = False

    def __init__(self, buf: SharedMemoryRingBuffer, mv: "memoryview[bytes]") -> None:
        self._b = buf
        self._mv = mv
        self._cap: int = buf.capacity
        self._data_off: int = _HEADER_SIZE

    def reset(self) -> None:
        """No-op: the writer resets both cursors at the iteration boundary."""

    def read_binary(self, offset: int, nbytes: int) -> "memoryview[bytes]":
        """Copy ``nbytes`` out of the arena starting at physical ``offset``.

        The ring reuses its space immediately, so it cannot hand out a live view;
        it always copies into a fresh writable buffer (concatenating across the
        physical seam when the payload wraps). Restored views therefore alias the
        copy, not shared memory.
        """
        buf = self._mv
        d = self._data_off
        end_run = self._cap - offset
        if nbytes <= end_run:
            data = bytes(buf[d + offset : d + offset + nbytes])
        else:
            data = bytes(buf[d + offset : d + self._cap]) + bytes(
                buf[d : d + (nbytes - end_run)]
            )
        # Wrap in a bytearray so the restored view is writable (and so Torch's
        # frombuffer does not warn about a read-only buffer).
        return memoryview(bytearray(data))

    def end_unit(self, span: int, pinned: list[object]) -> None:
        """Return the unit's region in bulk (advance tail).

        ``pinned`` is ignored: read_binary already copied the data out, so
        restored views alias the copy, not the ring, and the region is free to
        reuse immediately. Advancing ``tail`` via the property setter wakes any
        producer blocked in ``write_binary``.
        """
        self._b.tail = self._b.tail + span
