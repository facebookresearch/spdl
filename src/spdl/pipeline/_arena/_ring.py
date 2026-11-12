# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Single shared-memory ring buffer — the first-cut arena backend.

A :py:class:`SharedMemoryRingBuffer` is a fixed-size contiguous shared-memory
block that carries large binary payloads from a worker process to a parent
process. It is the simplest object accepted by the ``arena`` argument of
:py:func:`spdl.pipeline.iterate_in_subprocess`.

The block starts with a small control header holding two absolute byte
cursors — ``_head`` (advanced by the writer when it publishes a unit) and
``_tail`` (advanced by the reader when it releases a unit) — followed by the
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

import logging
import multiprocessing
import os
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import shared_memory
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Condition as _Condition, Semaphore as _Semaphore
from typing import cast, TypedDict

from ._pinning import _allocate_pinned, _check_host_budget
from ._shm import _attach
from ._sync import _process_exists, _signal_space

__all__ = ["SharedMemoryRingBuffer"]

_LG: logging.Logger = logging.getLogger(__name__)


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
    space_signal: _Semaphore
    pin_memory: bool
    creator_pid: int


# Control header: head, tail (absolute, monotonic byte counters), shutdown
# (uint8 flag set on teardown to wake any blocked producer cleanly).
_HEADER: struct.Struct = struct.Struct("<QQBQQ")
_HEADER_SIZE: int = _HEADER.size
_CONTROL_LOCK_TIMEOUT = 5.0
_RETIREMENT_ROOTS: set[SharedMemoryRingBuffer] = set()
_RETIREMENT_LOCK = threading.Lock()


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

    .. seealso::

       :py:mod:`benchmark_arena_transport`
          Benchmarks this ring buffer against a plain pickle/queue transfer and
          the segment pool, across payload types and sizes.

       :py:mod:`benchmark_pinned_arena`
          Matched public pool/ring ``pin_memory=False/True`` comparisons through
          subprocess iteration and synchronized H2D, with archival earlier results.

       :ref:`shared-memory-arena`
          Case study with the measured CPU, throughput, and memory numbers.

    Args:
        capacity: Size of the payload arena in bytes. Must be at least as large
            as the biggest single pipeline unit (the sum of the binaries
            offloaded from one item). Size it to the in-flight high-water mark,
            roughly ``(buffer_size + 2) * max_unit_bytes``.
        acquire_timeout: Seconds ``write_binary`` waits for the consumer to free
            enough space before raising :py:exc:`BufferError`. Guards against a
            stalled/dead consumer; set to ``0`` for non-blocking behavior (raise
            immediately when the in-progress unit would not fit).
        pin_memory: Copy supported restored tensor/array/memoryview payloads
            directly into independent Torch-allocator pinned destinations on
            the CUDA device selected by ``LOCAL_RANK``. The source ring is not
            CUDA-registered and its existing copy-out/reclaim contract is
            unchanged. CUDA initializes lazily only in the consumer. Retained
            outputs can accumulate pinned memory beyond ring capacity.
            Rebuilt owning bytes/bytearrays and Packets may be pageable.
            Defaults to ``False``.

    .. versionchanged:: 0.9.0
       Public endpoint methods conform to ``ArenaProtocol``. Storage sizing,
       counters and lifecycle helpers used only by SPDL are private; constructor
       configuration remains public. Native exports retain storage independently
       of the reader anchor list.

    .. versionadded:: 0.9.0
       The ``pin_memory`` argument.
    """

    def __init__(
        self,
        capacity: int,
        *,
        acquire_timeout: float = _DEFAULT_ACQUIRE_TIMEOUT,
        pin_memory: bool = False,
        _mp_context: BaseContext | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if acquire_timeout < 0:
            raise ValueError("acquire_timeout must be non-negative")
        self._capacity: int = capacity
        self._acquire_timeout: float = acquire_timeout
        self._pin_memory = pin_memory
        context = _mp_context or multiprocessing.get_context("spawn")
        self._space_cv: _Condition = context.Condition(context.RLock())
        self._space_signal: _Semaphore = context.Semaphore(0)
        shm = shared_memory.SharedMemory(create=True, size=_HEADER_SIZE + capacity)
        self._shm_name: str = shm.name
        self._shm: shared_memory.SharedMemory | None = shm
        # pyrefly: ignore [bad-assignment]
        self._buf: "memoryview[bytes] | None" = shm.buf
        # pyrefly: ignore [bad-argument-type]
        _HEADER.pack_into(shm.buf, 0, 0, 0, 0, 0, 0)
        self._closed = False
        self._closing = False
        self._shutdown_sent = False
        self._reader: _RingReader | None = None
        self._close_lock = threading.RLock()
        self._pid = os.getpid()
        self._creator_pid = self._pid
        self._owns_names = True
        self._retirement_error: BaseException | None = None

    # -- pickling: carry only the spec; attach lazily, once, per process --

    def __getstate__(self) -> _RingState:
        return {
            "shm_name": self._shm_name,
            "capacity": self._capacity,
            "acquire_timeout": self._acquire_timeout,
            "pin_memory": self._pin_memory,
            "space_cv": self._space_cv,
            "space_signal": self._space_signal,
            "creator_pid": self._creator_pid,
        }

    def __setstate__(self, state: _RingState) -> None:
        self._shm_name = state["shm_name"]
        self._capacity = state["capacity"]
        self._acquire_timeout = state["acquire_timeout"]
        self._pin_memory = state.get("pin_memory", False)
        self._space_cv = state["space_cv"]
        self._space_signal = state["space_signal"]
        self._shm = None
        self._buf = None
        self._closed = False
        self._closing = False
        self._shutdown_sent = False
        self._reader: _RingReader | None = None
        self._close_lock = threading.RLock()
        self._pid = os.getpid()
        self._creator_pid = state["creator_pid"]
        self._owns_names = False
        self._retirement_error = None

    @property
    def name(self) -> str:
        """Name of the underlying shared-memory segment."""
        return self._shm_name

    def _ensure(self) -> "memoryview[bytes]":
        if self._closed:
            raise RuntimeError("shared-memory ring is closed")
        if self._buf is None:
            if self._closing and self._shm is not None:
                raise RuntimeError(
                    "Ring retirement is pending; control is not reopened"
                )
            self._shm = _attach(self._shm_name)
            # pyrefly: ignore [bad-assignment]
            self._buf = self._shm.buf
        # pyrefly: ignore [bad-return]
        return self._buf

    @property
    def _head(self) -> int:
        """Absolute byte count published by the writer."""
        return int(_HEADER.unpack_from(self._ensure(), 0)[0])

    @_head.setter
    def _head(self, value: int) -> None:
        struct.pack_into("<Q", self._ensure(), 0, value)

    @property
    def _tail(self) -> int:
        """Absolute byte count released by the reader."""
        return int(_HEADER.unpack_from(self._ensure(), 0)[1])

    @_tail.setter
    def _tail(self, value: int) -> None:
        struct.pack_into("<Q", self._ensure(), 8, value)
        # Wake any producer blocked in ``write_binary`` waiting for free space.
        # Doing this in the setter (rather than only at the end_unit call site)
        # also wakes the producer when tests or other consumers advance ``_tail``
        # directly.
        with self._condition():
            self._signal_space()

    @property
    def _shutdown(self) -> bool:
        return bool(_HEADER.unpack_from(self._ensure(), 0)[2])

    @_shutdown.setter
    def _shutdown(self, value: bool) -> None:
        struct.pack_into("<B", self._ensure(), 16, 1 if value else 0)

    @contextmanager
    def _condition(
        self, *, deadline: float | None = None, local_close: bool = False
    ) -> Iterator[None]:
        if self._retirement_error is not None:
            if not local_close:
                _signal_space(self._space_signal)
            raise self._retirement_error
        timeout = _CONTROL_LOCK_TIMEOUT
        if deadline is not None:
            timeout = max(0.0, min(timeout, deadline - time.monotonic()))
        if not self._space_cv.acquire(timeout=timeout):
            error = RuntimeError("Ring control lock unavailable; mappings retained")
            with _RETIREMENT_LOCK:
                _RETIREMENT_ROOTS.add(self)
            if local_close:
                _LG.error(
                    "Local ring close could not acquire control; mapping retained"
                )
            else:
                with self._close_lock:
                    first = self._retirement_error is None
                    if first:
                        self._retirement_error = error
                    if self._buf is not None:
                        try:
                            struct.pack_into("<B", self._buf, 16, 1)
                        except (BufferError, TypeError, ValueError) as cleanup:
                            error.add_note(f"Could not mark ring shutdown: {cleanup!r}")
                if first:
                    _LG.error(
                        "Ring retirement failed; mapping retained",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                # Terminal wakes do not depend on the unavailable control lock.
                _signal_space(self._space_signal)
            raise error
        try:
            yield
        finally:
            self._space_cv.release()

    def shutdown_arena(self) -> None:
        """Request shutdown and wake a producer blocked on arena capacity.

        Call before worker join; this requests shutdown without closing mappings,
        unregistering live storage or unlinking names. Existing exports remain
        valid. Complete physical retirement separately through ``close()`` once
        actual worker and storage quiescence is established.

        Raises:
            RuntimeError: Control state is unavailable or retired. Owned backing
                remains retained; this failure is not proof of safe retirement.
        """
        with self._close_lock:
            if self._closed or (self._buf is None and self._shutdown_sent):
                return
        with self._condition(), self._close_lock:
            if self._closed or (self._buf is None and self._shutdown_sent):
                return
            self._shutdown = True
            self._signal_space()
            self._shutdown_sent = True

    def _signal_space(self) -> None:
        """Queue a capacity hint while the caller holds the control lock."""
        _signal_space(self._space_signal)

    def _wait_space(self, deadline: float, control_deadline: float) -> bool:
        try:
            return self._space_signal.acquire(
                timeout=max(0.0, deadline - time.monotonic())
            )
        except BaseException as error:
            try:
                with self._condition(deadline=control_deadline):
                    self._signal_space()
            except BaseException as cleanup:  # noqa: B036 - Preserve the primary failure.
                error.add_note(f"Ring capacity wake handoff failed: {cleanup!r}")
            raise

    def _register_endpoint(self, side: int) -> None:
        with self._condition(), self._close_lock:
            if self._closing or self._shutdown:
                raise BufferError("shared-memory ring is closing")
            if side == 0 and self._head != self._tail:
                raise RuntimeError(
                    "Previous ring channel has an unread span; "
                    "confirm delivery/discard before opening a new writer"
                )
            previous = int(_HEADER.unpack_from(self._ensure(), 0)[3 + side])
            pid = os.getpid()
            if previous not in (0, pid) and _process_exists(previous):
                raise RuntimeError("Ring endpoint belongs to another process")
            struct.pack_into("<Q", self._ensure(), 17 + side * 8, pid)

    def open_writer(self) -> _RingWriter:
        """Return the writer endpoint (call in the worker process)."""
        self._register_endpoint(0)
        return _RingWriter(self, self._ensure())

    def open_reader(self) -> _RingReader:
        """Return the reader endpoint (call in the parent process)."""
        self._register_endpoint(1)
        with self._condition(), self._close_lock:
            if self._reader is None:
                self._reader = _RingReader(self, self._ensure())
            return self._reader

    def close(self) -> None:
        """Close this handle; only the original creator cancels shared arena work.

        Creator close precedes pipeline stop/join to wake blocked work. After
        actual worker reap, retry retirement and then unlink. Deserialized
        handles only close their own mappings and cannot unlink creator names.
        Independent private copies and pinned H2D outputs remain valid.
        A normal return may mean deferred physical retirement; ``unlink()``
        reports ``BufferError`` until the retained mapping is closed.

        Raises:
            RuntimeError: Bounded control acquisition failed or the handle was
                inherited across fork. Owned mappings remain retained; do not
                interpret an unsuccessful close as proof of safe retirement.

        .. versionchanged:: 0.9.0
           Creator close cancels and defers unsafe retirement; deserialized
           holder close does not change the shared terminal state.
        """
        if os.getpid() != self._pid:
            raise RuntimeError("Inherited ring handle cannot close across fork")
        self._closing = True
        with _RETIREMENT_LOCK:
            _RETIREMENT_ROOTS.add(self)
        with self._close_lock:
            if self._closed or (
                not self._owns_names and self._buf is None and self._shm is None
            ):
                self._close_mapping()
                return
            if self._buf is None and self._shm is not None:
                self._close_mapping()
                return
        with self._condition(local_close=not self._owns_names), self._close_lock:
            if self._closed or self._buf is None:
                self._close_mapping()
                return
            if self._owns_names:
                self.shutdown_arena()
            pids = _HEADER.unpack_from(self._ensure(), 0)[3:]
            if self._owns_names and any(
                pid != os.getpid() and _process_exists(pid) for pid in pids
            ):
                return
            if self._reader is not None and self._reader.pending_unit:
                return
            self._close_mapping()

    def _close_mapping(self) -> None:
        """Retire only the local mapping, retaining exports for an explicit retry."""
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._shm is not None:
            try:
                self._shm.close()
            except BufferError:
                return
            self._shm = None
        self._closed = True
        with _RETIREMENT_LOCK:
            _RETIREMENT_ROOTS.discard(self)

    def unlink(self) -> None:
        """Remove creator names only after physical retirement completes.

        Raises:
            BufferError: The mapping is retained. Confirm worker/owner
                quiescence and retry ``close()`` before removing names.
            RuntimeError: This is not the original creator handle.
        """
        if not self._owns_names or os.getpid() != self._creator_pid:
            raise RuntimeError("Only the creator handle may unlink arena names")
        with self._close_lock:
            if not self._closed or self._shm is not None:
                raise BufferError(
                    "Ring retirement is pending; retry close before unlink"
                )
            try:
                shm = shared_memory.SharedMemory(name=self._shm_name)
            except FileNotFoundError:
                return
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            finally:
                shm.close()


class _RingWriter:
    """Writer endpoint. Used as the last step in the worker process."""

    def __init__(self, buf: SharedMemoryRingBuffer, mv: "memoryview[bytes]") -> None:
        self._b = buf
        self._mv = mv
        self._cap: int = buf._capacity
        self._acquire_timeout: float = buf._acquire_timeout
        self._data_off: int = _HEADER_SIZE
        self._ustart: int = 0
        self._wpos: int = 0

    def reset(self) -> None:
        """Reset a drained epoch without invalidating unread committed spans."""
        self._ustart = self._wpos = self._b._head

    def begin_unit(self) -> None:
        """Start a new unit at the current published head."""
        self._ustart = self._wpos = self._b._head

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
        deadline = time.monotonic() + self._acquire_timeout
        control_deadline = deadline + _CONTROL_LOCK_TIMEOUT
        expired = False
        while True:
            with b._condition(deadline=control_deadline), b._close_lock:
                if b._closing or b._shutdown:
                    b._signal_space()
                    raise BufferError("shared-memory ring is closing")
                free = self._cap - (self._wpos - b._tail)
                if n <= free:
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
                if expired or time.monotonic() >= deadline:
                    raise BufferError(
                        "shared-memory arena full within acquire_timeout; "
                        "increase capacity or release consumed units"
                    )
            expired = not b._wait_space(deadline, control_deadline)

    def commit_unit(self) -> int:
        """Publish the unit (advance head); return the unit's byte span."""
        with self._b._condition(), self._b._close_lock:
            if self._b._closing or self._b._shutdown:
                raise BufferError("shared-memory ring is closing")
            self._b._head = self._wpos
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
        self.pending_unit = False
        self._mv = mv
        self._cap: int = buf._capacity
        self._data_off: int = _HEADER_SIZE
        self._budget_checked = False

    def reset(self) -> None:
        """No-op: the writer resets both cursors at the iteration boundary."""

    def read_binary(self, offset: int, nbytes: int) -> "memoryview[bytes]":
        """Copy ``nbytes`` out of the arena starting at physical ``offset``.

        The ring reuses its space immediately, so it cannot hand out a live view;
        it always copies into a fresh writable buffer (concatenating across the
        physical seam when the payload wraps). Restored views therefore alias the
        copy, not shared memory.
        """
        with self._b._condition(), self._b._close_lock:
            if self._b._closing or self._b._shutdown:
                raise BufferError("shared-memory ring is closing")
            buf = self._mv
            d = self._data_off
            end_run = self._cap - offset
            if not 0 <= offset < self._cap or not 0 <= nbytes <= self._cap:
                raise ValueError("binary is outside the ring")
            if self._b._pin_memory and not self._budget_checked:
                _check_host_budget(self._cap, shared=False)
                self._budget_checked = True
            result = (
                _allocate_pinned(nbytes)
                if self._b._pin_memory
                else memoryview(bytearray(nbytes))
            )
            first = min(nbytes, end_run)
            result[:first] = buf[d + offset : d + offset + first]
            if first < nbytes:
                result[first:] = buf[d : d + nbytes - first]
            self.pending_unit = True
            return cast("memoryview[bytes]", result)

    def end_unit(self, span: int, anchors: list[object] | None = None) -> None:
        """Return the unit's region in bulk after independent copy-out.

        Returned views own their copied storage, not the ring. Advancing the
        private tail wakes any producer blocked on byte capacity.

        .. versionchanged:: 0.9.0
           Renamed ``pinned`` to optional ``anchors`` with a ``None`` default.
           Replace the old ``pinned=`` keyword with ``anchors=``; positional
           calls are unchanged.
        """
        if self._b._closed or self._b._buf is None:
            return
        with self._b._condition(), self._b._close_lock:
            if not self._b._closed and self._b._buf is not None:
                self._b._tail = self._b._tail + span
                self.pending_unit = False
