# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Rotating pool of shared-memory segments — one segment per unit, zero-copy.

An alternative arena backend to :py:class:`SharedMemoryRingBuffer`. Each unit
(one pipeline item) is written into a whole, isolated shared-memory segment, and
the pool rotates through a fixed number of them. Two segment counters live in a
small control segment: ``published`` (units published by the worker) and
``reclaimed`` (units whose segments have been returned to the pool). Unit ``i``
always uses segment ``i % count``, so the reader derives the segment from its own
cursor and markers only need an in-segment offset.

Because each unit owns a whole contiguous segment, the reader hands restored
NumPy/Torch payloads back as **zero-copy views** directly over shared memory
(``read_binary`` returns a live ``memoryview``). The segment is only returned to
the pool once every underlying storage export has been released. The writer is gated by
``reclaimed`` (``free = count - (published - reclaimed)``), so it never reuses a
segment that still has live views.

Backpressure is Mode B (blocking): ``begin_unit`` waits on a process-shared
:py:class:`multiprocessing.Condition` until the consumer reclaims a segment
(up to ``acquire_timeout``), throttling a fast producer to the consumer's reclaim
rate. Set ``acquire_timeout=0`` for non-blocking behavior (raise immediately when
full). Elastic growth remains future work.

Outstanding views remain valid across re-iterations of the same worker. Pool
accounting is monotonic for the lifetime of the worker, so an iteration boundary
does not make a segment reusable while an earlier view still aliases it.
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
from functools import partial
from multiprocessing import shared_memory
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Condition as _Condition, Semaphore as _Semaphore
from typing import cast, TypedDict

from ._ownership import _export_buffer, _ExportGuard
from ._pinning import _check_host_budget, _HostRegistration, _local_device
from ._shm import _attach
from ._sync import _process_exists, _signal_space

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = ["SharedMemorySegmentPool"]

# Default time ``begin_unit`` waits for the consumer to reclaim a segment before
# giving up and raising. Generous, so it only fires on a genuinely stalled or
# dead consumer rather than normal slow-consumer backpressure.
_DEFAULT_ACQUIRE_TIMEOUT: float = 60.0

# Control segment (<QQBBQQQ>, 42 bytes): published and reclaimed unit counters
# (uint64), shared shutdown and channel mode flags (uint8), producer and consumer
# process IDs (uint64), and the next unit to read (uint64). The read cursor tracks
# delivery/discard separately from the reclaim watermark: a delivered unit may
# still have native exports, while an unread unit must block a fresh writer.
# Mode 0 is unselected; _ORDINARY selects ordinary serialized transport.
_CTRL: struct.Struct = struct.Struct("<QQBBQQQ")
_ORDINARY = 1
_CONTROL_LOCK_TIMEOUT = 5.0

# Binaries are aligned to this many bytes within a segment so that zero-copy
# views into them (e.g. ``spdl.io`` Packets payloads) land on aligned addresses.
# Segment bases are page-aligned mmaps, so an aligned in-segment offset yields an
# aligned absolute address. Keep in sync with ``SERIALIZATION_ALIGNMENT`` in
# ``libspdl/core/packets.cpp``.
_ALIGNMENT: int = 64

_REGISTRATION_QUARANTINE: set[SharedMemorySegmentPool] = set()
_REGISTRATION_QUARANTINE_LOCK = threading.Lock()


def _create_segments(
    segment_size: int, count: int
) -> tuple[shared_memory.SharedMemory, list[shared_memory.SharedMemory]]:
    ctrl = shared_memory.SharedMemory(create=True, size=_CTRL.size)
    segments = []
    try:
        _CTRL.pack_into(cast(memoryview, ctrl.buf), 0, 0, 0, 0, 0, 0, 0, 0)
        for _ in range(count):
            segments.append(shared_memory.SharedMemory(create=True, size=segment_size))
    except BaseException as error:
        for segment in [ctrl, *segments]:
            try:
                segment.unlink()
            except Exception as cleanup_error:
                error.add_note(
                    f"Shared allocation unlink rollback failed: {cleanup_error!r}"
                )
            try:
                segment.close()
            except Exception as cleanup_error:
                error.add_note(
                    f"Shared allocation close rollback failed: {cleanup_error!r}"
                )
        raise
    return ctrl, segments


class _PoolState(TypedDict):
    """Picklable state of a :py:class:`SharedMemorySegmentPool`."""

    ctrl_name: str
    seg_names: list[str]
    segment_size: int
    count: int
    acquire_timeout: float
    space_cv: _Condition
    space_signal: _Semaphore
    pin_memory: bool
    creator_pid: int


class SharedMemorySegmentPool:
    """**[Experimental]** A rotating pool of shared-memory segments, one per unit.

    Construct it in the parent process and pass it to
    :py:func:`spdl.pipeline.iterate_in_subprocess` via the ``arena`` argument;
    ownership transfers to the returned iterable, which closes and unlinks it at
    teardown. Restored NumPy/Torch payloads are zero-copy views over shared
    memory; the pool keeps each segment until its views are released.

    .. versionadded:: 0.5.0

    .. versionchanged:: 0.6.0
       Backpressure now uses a process-shared
       :py:class:`multiprocessing.Condition` instead of busy-polling the shared
       reclaim counter; idle producers no longer consume CPU while waiting.

    .. versionchanged:: 0.7.0
       Restored zero-copy views remain valid across worker re-iterations.
       Making ``SharedMemorySegmentPool`` safe to use with continuous source.

    .. note::

       **Performance.** Because each unit is restored as a zero-copy view directly
       over shared memory, this backend avoids the per-item serialization and
       copy-out that a plain pickle/queue transfer pays on *both* sides of the
       process boundary. The consistent, measured benefit is **substantially lower
       host CPU usage** for moving large payloads across the boundary — the
       transfer stops being a serialize-then-copy operation.

       Whether that converts into higher end-to-end throughput depends on where
       the bottleneck is. It helps most when the pipeline is **CPU-bound or
       transfer/serialization-bound**. When the consumer is the bottleneck (for
       example a GPU-starved, decode-bound stage), throughput can be unchanged and
       the gain instead shows up as **freed CPU headroom** — which still matters
       when many workers contend for host cores. Treat this as an opt-in
       optimization and measure your own pipeline before relying on it.

    .. seealso::

       :py:mod:`benchmark_arena_transport`
          Benchmarks this pool against a plain pickle/queue transfer and the
          ring buffer, across payload types and sizes.

       :py:mod:`benchmark_pinned_arena`
          Matched public pool/ring ``pin_memory=False/True`` comparisons through
          subprocess iteration and synchronized H2D, with archival earlier results.

       :ref:`shared-memory-arena`
          Case study with the measured CPU, throughput, and memory numbers.

    Args:
        segment_size: Size of each segment in bytes. Must be at least as large as
            the biggest single pipeline unit (the sum of its offloaded binaries).
        count: Number of segments. With Mode B backpressure (the default) the
            producer waits for the consumer to reclaim a segment when the pool is
            full, so ``count`` bounds memory and in-flight units rather than
            being a hard cap that errors. Size it to the working set that keeps
            the consumer fed (``buffer_size`` plus concurrent consumer holds).
        acquire_timeout: Seconds ``begin_unit`` waits for a free segment before
            raising :py:exc:`BufferError`. Guards against a stalled/dead
            consumer; set to ``0`` for non-blocking (Mode A) behavior.
        pin_memory: Register the consumer's complete data mappings once on the
            CUDA device selected by ``LOCAL_RANK``. Supported restored CPU
            tensor/array/memoryview views then have page-locked backing. CUDA
            initializes lazily in the consumer, never the writer. Defaults to
            ``False``. Budget ``count * segment_size`` per pool and local rank;
            retained aliases and synchronized H2D sources prevent reuse/unmap.

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
        segment_size: int,
        count: int,
        *,
        acquire_timeout: float = _DEFAULT_ACQUIRE_TIMEOUT,
        pin_memory: bool = False,
        _mp_context: BaseContext | None = None,
    ) -> None:
        if segment_size <= 0:
            raise ValueError("segment_size must be positive")
        if count <= 0:
            raise ValueError("count must be positive")
        if acquire_timeout < 0:
            raise ValueError("acquire_timeout must be non-negative")
        self._segment_size: int = segment_size
        self._count: int = count
        self._acquire_timeout: float = acquire_timeout
        self._pin_memory = pin_memory
        context = _mp_context or multiprocessing.get_context("spawn")
        self._space_cv: _Condition = context.Condition(context.RLock())
        self._space_signal: _Semaphore = context.Semaphore(0)
        if pin_memory:
            _check_host_budget(segment_size * count, shared=True)
        ctrl, segs = _create_segments(segment_size, count)
        self._ctrl_name: str = ctrl.name
        self._seg_names: list[str] = [s.name for s in segs]
        self._ctrl: shared_memory.SharedMemory | None = ctrl
        self._segs: list[shared_memory.SharedMemory] | None = segs
        self._ctrl_buf: "memoryview[bytes] | None" = cast("memoryview[bytes]", ctrl.buf)
        self._seg_bufs: "list[memoryview[bytes]] | None" = [
            cast("memoryview[bytes]", segment.buf) for segment in segs
        ]
        self._closed = False
        self._closing = False
        self._reader: _PoolReader | None = None
        self._close_lock = threading.RLock()
        self._registrations: dict[str, _HostRegistration] = {}
        self._mapping_exports: dict[str, int] = {}
        self._pin_ready = False
        self._pin_error: BaseException | None = None
        self._pin_pid: int | None = None
        self._mode = 0
        self._pid = os.getpid()
        self._creator_pid = self._pid
        self._owns_names = True
        self._retirement_error: BaseException | None = None

    # -- pickling: carry only the spec; attach lazily, once, per process --

    def __getstate__(self) -> _PoolState:
        return {
            "ctrl_name": self._ctrl_name,
            "seg_names": self._seg_names,
            "segment_size": self._segment_size,
            "count": self._count,
            "acquire_timeout": self._acquire_timeout,
            "pin_memory": self._pin_memory,
            "space_cv": self._space_cv,
            "space_signal": self._space_signal,
            "creator_pid": self._creator_pid,
        }

    def __setstate__(self, state: _PoolState) -> None:
        self._ctrl_name = state["ctrl_name"]
        self._seg_names = state["seg_names"]
        self._segment_size = state["segment_size"]
        self._count = state["count"]
        self._acquire_timeout = state["acquire_timeout"]
        self._pin_memory = state.get("pin_memory", False)
        self._space_cv = state["space_cv"]
        self._space_signal = state["space_signal"]
        self._ctrl = None
        self._segs = None
        self._ctrl_buf = None
        self._seg_bufs = None
        self._closed = False
        self._closing = False
        self._reader: _PoolReader | None = None
        self._close_lock = threading.RLock()
        self._registrations = {}
        self._mapping_exports = {}
        self._pin_ready = False
        self._pin_error = None
        self._pin_pid = None
        self._mode = 0
        self._pid = os.getpid()
        self._creator_pid = state["creator_pid"]
        self._owns_names = False
        self._retirement_error = None

    def _ctrl_view(self) -> "memoryview[bytes]":
        if self._closed:
            raise RuntimeError("segment pool is closed")
        if self._ctrl_buf is None:
            self._ctrl = _attach(self._ctrl_name)
            # pyrefly: ignore [bad-assignment]
            self._ctrl_buf = self._ctrl.buf
        # pyrefly: ignore [bad-return]
        return self._ctrl_buf

    def _segment(self, index: int) -> "memoryview[bytes]":
        if self._closed:
            raise RuntimeError("segment pool is closed")
        if self._seg_bufs is None:
            self._segs = [_attach(n) for n in self._seg_names]
            # pyrefly: ignore [bad-assignment]
            self._seg_bufs = [s.buf for s in self._segs]
        # pyrefly: ignore [unsupported-operation]
        return self._seg_bufs[index % self._count]

    def _ensure_pinned(self) -> None:
        if not self._pin_memory:
            return
        if self._pin_pid is not None and self._pin_pid != os.getpid():
            raise RuntimeError(
                "Pinned consumer mappings were inherited across fork; use spawn."
            )
        if self._pin_error is not None:
            raise RuntimeError(
                "Arena pinning initialization previously failed"
            ) from self._pin_error
        if self._pin_ready:
            return
        device = _local_device()
        self._segment(0)
        _check_host_budget(self._segment_size * self._count, shared=False)
        self._pin_pid = os.getpid()
        with _REGISTRATION_QUARANTINE_LOCK:
            _REGISTRATION_QUARANTINE.add(self)
        try:
            for segment in self._segs or []:
                view = segment.buf
                if view is None:
                    raise RuntimeError("Shared segment mapping is closed")
                registration = _HostRegistration(
                    cast("memoryview[bytes]", view), device
                )
                self._registrations[segment.name] = registration
                registration.register()
            self._pin_ready = True
        except BaseException as error:
            self._pin_error = error
            for name, registration in list(self._registrations.items())[::-1]:
                try:
                    registration.close()
                except BaseException as rollback_error:  # noqa: B036 - Primary error wins.
                    with _REGISTRATION_QUARANTINE_LOCK:
                        _REGISTRATION_QUARANTINE.add(self)
                    error.add_note(
                        "CUDA registration rollback failed; "
                        f"mapping retained: {rollback_error!r}"
                    )
                else:
                    del self._registrations[name]
            if not self._registrations:
                with _REGISTRATION_QUARANTINE_LOCK:
                    _REGISTRATION_QUARANTINE.discard(self)
            raise

    def _export_released(self, name: str) -> None:
        with self._close_lock:
            remaining = self._mapping_exports.get(name, 0)
            if remaining < 1:
                raise RuntimeError("Arena mapping export released twice")
            self._mapping_exports[name] = remaining - 1
            if self._closed:
                self._close_segments()

    @property
    def _published(self) -> int:
        """Number of units published by the writer."""
        return int(_CTRL.unpack_from(self._ctrl_view(), 0)[0])

    @_published.setter
    def _published(self, value: int) -> None:
        struct.pack_into("<Q", self._ctrl_view(), 0, value)

    @property
    def _reclaimed(self) -> int:
        """Number of units whose segments have been returned to the pool."""
        return int(_CTRL.unpack_from(self._ctrl_view(), 0)[1])

    @_reclaimed.setter
    def _reclaimed(self, value: int) -> None:
        struct.pack_into("<Q", self._ctrl_view(), 8, value)
        # Wake any producer blocked in ``begin_unit`` waiting for a free segment.
        # Doing this in the setter (rather than at every call site) means manual
        # mutations of the counter — including the consumer's ``_unit_released``
        # path and tests that simulate a reclaim — also wake the producer.
        with self._condition():
            self._signal_space()

    @property
    def _shutdown(self) -> bool:
        return bool(_CTRL.unpack_from(self._ctrl_view(), 0)[2])

    @_shutdown.setter
    def _shutdown(self, value: bool) -> None:
        struct.pack_into("<B", self._ctrl_view(), 16, 1 if value else 0)

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
            if self._closed:
                return
        with self._condition(), self._close_lock:
            if not self._closed:
                self._shutdown = True
                self._signal_space()

    def _signal_space(self) -> None:
        """Queue a capacity hint while the caller holds the control lock."""
        _signal_space(self._space_signal)

    def _control_deadline(self, deadline: float) -> float:
        return deadline + _CONTROL_LOCK_TIMEOUT

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
                error.add_note(f"Capacity wake handoff failed: {cleanup!r}")
            raise

    @property
    def _read_cursor(self) -> int:
        return int(_CTRL.unpack_from(self._ctrl_view(), 0)[6])

    @_read_cursor.setter
    def _read_cursor(self, value: int) -> None:
        struct.pack_into("<Q", self._ctrl_view(), 34, value)

    def _retain_retirement_failure(self, error: BaseException) -> None:
        with self._close_lock:
            first = self._retirement_error is None
            if first:
                self._retirement_error = error
            with _REGISTRATION_QUARANTINE_LOCK:
                _REGISTRATION_QUARANTINE.add(self)
            if self._ctrl_buf is not None:
                try:
                    struct.pack_into("<B", self._ctrl_buf, 16, 1)
                except (BufferError, TypeError, ValueError) as cleanup:
                    error.add_note(f"Could not mark shared shutdown: {cleanup!r}")
        if first:
            _LG.error(
                "Arena retirement failed; shared mappings retained",
                exc_info=(type(error), error, error.__traceback__),
            )
        # A terminal wake cannot depend on the unavailable allocator lock.
        _signal_space(self._space_signal)

    @contextmanager
    def _condition(
        self, *, deadline: float | None = None, local_close: bool = False
    ) -> Iterator[None]:
        """Bound metadata access; retain storage on uncertain allocator state.

        Local-only retirement cannot cancel another handle's shared work.
        """
        if self._retirement_error is not None:
            if not local_close:
                _signal_space(self._space_signal)
            raise self._retirement_error
        timeout = _CONTROL_LOCK_TIMEOUT
        if deadline is not None:
            timeout = max(0.0, min(timeout, deadline - time.monotonic()))
        if not self._space_cv.acquire(timeout=timeout):
            error = RuntimeError(
                "Arena control lock unavailable; pool retired and mappings retained"
            )
            if local_close:
                _LG.error(
                    "Local pool close could not acquire control; mappings retained"
                )
            else:
                self._retain_retirement_failure(error)
            raise error
        try:
            yield
        finally:
            self._space_cv.release()

    def _select_mode(self, mode: int) -> None:
        if self._closed or self._retirement_error is not None:
            raise RuntimeError("segment pool is closed or retired")
        if self._mode == mode:
            return
        with self._condition(), self._close_lock:
            current = int(_CTRL.unpack_from(self._ctrl_view(), 0)[3])
            if current not in (0, mode):
                raise RuntimeError(
                    "Direct reservations cannot share ordinary pool endpoints"
                )
            struct.pack_into("<B", self._ctrl_view(), 17, mode)
            self._mode = mode

    def _register_endpoint(self, side: int) -> None:
        with self._condition(), self._close_lock:
            if self._closing or self._shutdown:
                raise BufferError("segment pool is closing")
            if (
                self._mode == _ORDINARY
                and side == 0
                and self._published > self._read_cursor
            ):
                raise RuntimeError(
                    "Previous arena channel has unread published units; "
                    "confirm delivery/discard before opening a new writer"
                )
            previous = int(_CTRL.unpack_from(self._ctrl_view(), 0)[4 + side])
            pid = os.getpid()
            if previous not in (0, pid) and (_process_exists(previous)):
                raise RuntimeError("Arena endpoint already belongs to another process")
            struct.pack_into("<Q", self._ctrl_view(), 18 + 8 * side, pid)

    def open_writer(self) -> _PoolWriter:
        """Return the writer endpoint (call in the worker process)."""
        self._select_mode(_ORDINARY)
        self._register_endpoint(0)
        return _PoolWriter(self)

    def open_reader(self) -> _PoolReader:
        """Return the reader endpoint (call in the parent process)."""
        self._select_mode(_ORDINARY)
        self._register_endpoint(1)
        with self._condition(), self._close_lock:
            if self._reader is None:
                self._reader = _PoolReader(self)
            return self._reader

    def close(self) -> None:
        """Close this handle, cancelling global work only for its original creator.

        The constructor-created handle owns names and terminal cancellation.
        Deserialized handles, including same-process copies, only retire their
        local mappings; closing a worker handle never consumes an in-transit
        allocation or cancels the creator. Only the creator may unlink names.

        Call the creator's ``close()`` in its ``finally`` block before stopping
        a pipeline whose producer may be blocked. The terminal signal wakes
        reservations but never invalidates existing storage aliases or H2D
        source owners. A known-live or uncertain remote endpoint, native export
        or active claim keeps control and data mappings durably retained.

        After workers are actually reaped and local aliases/claims are gone,
        call ``close()`` again to finish retirement, then ``unlink()``. Repeated
        calls do not reopen mappings, reset counters or permit future reuse.
        A normal return may mean deferred physical retirement; ``unlink()``
        reports ``BufferError`` until all retained mappings are closed.

        Raises:
            RuntimeError: Control acquisition is unavailable or the handle was
                inherited across fork. Uncertain storage remains retained.

        .. versionchanged:: 0.9.0
           Creator close is terminal cancellation with deferred retirement;
           deserialized holder close only retires its process-local mappings.
        """
        if os.getpid() != self._pid:
            raise RuntimeError("Inherited pool handle cannot close across fork")
        self._closing = True
        with _REGISTRATION_QUARANTINE_LOCK:
            _REGISTRATION_QUARANTINE.add(self)
        if self._closed:
            self._close_local()
            return
        if not self._owns_names and self._ctrl is None and self._segs is None:
            self._close_local()
            return
        with self._condition(local_close=not self._owns_names), self._close_lock:
            if self._closed:
                self._close_local()
                return
            if self._owns_names:
                self.shutdown_arena()
            producer, consumer = _CTRL.unpack_from(self._ctrl_view(), 0)[4:6]
            if self._owns_names and any(
                pid != os.getpid() and _process_exists(pid)
                for pid in (producer, consumer)
            ):
                return
            if any(self._mapping_exports.values()):
                return
            if consumer == os.getpid():
                reader = self._reader
                if reader is not None and reader.lease is not None:
                    return
                # Unused holders may close their independent local mapping;
                # actual readers and the creator retain outstanding unit state.
                if (self._owns_names or reader is not None) and (
                    self._reclaimed < self._read_cursor
                ):
                    return
            self._close_local()

    def _close_local(self) -> None:
        with self._close_lock:
            self._closed = True
            if self._ctrl_buf is not None:
                self._ctrl_buf.release()
                self._ctrl_buf = None
            if self._ctrl is not None:
                try:
                    self._ctrl.close()
                except BufferError:
                    pass  # Retain the control mapping for a later safe retry.
                else:
                    self._ctrl = None
            if self._seg_bufs is not None:
                for buf in self._seg_bufs:
                    buf.release()
                self._seg_bufs = None
            self._close_segments()

    def _close_segments(self) -> None:
        with self._close_lock:
            stuck = []
            for segment in self._segs or []:
                if self._mapping_exports.get(segment.name, 0):
                    stuck.append(segment)
                    continue
                if registration := self._registrations.get(segment.name):
                    try:
                        registration.close()
                    except Exception:
                        with _REGISTRATION_QUARANTINE_LOCK:
                            _REGISTRATION_QUARANTINE.add(self)
                        _LG.exception("CUDA unregister failed; shared mapping retained")
                        stuck.append(segment)
                        continue
                    del self._registrations[segment.name]
                try:
                    segment.close()
                except BufferError:
                    stuck.append(segment)
            self._segs = stuck or None
            if not self._registrations and self._segs is None and self._ctrl is None:
                with _REGISTRATION_QUARANTINE_LOCK:
                    _REGISTRATION_QUARANTINE.discard(self)

    def unlink(self) -> None:
        """Remove creator names only after physical retirement completes.

        Raises:
            BufferError: Mappings or registrations are retained. Release live
                owners, confirm worker reap and retry ``close()`` before unlink.
            RuntimeError: This is not the original creator handle.
        """
        if not self._owns_names or os.getpid() != self._creator_pid:
            raise RuntimeError("Only the creator handle may unlink arena names")
        with self._close_lock:
            if not self._closed or self._ctrl is not None or self._segs is not None:
                raise BufferError(
                    "Pool retirement is pending; retry close before unlink"
                )
            for name in [self._ctrl_name, *self._seg_names]:
                try:
                    shm = shared_memory.SharedMemory(name=name)
                except FileNotFoundError:
                    continue
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                finally:
                    shm.close()


class _PoolWriter:
    """Writer endpoint. Used as the last step in the worker process."""

    def __init__(self, pool: SharedMemorySegmentPool) -> None:
        self._p = pool
        self._seg_size: int = pool._segment_size
        self._acquire_timeout: float = pool._acquire_timeout
        self._cursor: int = 0

    def reset(self) -> None:
        """Clear in-progress state while preserving outstanding unit leases."""
        self._cursor = 0

    def begin_unit(self) -> None:
        """Acquire the next free segment for a new unit.

        Mode B backpressure: if the pool is full, wait on the arena's process-
        shared condition variable for the consumer to release a segment, up to
        ``acquire_timeout`` seconds, then raise. This throttles a fast producer
        to the consumer's reclaim rate instead of failing when a slow consumer
        legitimately holds many zero-copy views.

        Raises:
            BufferError: when the pool is full and the consumer does not reclaim
                a segment within ``acquire_timeout``, when the arena is shut
                down (e.g. the parent has begun teardown), or when
                ``acquire_timeout=0`` and no segment is currently free.
        """
        p = self._p
        deadline = time.monotonic() + self._acquire_timeout
        control_deadline = deadline + _CONTROL_LOCK_TIMEOUT
        expired = False
        while True:
            with p._condition(deadline=control_deadline):
                if p._closed or p._shutdown:
                    p._signal_space()
                    raise BufferError("segment pool is closing")
                if p._count - (p._published - p._reclaimed) >= 1:
                    self._cursor = 0
                    return
                if expired or time.monotonic() >= deadline:
                    raise BufferError(
                        "segment pool exhausted: no free segment within acquire_timeout"
                    )
            expired = not p._wait_space(deadline, control_deadline)

    def write_binary(
        self, data: "bytes | bytearray | memoryview[bytes]"
    ) -> tuple[int, int]:
        """Copy one binary into the current segment; return ``(offset, nbytes)``."""
        mv = memoryview(data).cast("B")
        n = mv.nbytes
        # Align the binary so zero-copy views into it land on aligned addresses.
        self._cursor = (self._cursor + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)
        if self._cursor + n > self._seg_size:
            raise BufferError(
                f"unit exceeds segment_size ({self._seg_size}); increase "
                "SharedMemorySegmentPool segment_size"
            )
        with self._p._condition(), self._p._close_lock:
            if self._p._closed or self._p._shutdown:
                raise BufferError("segment pool is closing")
            seg = self._p._segment(self._p._published)
            offset = self._cursor
            seg[offset : offset + n] = mv
            self._cursor += n
            return offset, n

    def commit_unit(self) -> int:
        """Publish the unit (advance the publish counter); return its byte span."""
        with self._p._condition(), self._p._close_lock:
            if self._p._closed or self._p._shutdown:
                raise BufferError("segment pool is closing")
            span = self._cursor
            self._p._published = self._p._published + 1
            return span

    def abort_unit(self) -> None:
        """Discard an in-progress unit without publishing it."""
        self._cursor = 0


class _SegmentLease:
    """A unit hold plus one hold for each underlying buffer exporter."""

    def __init__(self, reader: _PoolReader, unit: int) -> None:
        self._reader = reader
        self._unit = unit
        self.remaining = 1
        self.lock = threading.RLock()

    def acquire(self) -> None:
        with self.lock:
            self.remaining += 1

    def release_export(self, name: str) -> None:
        pool = self._reader.pool
        pool._export_released(name)
        self.release_one()

    def release_one(self) -> None:
        with self.lock:
            self.remaining -= 1
            done = self.remaining == 0
        if done:
            self._reader.unit_released(self._unit)


class _PoolReader:
    """Reader endpoint. Used as the first step in the parent process.

    Returns shared byte views and defers segment reuse until every actual
    buffer export has retired, independent of payload-wrapper finalization.
    """

    # Each returned buffer acquires a native export hold before reconstruction.
    zero_copy: bool = True

    def __init__(self, pool: SharedMemorySegmentPool) -> None:
        self.pool = pool
        self._next: int = pool._read_cursor
        self._reclaimed: int = pool._reclaimed
        self._released: set[int] = set()  # released units above the watermark
        self._lock = threading.RLock()
        self.lease: _SegmentLease | None = None

    def reset(self) -> None:
        """Preserve monotonic accounting and outstanding leases."""

    def _acquire_export(
        self, lease: _SegmentLease, guard: _ExportGuard, name: str
    ) -> None:
        """Acquire both setup holds, or restore them and cancel the native ticket."""
        try:
            guard.activate()
            with lease.lock:
                previous_exports = self.pool._mapping_exports.get(name, 0)
                previous_holds = lease.remaining
                try:
                    self.pool._mapping_exports[name] = previous_exports + 1
                    lease.remaining = previous_holds + 1
                    self.lease = lease
                except BaseException:
                    self.pool._mapping_exports[name] = previous_exports
                    lease.remaining = previous_holds
                    raise
        except BaseException as error:
            try:
                guard.cancel()
            except BaseException as cleanup_error:  # noqa: B036 - Primary error wins.
                error.add_note(f"Export acquisition rollback failed: {cleanup_error!r}")
            raise

    def read_binary(self, offset: int, nbytes: int) -> "memoryview[bytes]":
        """Return a storage-owned view into the current unit's segment."""
        if offset < 0 or nbytes < 0 or offset + nbytes > self.pool._segment_size:
            raise ValueError("binary is outside the segment")
        guard = None
        try:
            with self.pool._condition(), self.pool._close_lock:
                if self.pool._closed or self.pool._shutdown:
                    raise RuntimeError("segment pool is closed")
                self.pool._ensure_pinned()
                name = self.pool._seg_names[self._next % self.pool._count]
                view = self.pool._segment(self._next)[offset : offset + nbytes]
                lease = self.lease
                if lease is None:
                    lease = _SegmentLease(self, self._next)
                guard = _ExportGuard(partial(lease.release_export, name))
                self._acquire_export(lease, guard, name)
            return _export_buffer(view, guard)
        except BaseException as error:
            if guard is not None:
                try:
                    guard.rollback()
                except BaseException as cleanup_error:  # noqa: B036 - Primary error wins.
                    error.add_note(f"Export setup rollback failed: {cleanup_error!r}")
            raise

    def end_unit(self, span: int, anchors: list[object] | None = None) -> None:
        """Release the logical unit; actual buffer exports still gate FIFO reuse.

        Holds are acquired at ``read_binary``, including an export that escapes
        a failed payload reconstruction. ``anchors`` remains accepted for the
        extension contract; this builtin ignores the list.

        .. versionchanged:: 0.9.0
           Renamed ``pinned`` to optional ``anchors`` with a ``None`` default.
           Replace the old ``pinned=`` keyword with ``anchors=``; positional
           calls are unchanged.
        """
        with self.pool._condition(), self.pool._close_lock, self._lock:
            if self.pool._closed:
                return
            lease = self.lease
            unit = self._next
            self._next += 1
            self.pool._read_cursor = self._next
            self.lease = None
            if lease is None:
                self.unit_released(unit)
            else:
                lease.release_one()

    def unit_released(self, unit: int) -> None:
        with self.pool._close_lock, self._lock:
            self._released.add(unit)
        if self.pool._closed or self.pool._retirement_error is not None:
            return
        try:
            with self.pool._condition(), self.pool._close_lock, self._lock:
                watermark = self._reclaimed
                while watermark in self._released:
                    watermark += 1
                if watermark > self._reclaimed and not self.pool._closed:
                    # Publish before forgetting pending releases. A failure
                    # retains the pending units and forbids unsafe reuse.
                    self.pool._reclaimed = watermark
                    self._released.difference_update(range(self._reclaimed, watermark))
                    self._reclaimed = watermark
        except Exception as error:
            # Native-export finalizers cannot propagate reclaim errors. Make
            # the terminal failure observable and wake blocked producers.
            self.pool._retain_retirement_failure(error)
