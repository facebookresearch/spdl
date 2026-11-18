# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit pool reservations and storage-owned, two-sided unit lifetimes."""

from __future__ import annotations

import ctypes
import hashlib
import os
import pickle
import struct
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from multiprocessing import shared_memory
from types import TracebackType
from typing import Any, cast, NoReturn

from ._ownership import _export_buffer, _ExportGuard
from ._shm import _attach

__all__ = [
    "ArenaReservation",
    "_ArenaAllocationTicket",
    "_DirectPool",
    "_control_size",
    "_RecoveredAllocation",
]

_HEADER = struct.Struct("<QQQQ")
_OWNER = struct.Struct("<Q")
_SLOT = struct.Struct("<QQQB32s16s")
_RESERVED, _PUBLISHED, _CONSUMED, _ABORTED, _RESTORING = 1, 3, 4, 5, 6


@dataclass(frozen=True, slots=True)
class _ArenaAllocationTicket:
    """Opaque immutable byte descriptor; serialization never changes ownership."""

    _arena_id: str
    unit: int
    producer_pid: int
    extent: int

    @property
    def arena_id(self) -> str:
        """Stable routing identity of the pool that owns this opaque ticket."""
        return self._arena_id


def _control_size(count: int) -> int:
    return _HEADER.size + _OWNER.size + count * _SLOT.size


class _DirectPool:
    def __init__(self, pool: Any, offset: int) -> None:
        self.pool = pool
        self.pid = os.getpid()
        self._control_depth = 0
        self._draining_exports = False
        self._retirement_generation = 0
        self._deferred_exports: list[tuple[int, int, str]] = []
        self._deferred_aborts: list[int] = []
        self._shm = _attach(pool.arena_id)
        self.name = pool.arena_id
        try:
            buffer = self._shm.buf
            if buffer is None:
                raise RuntimeError("Direct pool control mapping is unavailable")
            self._buf = cast("memoryview[int]", buffer[offset:])
            self._closed = False
        except BaseException:
            self._shm.close()
            raise

    @contextmanager
    def _control(self) -> Iterator[None]:
        """Serialize control updates without reentrant GC changing their snapshots.

        Only nonblocking updates use this scope. Reservation waits hold neither
        the local close lock nor an active update, so retiring aliases can wake
        a waiting producer. Direct paths always acquire the condition first.
        """
        with self.pool._condition(), self.pool._close_lock:
            self._control_depth += 1
            primary: BaseException | None = None
            try:
                yield
            except BaseException as error:
                primary = error
                raise
            finally:
                self._control_depth -= 1
                if not self._control_depth and not self._draining_exports:
                    self._draining_exports = True
                    try:
                        self._drain_deferred()
                    except Exception as failure:
                        # Never replace the update's own failure.
                        if primary is None:
                            raise
                        primary.add_note(
                            f"Deferred arena retirement also failed: {failure!r}"
                        )
                    finally:
                        self._draining_exports = False

    def _drain_deferred(self) -> None:
        # One failed retirement must not strand the holds queued behind it.
        error: Exception | None = None
        while self._deferred_exports or self._deferred_aborts:
            try:
                if self._deferred_aborts:
                    self._abort_unit(self._deferred_aborts.pop())
                else:
                    unit, side, name = self._deferred_exports.pop()
                    self._retire_export(unit, side, name)
            except Exception as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(
                        f"Later deferred arena retirement also failed: {failure!r}"
                    )
        if error is not None:
            raise error

    def _slot_offset(self, unit: int) -> int:
        return _HEADER.size + _OWNER.size + (unit % self.pool._count) * _SLOT.size

    def _record(self, unit: int) -> tuple[int, int, int, int]:
        generation, producer, consumer, state, _, _ = _SLOT.unpack_from(
            self._buf, self._slot_offset(unit)
        )
        if generation != unit or state == 0:
            raise RuntimeError("Stale arena reservation generation")
        return generation, producer, consumer, state

    def _set(self, unit: int, producer: int, consumer: int, state: int) -> None:
        if producer < 0 or consumer < 0:
            raise RuntimeError("Arena reservation hold released twice")
        digest, claim = (
            (bytes(32), bytes(16))
            if state == _RESERVED
            else _SLOT.unpack_from(self._buf, self._slot_offset(unit))[4:]
        )
        _SLOT.pack_into(
            self._buf,
            self._slot_offset(unit),
            unit,
            producer,
            consumer,
            state,
            digest,
            claim,
        )

    def _reclaim(self) -> None:
        reserved, cursor, published, reclaimed = _HEADER.unpack_from(self._buf, 0)
        while cursor < reserved:
            _, producer, consumer, state = self._record(cursor)
            if producer or consumer or state not in (_CONSUMED, _ABORTED):
                break
            if state == _CONSUMED:
                reclaimed += 1
            cursor += 1
        _HEADER.pack_into(self._buf, 0, reserved, cursor, published, reclaimed)
        self.pool._signal_space()

    def _change_hold(self, unit: int, side: int, delta: int) -> None:
        with self._control():
            if self._closed:
                return
            _, producer, consumer, state = self._record(unit)
            if delta > 0 and state != (_RESERVED if side == 0 else _RESTORING):
                raise RuntimeError(
                    "Arena storage cannot be exported outside its owned transaction"
                )
            producer += delta if side == 0 else 0
            consumer += delta if side == 1 else 0
            if producer < 0 or consumer < 0:
                raise RuntimeError("Arena reservation hold released twice")
            self._set(unit, producer, consumer, state)
            self._reclaim()

    def reserve(self, max_bytes: int | None) -> ArenaReservation:
        limit = self.pool._segment_size if max_bytes is None else max_bytes
        if type(limit) is not int or not 0 < limit <= self.pool._segment_size:
            raise ValueError("max_bytes must fit one positive pool segment")
        p = self.pool
        deadline = time.monotonic() + p._acquire_timeout
        control_deadline = p._control_deadline(deadline)
        expired = False
        try:
            while True:
                with p._condition(deadline=control_deadline):
                    if self._closed or p._closed or p._shutdown:
                        p._signal_space()
                        raise BufferError("Direct output pool shut down")
                    owner = _OWNER.unpack_from(self._buf, _HEADER.size)[0]
                    if owner not in (0, os.getpid()):
                        raise RuntimeError(
                            "Direct output pool already belongs to another producer"
                        )
                    _OWNER.pack_into(self._buf, _HEADER.size, os.getpid())
                    with self._control():
                        reserved, cursor, published, consumed = _HEADER.unpack_from(
                            self._buf, 0
                        )
                        if reserved - cursor < p._count:
                            self._set(reserved, 1, 0, _RESERVED)
                            _HEADER.pack_into(
                                self._buf, 0, reserved + 1, cursor, published, consumed
                            )
                            if reserved + 1 - cursor < p._count:
                                p._signal_space()
                            break
                    if expired or time.monotonic() >= deadline:
                        raise BufferError(
                            "Direct output pool exhausted; release producer/consumer "
                            "aliases or increase count"
                        )
                expired = not p._wait_space(deadline, control_deadline)
        except BaseException as error:
            try:
                with p._condition(deadline=control_deadline):
                    p._signal_space()
            except BaseException as cleanup:  # noqa: B036 - Preserve the primary failure.
                error.add_note(f"Direct capacity handoff failed: {cleanup!r}")
            raise
        try:
            return ArenaReservation(self, reserved, limit)
        except BaseException:
            _abort_reservation(self, reserved)
            raise

    def _acquire_export(
        self, guard: _ExportGuard, unit: int, side: int, name: str
    ) -> None:
        """Acquire the setup holds, or cancel the ticket after restoring state."""
        p = self.pool
        try:
            guard.activate()
            with self._control():
                previous_exports = p._mapping_exports.get(name, 0)
                _, producer, consumer, state = self._record(unit)
                try:
                    p._mapping_exports[name] = previous_exports + 1
                    self._change_hold(unit, side, 1)
                except BaseException:
                    p._mapping_exports[name] = previous_exports
                    self._set(unit, producer, consumer, state)
                    raise
        except BaseException as error:
            try:
                guard.cancel()
            except BaseException as cleanup_error:  # noqa: B036 - Primary error wins.
                error.add_note(
                    f"Direct export acquisition rollback failed: {cleanup_error!r}"
                )
            raise

    def export(
        self, unit: int, offset: int, nbytes: int, side: int
    ) -> "memoryview[int]":
        p = self.pool
        guard = None
        try:
            with self._control():
                if p._closed:
                    raise RuntimeError("Direct output pool is closed")
                if side == 1:
                    p._ensure_pinned()
                name = p._seg_names[unit % p._count]
                view = p._segment(unit)[offset : offset + nbytes]
                guard = _ExportGuard(partial(self._release_export, unit, side, name))
                self._acquire_export(guard, unit, side, name)
            return cast("memoryview[int]", _export_buffer(view, guard))
        except BaseException as error:
            if guard is not None:
                try:
                    guard.rollback()
                except BaseException as cleanup_error:  # noqa: B036 - Primary error wins.
                    error.add_note(
                        f"Direct export setup rollback failed: {cleanup_error!r}"
                    )
            raise

    def _release_export(self, unit: int, side: int, name: str) -> None:
        p = self.pool
        if os.getpid() != self.pid:
            return
        if p._retirement_error is not None:
            p._export_released(name)
            return
        try:
            with p._condition(), p._close_lock:
                if self._control_depth or self._draining_exports:
                    # Native storage is gone; retire only after the active snapshot commits.
                    self._deferred_exports.append((unit, side, name))
                    return
                self._retire_export(unit, side, name)
        except RuntimeError as error:
            if p._retirement_error is not error:
                raise
            p._export_released(name)

    def _retire_export(self, unit: int, side: int, name: str) -> None:
        with self._control():
            self._change_hold(unit, side, -1)
            self.pool._export_released(name)
            self._retirement_generation += 1

    def seal(self, descriptor: _ArenaAllocationTicket) -> None:
        unit = descriptor.unit
        digest = self._validate(descriptor)
        with self._control():
            _, producer, consumer, state = self._record(unit)
            if state != _RESERVED or producer < 1:
                raise RuntimeError("Arena reservation is not open")
            if descriptor.producer_pid != self._producer_pid():
                raise RuntimeError("Arena reservation belongs to another producer PID")
            reserved, cursor, published, reclaimed = _HEADER.unpack_from(self._buf, 0)
            try:
                _HEADER.pack_into(
                    self._buf, 0, reserved, cursor, published + 1, reclaimed
                )
                _SLOT.pack_into(
                    self._buf,
                    self._slot_offset(unit),
                    unit,
                    producer - 1,
                    consumer + 1,
                    _PUBLISHED,
                    digest,
                    bytes(16),
                )
            except BaseException:
                if self._record(unit)[3] == _RESERVED:
                    _HEADER.pack_into(
                        self._buf, 0, reserved, cursor, published, reclaimed
                    )
                raise

    def abort(self, unit: int) -> None:
        with self.pool._condition(), self.pool._close_lock:
            if self._control_depth or self._draining_exports:
                self._deferred_aborts.append(unit)
                return
            self._abort_unit(unit)

    def _abort_unit(self, unit: int) -> None:
        with self._control():
            if self._closed:
                return
            generation, producer, consumer, state, _, _ = _SLOT.unpack_from(
                self._buf, self._slot_offset(unit)
            )
            if generation != unit or state != _RESERVED:
                return
            if producer < 1:
                raise RuntimeError("Arena reservation has no cleanup-owner hold")
            self._set(unit, producer - 1, consumer, _ABORTED)
            self._reclaim()
            self._retirement_generation += 1

    def finish(self, unit: int, claim: bytes) -> None:
        with self._control():
            if self._closed:
                return
            generation, producer, consumer, state, _, actual = _SLOT.unpack_from(
                self._buf, self._slot_offset(unit)
            )
            if generation != unit or state != _RESTORING or actual != claim:
                return
            if consumer < 1:
                raise RuntimeError("Arena allocation has no caller claim hold")
            self._set(unit, producer, consumer - 1, _CONSUMED)
            self._reclaim()

    def _validate(self, descriptor: _ArenaAllocationTicket) -> bytes:
        if self._closed or self.pool._closed:
            raise RuntimeError("Direct output pool is closed")
        if (
            type(descriptor) is not _ArenaAllocationTicket
            or descriptor.arena_id != self.name
        ):
            raise RuntimeError("Arena allocation ticket belongs to a different pool")
        if type(descriptor.unit) is not int or not 0 <= descriptor.unit < 2**64:
            raise RuntimeError("Invalid arena allocation generation")
        if type(descriptor.producer_pid) is not int or descriptor.producer_pid <= 0:
            raise RuntimeError("Invalid arena producer PID")
        if (
            type(descriptor.extent) is not int
            or not 0 <= descriptor.extent <= self.pool._segment_size
        ):
            raise RuntimeError("Invalid arena allocation extent")
        return _descriptor_digest(descriptor)

    def _claim(
        self, descriptor: _ArenaAllocationTicket, claim: bytes, *, discard: bool = False
    ) -> None:
        digest = self._validate(descriptor)
        with self._control():
            if not discard and (self.pool._closing or self.pool._shutdown):
                raise RuntimeError("Arena recovery is cancelled")
            _, producer, consumer, state = self._record(descriptor.unit)
            if state != _PUBLISHED:
                raise RuntimeError(
                    "Arena allocation ticket is stale or already claimed"
                )
            if descriptor.producer_pid != self._producer_pid():
                raise RuntimeError(
                    "Arena allocation ticket producer PID does not match"
                )
            expected = _SLOT.unpack_from(self._buf, self._slot_offset(descriptor.unit))[
                4
            ]
            if expected != digest:
                raise RuntimeError("Arena ticket does not match its sealed allocation")
            if consumer != 1:
                raise RuntimeError("Arena allocation transfer hold is invalid")
            _SLOT.pack_into(
                self._buf,
                self._slot_offset(descriptor.unit),
                descriptor.unit,
                producer,
                consumer,
                _RESTORING,
                expected,
                claim,
            )

    def recover(self, descriptor: _ArenaAllocationTicket) -> _RecoveredAllocation:
        self._validate(descriptor)
        result = _RecoveredAllocation(self, descriptor.unit)
        try:
            self._claim(descriptor, result.claim)
            result.value = self._restore(descriptor)
            return result
        except BaseException:
            self.finish(descriptor.unit, result.claim)
            raise

    def discard(self, descriptor: _ArenaAllocationTicket) -> None:
        self._validate(descriptor)
        claim = os.urandom(16)
        try:
            self._claim(descriptor, claim, discard=True)
        finally:
            self.finish(descriptor.unit, claim)

    def _restore(self, descriptor: _ArenaAllocationTicket) -> "memoryview[int]":
        if not descriptor.extent:
            return memoryview(b"")
        return self.export(descriptor.unit, 0, descriptor.extent, 1)

    def _producer_pid(self) -> int:
        with self._control():
            if self._closed:
                raise RuntimeError("Direct pool control mapping is closed")
            return int(_OWNER.unpack_from(self._buf, _HEADER.size)[0])

    def owned_by_process(self) -> bool:
        with self._control():
            return (
                not self._closed
                and _OWNER.unpack_from(self._buf, _HEADER.size)[0] == os.getpid()
            )

    def counts(self) -> tuple[int, int, int, int]:
        with self._control():
            return _HEADER.unpack_from(self._buf, 0)

    def pending_close(self, producer: int, consumer: int) -> bool:
        with self._control():
            reserved, cursor, _, _ = _HEADER.unpack_from(self._buf, 0)
            for unit in range(cursor, reserved):
                state = self._record(unit)[3]
                if state == _RESERVED and producer == os.getpid():
                    return True
                if state == _RESTORING and consumer in (0, os.getpid()):
                    return True
            return False

    def close(self) -> None:
        with self._control():
            if not self._closed:
                self._closed = True
                self._buf.release()
                self._shm.close()

    def unlink(self) -> None:
        try:
            shm = shared_memory.SharedMemory(name=self.name)
        except FileNotFoundError:
            return
        try:
            shm.unlink()
        finally:
            shm.close()


def _descriptor_digest(descriptor: _ArenaAllocationTicket) -> bytes:
    # @lint-ignore PYTHONPICKLEISBAD Caller-managed IPC between trusted processes.
    packet = pickle.dumps(descriptor, protocol=pickle.HIGHEST_PROTOCOL)
    if len(packet) > 65536:
        raise ValueError("Allocation ticket metadata exceeds 64 KiB")
    return hashlib.sha256(packet).digest()


class _RecoveredAllocation:
    def __init__(self, owner: _DirectPool, unit: int) -> None:
        self._owner, self._unit = owner, unit
        self.claim = os.urandom(16)
        self._pid = os.getpid()
        self.value: memoryview[int] | None = None
        self._entered = False
        self._finished = False
        self._lock = threading.Lock()

    def __enter__(self) -> "memoryview[int]":
        if os.getpid() != self._pid:
            raise RuntimeError("Recovered allocation context belongs to another PID")
        with self._lock:
            if self._entered or self._finished or self.value is None:
                raise RuntimeError("Recovered allocation context is not reusable")
            self._entered = True
            return self.value

    def __exit__(self, *exc: object) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("Recovered allocation context belongs to another PID")
        with self._lock:
            if self._finished or not self._entered:
                raise RuntimeError("Recovered allocation context is not owned here")
            self._owner.finish(self._unit, self.claim)
            self._finished = True
            self.value = None

    def __reduce__(self) -> NoReturn:
        raise TypeError("Recovered allocation contexts cannot be transferred")


class ArenaReservation:
    """**[Experimental]** A public writable byte reservation obtained from a pool.

    Users obtain this object from
    :py:meth:`spdl.pipeline.SharedMemorySegmentPool.reserve`; do not construct
    it with internal endpoint parameters. Its allocation and seal operations are
    the public caller interface, separate from transparent transport protocols::

        with pool.reserve() as unit:
            data = unit.allocate_bytes(128)
            data[:] = bytes(128)
            ticket = unit.seal()

    Prefer one allocation for a collated payload. The caller computes offsets,
    creates tensor/array views and sends dtype, shape and other metadata beside
    the ticket through ordinary IPC. ``seal()`` publishes only the byte extent
    from the segment start through the last allocation, including alignment
    padding. ``pool.recover(ticket)`` returns a context yielding that whole
    extent as one lifetime-owned memoryview; no payload decoding is implicit.

    Context exit aborts an unsealed reservation, but escaped producer storage
    continues to gate reuse. Seal creates an independent in-transit hold before
    returning the immutable ticket; losing it does not abort the allocation.
    After sealing, do not mutate producer aliases. Allocation sizes must be
    positive; sealing without allocations yields an empty recovered view and
    still requires exactly one recovery or discard.

    Args:
        owner: Internal control endpoint; users obtain reservations from a pool.
        unit: Allocation generation assigned by the pool.
        limit: Positive allocation byte limit within one segment.

    .. versionadded:: 0.9.0

    .. versionchanged:: 0.9.0
       Allocation and recovery are byte-only. Callers create typed views and
       carry metadata separately; ``seal()`` takes no payload argument.
    """

    def __init__(self, owner: _DirectPool, unit: int, limit: int) -> None:
        self._owner = owner
        self._unit = unit
        self._limit = limit
        self._cursor = 0
        self._sealed = False
        self._lock = threading.Lock()
        self._lease = _ReservationLease(owner, unit)
        self._finalizer = weakref.finalize(self, self._lease.retire, 0)

    def _check_owner(self) -> None:
        if os.getpid() != self._owner.pid:
            raise RuntimeError("Arena reservation belongs to another producer PID")

    def __enter__(self) -> ArenaReservation:
        self._check_owner()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._check_owner()
        if not self._sealed:
            self._finalizer()

    def _allocate(self, nbytes: int, alignment: int) -> "memoryview[int]":
        if self._sealed or not self._lease.is_open():
            raise RuntimeError("Arena unit is sealed or aborted")
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Allocation size must be positive")
        if type(alignment) is not int or alignment < 1 or alignment & (alignment - 1):
            raise ValueError("Alignment must be a positive power of two")
        offset = (self._cursor + alignment - 1) & -alignment
        if offset + nbytes > self._limit:
            raise BufferError("Allocation exceeds reserved max_bytes")
        view = self._owner.export(self._unit, offset, nbytes, 0)
        try:
            address = ctypes.addressof(ctypes.c_char.from_buffer(view))
            if address % alignment:
                raise ValueError(
                    "Requested alignment exceeds segment mapping alignment"
                )
        except BaseException as error:
            # A retained traceback must not keep the rejected export's hold.
            try:
                view.release()
            except BaseException as cleanup_error:  # noqa: B036 - Primary error wins.
                error.add_note(f"Rejected allocation release failed: {cleanup_error!r}")
            raise
        self._cursor = offset + nbytes
        return view

    def allocate_bytes(self, nbytes: int, *, alignment: int = 64) -> "memoryview[int]":
        """Allocate writable bytes within this unit; return a lifetime-owned view.

        Args:
            nbytes: Positive byte extent.
            alignment: Power-of-two address alignment.

        Returns:
            Writable memoryview, e.g. ``data = unit.allocate_bytes(128)``.
            Callers create typed views over these bytes and retain those views
            for as long as they are in use, including asynchronous transfers.

        .. versionadded:: 0.9.0
        """
        self._check_owner()
        with self._lock:
            return self._allocate(nbytes, alignment)

    def seal(self) -> _ArenaAllocationTicket:
        """Publish this unit's byte extent and return an opaque immutable ticket.

        Returns:
            Ticket for all allocated bytes and alignment padding, e.g.
            ``ticket = unit.seal()``. Send application metadata beside it, not
            inside it. Any number of serialized copies refer to the same single
            caller claim. Explicitly discard unread tickets; delivery failure
            and GC do not release their independent in-transit hold.

        Raises:
            RuntimeError: The reservation is already sealed or aborted.

        .. versionadded:: 0.9.0

        .. versionchanged:: 0.9.0
           Takes no payload argument; the ticket describes bytes only.
        """
        self._check_owner()
        with self._lock:
            if self._sealed or not self._finalizer.alive:
                raise RuntimeError("Arena unit is sealed or aborted")
            descriptor = _ArenaAllocationTicket(
                self._owner.name,
                self._unit,
                os.getpid(),
                self._cursor,
            )
            self._owner.seal(descriptor)
            self._lease.transfer()
            self._sealed = True
            return descriptor


def _abort_reservation(owner: _DirectPool, unit: int) -> None:
    owner.abort(unit)


class _ReservationLease:
    def __init__(self, owner: _DirectPool, unit: int) -> None:
        self._owner, self._unit = owner, unit
        self._phase = 0
        self._lock = threading.RLock()

    def is_open(self) -> bool:
        with self._lock:
            return self._phase == 0

    def retire(self, role: int) -> None:
        p = self._owner.pool
        if self._phase != role or os.getpid() != self._owner.pid:
            return
        if p._retirement_error is not None:
            return
        try:
            with p._condition(), p._close_lock, self._lock:
                if self._phase == role:
                    self._owner.abort(self._unit)
                    self._phase = 2
        except RuntimeError as error:
            if p._retirement_error is not error:
                raise

    def transfer(self) -> None:
        with self._lock:
            if self._phase != 0:
                raise RuntimeError(
                    "Reservation cleanup ownership was already transferred"
                )
            self._phase = 1
