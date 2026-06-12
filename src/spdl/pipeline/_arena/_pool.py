# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
the pool once every view into it has been released — tracked with
``weakref.finalize`` on the restored objects. The writer is gated by
``reclaimed`` (``free = count - (published - reclaimed)``), so it never reuses a
segment that still has live views.

Backpressure is Mode B (blocking): ``begin_unit`` waits for the consumer to
reclaim a segment when the pool is full (up to ``acquire_timeout``), throttling a
fast producer to the consumer's reclaim rate. Set ``acquire_timeout=0`` for
non-blocking behavior (raise immediately when full). Elastic growth remains
future work.

.. note::

   Do not retain zero-copy views across a re-iteration of the same worker: the
   iteration boundary resets the cursors and the writer may overwrite a segment
   a stale view still points at.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
import weakref
from multiprocessing import shared_memory
from typing import TypedDict

from ._shm import _attach

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = ["SharedMemorySegmentPool"]

# How often ``begin_unit`` polls the (shared-memory) reclaim counter while
# waiting for a free segment. Small enough to be responsive, large enough to
# keep the busy-wait cheap.
_POLL_INTERVAL: float = 0.0005

# Default time ``begin_unit`` waits for the consumer to reclaim a segment before
# giving up and raising. Generous, so it only fires on a genuinely stalled or
# dead consumer rather than normal slow-consumer backpressure.
_DEFAULT_ACQUIRE_TIMEOUT: float = 60.0

# Control segment: published (units published), reclaimed (units returned).
_CTRL: struct.Struct = struct.Struct("<QQ")

# Binaries are aligned to this many bytes within a segment so that zero-copy
# views into them (e.g. ``spdl.io`` Packets payloads) land on aligned addresses.
# Segment bases are page-aligned mmaps, so an aligned in-segment offset yields an
# aligned absolute address. Keep in sync with ``SERIALIZATION_ALIGNMENT`` in
# ``libspdl/core/packets.cpp``.
_ALIGNMENT: int = 64


class _PoolState(TypedDict):
    """Picklable state of a :py:class:`SharedMemorySegmentPool`."""

    ctrl_name: str
    seg_names: list[str]
    segment_size: int
    count: int
    acquire_timeout: float


class SharedMemorySegmentPool:
    """**[Experimental]** A rotating pool of shared-memory segments, one per unit.

    Construct it in the parent process and pass it to
    :py:func:`spdl.pipeline.iterate_in_subprocess` via the ``arena`` argument;
    ownership transfers to the returned iterable, which closes and unlinks it at
    teardown. Restored NumPy/Torch payloads are zero-copy views over shared
    memory; the pool keeps each segment until its views are released.

    .. versionadded:: 0.5.0

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
    """

    def __init__(
        self,
        segment_size: int,
        count: int,
        *,
        acquire_timeout: float = _DEFAULT_ACQUIRE_TIMEOUT,
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
        ctrl = shared_memory.SharedMemory(create=True, size=_CTRL.size)
        # pyrefly: ignore [bad-argument-type]
        _CTRL.pack_into(ctrl.buf, 0, 0, 0)
        segs = [
            shared_memory.SharedMemory(create=True, size=segment_size)
            for _ in range(count)
        ]
        self._ctrl_name: str = ctrl.name
        self._seg_names: list[str] = [s.name for s in segs]
        self._ctrl: shared_memory.SharedMemory | None = ctrl
        self._segs: list[shared_memory.SharedMemory] | None = segs
        # pyrefly: ignore [bad-assignment]
        self._ctrl_buf: "memoryview[bytes] | None" = ctrl.buf
        # pyrefly: ignore [bad-assignment]
        self._seg_bufs: "list[memoryview[bytes]] | None" = [s.buf for s in segs]

    # -- pickling: carry only the spec; attach lazily, once, per process --

    def __getstate__(self) -> _PoolState:
        return {
            "ctrl_name": self._ctrl_name,
            "seg_names": self._seg_names,
            "segment_size": self._segment_size,
            "count": self._count,
            "acquire_timeout": self._acquire_timeout,
        }

    def __setstate__(self, state: _PoolState) -> None:
        self._ctrl_name = state["ctrl_name"]
        self._seg_names = state["seg_names"]
        self._segment_size = state["segment_size"]
        self._count = state["count"]
        self._acquire_timeout = state["acquire_timeout"]
        self._ctrl = None
        self._segs = None
        self._ctrl_buf = None
        self._seg_bufs = None

    @property
    def segment_size(self) -> int:
        """Size of each segment in bytes."""
        return self._segment_size

    @property
    def count(self) -> int:
        """Number of segments in the pool."""
        return self._count

    def _ctrl_view(self) -> "memoryview[bytes]":
        if self._ctrl_buf is None:
            self._ctrl = _attach(self._ctrl_name)
            # pyrefly: ignore [bad-assignment]
            self._ctrl_buf = self._ctrl.buf
        # pyrefly: ignore [bad-return]
        return self._ctrl_buf

    def _segment(self, index: int) -> "memoryview[bytes]":
        if self._seg_bufs is None:
            self._segs = [_attach(n) for n in self._seg_names]
            # pyrefly: ignore [bad-assignment]
            self._seg_bufs = [s.buf for s in self._segs]
        # pyrefly: ignore [unsupported-operation]
        return self._seg_bufs[index % self._count]

    @property
    def published(self) -> int:
        """Number of units published by the writer."""
        return int(_CTRL.unpack_from(self._ctrl_view(), 0)[0])

    @published.setter
    def published(self, value: int) -> None:
        struct.pack_into("<Q", self._ctrl_view(), 0, value)

    @property
    def reclaimed(self) -> int:
        """Number of units whose segments have been returned to the pool."""
        return int(_CTRL.unpack_from(self._ctrl_view(), 0)[1])

    @reclaimed.setter
    def reclaimed(self, value: int) -> None:
        struct.pack_into("<Q", self._ctrl_view(), 8, value)

    def open_writer(self) -> _PoolWriter:
        """Return the writer endpoint (call in the worker process)."""
        return _PoolWriter(self)

    def open_reader(self) -> _PoolReader:
        """Return the reader endpoint (call in the parent process)."""
        return _PoolReader(self)

    def close(self) -> None:
        """Release this process's mappings of the segments.

        Restored zero-copy views (Packets / NumPy / Torch) may still alias a
        segment at teardown — their slices keep the underlying mmap exported, so
        unmapping it raises ``BufferError``. We release what we can and *retain*
        the segments that are still mapped; their shared memory is freed once the
        views are garbage-collected, and :py:meth:`unlink` removes the names
        regardless. Raising here would crash an otherwise-successful run.
        """
        if (ctrl_buf := self._ctrl_buf) is not None:
            ctrl_buf.release()
            self._ctrl_buf = None
        if (seg_bufs := self._seg_bufs) is not None:
            for b in seg_bufs:
                try:
                    b.release()
                except BufferError:
                    pass
            self._seg_bufs = None
        if (ctrl := self._ctrl) is not None:
            ctrl.close()
            self._ctrl = None
        if (segs := self._segs) is not None:
            stuck = []
            for s in segs:
                try:
                    s.close()
                except BufferError:
                    # A live zero-copy view still maps this segment. Retain the
                    # SharedMemory object so its destructor doesn't re-raise now;
                    # it unmaps once the view is released.
                    stuck.append(s)
            self._segs = stuck or None
            if stuck:
                _LG.warning(
                    "SharedMemorySegmentPool: %d of %d segments still had live "
                    "zero-copy views at close(); their shared memory will be "
                    "released once those objects are garbage-collected. Release "
                    "restored Packets/arrays before tearing down the pipeline "
                    "to avoid this.",
                    len(stuck),
                    self._count,
                )

    def unlink(self) -> None:
        """Remove all segments from the system (call once, by the owner)."""
        for name in [self._ctrl_name, *self._seg_names]:
            # Attach a temporary handle via _attach (so it is not registered with
            # the resource_tracker) and close it after unlinking, to avoid
            # leaking the mapping or emitting already-unlinked warnings.
            try:
                shm = _attach(name)
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
        self._seg_size: int = pool.segment_size
        self._acquire_timeout: float = pool._acquire_timeout
        self._cursor: int = 0

    def reset(self) -> None:
        """Reset both cursors at an iteration boundary (the pool is quiescent)."""
        self._p.published = 0
        self._p.reclaimed = 0
        self._cursor = 0

    def begin_unit(self) -> None:
        """Acquire the next free segment for a new unit.

        Mode B backpressure: if the pool is full, wait (polling the shared
        reclaim counter) for the consumer to release a segment, up to
        ``acquire_timeout`` seconds, then raise. This throttles a fast producer
        to the consumer's reclaim rate instead of failing when a slow consumer
        legitimately holds many zero-copy views.
        """
        p = self._p
        if p.count - (p.published - p.reclaimed) < 1:
            deadline = time.monotonic() + self._acquire_timeout
            while p.count - (p.published - p.reclaimed) < 1:
                if time.monotonic() >= deadline:
                    raise BufferError(
                        "segment pool exhausted: no free segment after waiting "
                        f"{self._acquire_timeout:g}s — the consumer is releasing "
                        "views too slowly or has stalled. Increase "
                        "SharedMemorySegmentPool count/segment_size or "
                        "acquire_timeout, or check the consumer."
                    )
                time.sleep(_POLL_INTERVAL)
        self._cursor = 0

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
        seg = self._p._segment(self._p.published)
        offset = self._cursor
        seg[offset : offset + n] = mv
        self._cursor += n
        return offset, n

    def commit_unit(self) -> int:
        """Publish the unit (advance the publish counter); return its byte span."""
        span = self._cursor
        self._p.published = self._p.published + 1
        return span

    def abort_unit(self) -> None:
        """Discard an in-progress unit without publishing it."""
        self._cursor = 0


class _SegmentLease:
    """Reclaims a unit's segment once every anchor handed out for it is released.

    ``end_unit`` registers a :py:func:`weakref.finalize` per anchor (the
    ``np.frombuffer`` array / Torch tensor a restored view chains to); each
    firing decrements the count, and the unit is reclaimed when it reaches zero.
    """

    def __init__(
        self, reader: _PoolReader, unit: int, count: int, generation: int
    ) -> None:
        self._reader = reader
        self._unit = unit
        self._remaining = count
        self._generation = generation
        self._lock = threading.Lock()

    def release_one(self) -> None:
        with self._lock:
            self._remaining -= 1
            done = self._remaining == 0
        if done:
            self._reader._unit_released(self._unit, self._generation)


class _PoolReader:
    """Reader endpoint. Used as the first step in the parent process.

    Returns zero-copy views into the segments and defers reclaiming a segment
    until every anchor of its unit has been released.
    """

    # ``read_binary`` hands out live, reusable views into shared memory, so
    # handlers restore zero-copy views (anchored to defer reclamation).
    zero_copy: bool = True

    def __init__(self, pool: SharedMemorySegmentPool) -> None:
        self._p = pool
        self._next: int = 0  # next unit index to restore
        self._reclaimed: int = 0  # contiguous reclaim watermark (mirrors shm)
        self._released: set[int] = set()  # released units above the watermark
        self._generation: int = 0  # invalidates leases from prior iterations
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Reset cursors at an iteration boundary and orphan stale leases."""
        with self._lock:
            self._next = 0
            self._reclaimed = 0
            self._released.clear()
            self._generation += 1

    def read_binary(self, offset: int, nbytes: int) -> "memoryview[bytes]":
        """Return a live view into the current unit's segment (zero copy)."""
        return self._p._segment(self._next)[offset : offset + nbytes]

    def end_unit(self, span: int, pinned: list[object]) -> None:
        """Finish the current unit; reclaim now or once its anchors are released.

        ``pinned`` holds the lifetime anchors of the unit's zero-copy views. The
        segment is reclaimed once all of them are garbage-collected; finalizing
        the anchor (not the restored object) is what makes a view or shallow copy
        that outlives the original object keep the segment alive.
        """
        unit = self._next
        self._next += 1
        if not pinned:
            # Nothing aliases the segment (e.g. all fields copied out) — reclaim
            # immediately.
            self._unit_released(unit, self._generation)
            return
        lease = _SegmentLease(self, unit, len(pinned), self._generation)
        for anchor in pinned:
            weakref.finalize(anchor, lease.release_one)

    def _unit_released(self, unit: int, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return  # stale lease from a previous iteration; ignore
            self._released.add(unit)
            advanced = False
            while self._reclaimed in self._released:
                self._released.discard(self._reclaimed)
                self._reclaimed += 1
                advanced = True
            if advanced:
                self._p.reclaimed = self._reclaimed
