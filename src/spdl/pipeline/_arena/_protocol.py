# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Structural interfaces shared by arena backends.

These let ``iterate_in_subprocess`` accept any arena object through one
``arena`` argument and dispatch on its type. The single ring
(:py:class:`~spdl.pipeline._arena._ring.SharedMemoryRingBuffer`) and the rotating
segment pool (:py:class:`~spdl.pipeline._arena._pool.SharedMemorySegmentPool`)
both satisfy these protocols structurally.

A unit is one pipeline item (possibly several binaries). The writer brackets a
unit with ``begin_unit`` / ``commit_unit`` and the reader returns it with
``end_unit``, so reservation and return happen in bulk per unit regardless of
the backend.
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["ArenaProtocol", "ArenaReaderProtocol", "ArenaWriterProtocol"]


class ArenaWriterProtocol(Protocol):
    """Writer endpoint: the last step in the worker process.

    Backends implement this protocol to receive offloaded binaries. A *unit*
    is one pipeline item: it groups all binaries produced by a single
    :py:func:`_offload` call so they can be reserved
    and returned to the arena in bulk. Methods are called from a single
    thread in the worker; they need not be thread-safe.
    """

    def reset(self) -> None:
        """Reset per-iteration state at an iteration boundary.

        Called by the worker once per ``START_ITERATION``, while the parent
        is quiescent. Backends should reclaim any space that leaked across
        the boundary (e.g. units the parent never read) and return the
        writer to a clean starting state.

        Returns:
            None.
        """
        ...

    def begin_unit(self) -> None:
        """Open a new unit before any :py:meth:`write_binary` calls.

        Must be paired with exactly one terminal call —
        :py:meth:`commit_unit` on success or :py:meth:`abort_unit` on
        failure. Calling :py:meth:`write_binary` outside an open unit is
        an error.

        Returns:
            None.
        """
        ...

    def write_binary(
        self, data: "bytes | bytearray | memoryview[bytes]"
    ) -> tuple[int, int]:
        """Copy one binary into the arena and return its placement.

        Performs the single copy from ``data`` into shared memory; the
        caller is expected to have already exposed the source as a flat,
        contiguous byte view (no further copies on the source side).

        Args:
            data: Bytes-like source buffer. Must be flat and contiguous.

        Returns:
            ``(offset, nbytes)`` where ``offset`` is the physical byte
            offset of the payload within the arena's data region and
            ``nbytes`` is the number of bytes written. Both values are
            stored on the matching :py:class:`_ShmMarker` and used by
            :py:meth:`ArenaReaderProtocol.read_binary` to retrieve the
            payload on the reader side.
        """
        ...

    def commit_unit(self) -> int:
        """Close the current unit and return its span.

        Marks the unit as readable by the parent. After this call the
        writer is no longer in a unit; another :py:meth:`begin_unit` is
        required before further writes.

        Returns:
            The unit's *span*: the total byte range the unit occupies in
            the arena. The caller (typically :py:func:`_offload`) embeds
            this in the envelope so the reader can return the whole
            region in one :py:meth:`ArenaReaderProtocol.end_unit` call.
        """
        ...

    def abort_unit(self) -> None:
        """Roll back the in-progress unit on failure.

        Discards any binaries written since :py:meth:`begin_unit` and
        leaves the writer outside of a unit, ready for another
        :py:meth:`begin_unit`. Called when offloading raises (e.g. an
        unpicklable field or a reservation failure) so the writer is not
        left mid-unit.

        Returns:
            None.
        """
        ...


class ArenaReaderProtocol(Protocol):
    """Reader endpoint: the first step in the parent process.

    Backends implement this protocol to surface offloaded binaries to
    :py:func:`_restore`. Methods are called from the
    parent process's consumer thread and need not be thread-safe.
    """

    zero_copy: bool
    """Whether :py:meth:`read_binary` hands out live, reusable views into
    shared memory (segment pool) rather than private copies (ring).
    Handlers use this flag to choose between a zero-copy view and an
    owning / copied-out restore."""

    def reset(self) -> None:
        """Reset per-iteration state at an iteration boundary.

        Called by the parent once per iteration, after the worker has
        signaled ``ITERATION_STARTED`` and before the first item is
        consumed. Backends should clear any per-iteration cursors so the
        reader matches the writer's freshly-reset state.

        Returns:
            None.
        """
        ...

    def read_binary(self, offset: int, nbytes: int) -> "memoryview[bytes]":
        """Return the bytes for one offloaded binary.

        Args:
            offset: Physical byte offset in the arena (as returned by
                :py:meth:`ArenaWriterProtocol.write_binary`).
            nbytes: Number of bytes of the payload.

        Returns:
            A :py:class:`memoryview` of exactly ``nbytes`` bytes. With
            ``zero_copy=True`` (segment pool) the view aliases shared
            memory and remains valid as long as a corresponding anchor is
            kept alive (see :py:meth:`end_unit`); with ``zero_copy=False``
            (ring) the bytes have already been copied out into a private
            per-unit buffer.
        """
        ...

    def end_unit(self, span: int, pinned: list[object]) -> None:
        """Signal that one unit has been fully read.

        Args:
            span: The unit's span as returned by
                :py:meth:`ArenaWriterProtocol.commit_unit`. Lets the
                backend release the entire region in bulk regardless of
                how the binaries are laid out.
            pinned: Lifetime anchors for any zero-copy views handed out
                from this unit (e.g. ``np.frombuffer`` arrays). The
                backend must keep the underlying region alive until every
                anchor in this list is released; with copy-out backends
                the list is typically empty and ignored.

        Returns:
            None. Called from a ``finally`` block in
            :py:func:`_restore`, so accounting stays
            correct even if unpickling raised.
        """
        ...


class ArenaProtocol(Protocol):
    """An arena object passed to ``iterate_in_subprocess`` via ``arena``.

    Owns the shared-memory segment and produces the writer / reader
    endpoints used on each side of the inter-process boundary. The
    arena is created and owned by the parent, passed to the worker
    through process spawn arguments, and torn down by the parent after
    the worker is confirmed dead.
    """

    def open_writer(self) -> ArenaWriterProtocol:
        """Open the writer endpoint for the worker side.

        Should be called once per arena, in the worker process, before
        the first item is offloaded. The returned writer is the last
        step in the worker pipeline.

        Returns:
            A fresh :py:class:`ArenaWriterProtocol` bound to this arena.
        """
        ...

    def open_reader(self) -> ArenaReaderProtocol:
        """Open the reader endpoint for the parent side.

        Should be called once per arena, in the parent process, before
        the first item is consumed. The returned reader is the first
        step in the parent pipeline.

        Returns:
            A fresh :py:class:`ArenaReaderProtocol` bound to this arena.
        """
        ...

    def close(self) -> None:
        """Release this process's handle to the shared-memory segment.

        Called by the parent during teardown after the worker has been
        confirmed dead. Does not destroy the OS-level segment — see
        :py:meth:`unlink` for that. Safe to call once per arena.

        Returns:
            None.
        """
        ...

    def unlink(self) -> None:
        """Destroy the OS-level shared-memory segment.

        Called by the parent during teardown, in a ``finally`` after
        :py:meth:`close`, so a failing ``close`` cannot leak the segment.
        Must be called exactly once per arena and only after every
        process that mapped the segment has released its handle.

        Returns:
            None.
        """
        ...
