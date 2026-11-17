# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Public structural interfaces shared by arena backends.

These let ``iterate_in_subprocess`` accept any arena object through one
``arena`` argument using structural conformance. The single ring
(:py:class:`~spdl.pipeline.SharedMemoryRingBuffer`) and the rotating
segment pool (:py:class:`~spdl.pipeline.SharedMemorySegmentPool`)
both satisfy these protocols structurally.

A unit is one pipeline item (possibly several binaries). The writer brackets a
unit with ``begin_unit`` / ``commit_unit`` and the reader returns it with
``end_unit``, so reservation and return happen in bulk per unit regardless of
the backend. Payload adaptation and reconstruction stay in the serializer,
not in the byte-storage endpoints. Custom backends implement these ordinary
methods without importing or inheriting the protocols at runtime.
"""

from __future__ import annotations

from typing import Protocol

__all__ = [
    "ArenaProtocol",
    "ArenaReaderProtocol",
    "ArenaWriterProtocol",
    "_is_arena_backend",
]


def _is_arena_backend(arena: object) -> bool:
    return all(
        callable(getattr(arena, name, None))
        for name in (
            "open_writer",
            "open_reader",
            "shutdown_arena",
            "close",
            "unlink",
        )
    )


class ArenaWriterProtocol(Protocol):
    """ArenaWriterProtocol()

    Writer endpoint: the last step in the worker process.

    Backends implement this protocol to receive offloaded binaries. A *unit*
    is one pipeline item: it groups all binaries produced by a single
    :py:func:`_offload` call so they can be reserved
    and returned to the arena in bulk. Methods are called from a single
    thread in the worker; they need not be thread-safe. In a process-region
    channel the sending role may instead belong to the parent.

    .. versionadded:: 0.7.0

    .. versionchanged:: 0.9.0
       This extension contract is publicly available from ``spdl.pipeline``.
    """

    def reset(self) -> None:
        """Prepare the writer for a new iteration.

        Called by the worker once per ``START_ITERATION``. Copy-out backends
        may reclaim fully consumed storage here, but must not discard unread
        committed units. Zero-copy backends must preserve any units whose
        restored views or native storage exports are alive across the boundary.

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

        Raises:
            BufferError: Capacity is unavailable under the backend's bounded
                waiting policy, or the backend is shutting down.
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
            stored by the serializer as placement metadata and used by
            :py:meth:`ArenaReaderProtocol.read_binary` to retrieve the
            payload on the reader side. No payload-class interpretation is
            requested of the backend.

        Raises:
            BufferError: The fragment or accumulated unit exceeds the backend's
                configured capacity.
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
            The span includes any backend padding. Committed queued units must
            remain readable until consumed or explicitly discarded.
        """
        ...

    def abort_unit(self) -> None:
        """Roll back the in-progress unit on failure.

        Discards any binaries written since :py:meth:`begin_unit` and
        leaves the writer outside of a unit, ready for another
        :py:meth:`begin_unit`. Called when offloading raises (e.g. an
        unpicklable field or a reservation failure) so the writer is not
        left mid-unit. Previously committed units, surviving native exports
        and asynchronous transfer owners remain unaffected.

        Returns:
            None.
        """
        ...


class ArenaReaderProtocol(Protocol):
    """ArenaReaderProtocol()

    Reader endpoint: the first step in the parent process.

    Backends implement this protocol to surface offloaded binaries to
    :py:func:`_restore`. Methods are called from the
    parent process's consumer thread and need not be thread-safe. A
    process-region channel may reverse the sender/receiver process roles.

    .. versionadded:: 0.7.0

    .. versionchanged:: 0.9.0
       This extension contract is publicly available from ``spdl.pipeline``.
    """

    @property
    def zero_copy(self) -> bool:
        """Whether :py:meth:`read_binary` hands out live, reusable views into
        shared memory (segment pool) rather than private copies (ring).
        Handlers use this flag to choose between a zero-copy view and an
        owning / copied-out restore.
        """
        ...

    def reset(self) -> None:
        """Prepare the reader for a new iteration.

        Called by the parent once per iteration, after the worker has
        signaled ``ITERATION_STARTED`` and before the first item is consumed.
        Zero-copy backends must preserve the accounting for views that remain
        alive from previous iterations.

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
            memory and its actual exporter must retain the backing for every
            surviving storage export (see :py:meth:`end_unit`). With
            ``zero_copy=False`` (ring), the bytes have already been copied out
            into an independent owning buffer. Slices, casts, NumPy base chains,
            Torch storage and incomplete transfer sources must retain the
            correct backing; releasing the original Python wrapper alone does
            not prove physical storage is reusable.
        """
        ...

    def end_unit(self, span: int, anchors: list[object] | None = None) -> None:
        """Signal that one unit has been fully read.

        Args:
            span: The unit's span as returned by
                :py:meth:`ArenaWriterProtocol.commit_unit`. Lets the
                backend release the entire region in bulk regardless of
                how the binaries are laid out.
            anchors: Opaque lifetime owners supplied by the serializer for
                views restored from this unit, for example an array base owner.
                This metadata is for custom readers; it does not imply CUDA
                pinning. Builtins ignore the list because native exports
                independently retain their backing. Defaults to ``None``; an
                omitted, second-positional or ``anchors=`` argument is valid.
                SPDL continues passing an actual list positionally to custom
                readers whose existing implementation requires one.

        Logical end does not invalidate surviving exports or establish physical
        reuse: the backend must preserve their storage and ordering constraints.
        This method also accounts for an unread unit discarded without exporting
        bytes; the serializer supplies an empty list in that case.

        Returns:
            None. Called from a ``finally`` block in
            :py:func:`_restore`, so accounting stays
            correct even if unpickling raised.

        .. versionchanged:: 0.9.0
           Renamed ``pinned`` to optional ``anchors`` with a ``None`` default.
           Positional calls remain valid; replace the old ``pinned=`` keyword
           with ``anchors=``.
        """
        ...


class ArenaProtocol(Protocol):
    """ArenaProtocol()

    An arena object passed to ``iterate_in_subprocess`` via ``arena``.

    Owns the shared-memory segment and produces the writer / reader
    endpoints used on each side of the inter-process boundary. The
    arena is created and owned by the parent, passed to the worker
    through process spawn arguments, and torn down by the parent after
    the worker is confirmed dead. Custom objects conform structurally;
    importing or subclassing this protocol is not required. Ordinary iteration
    owns arena teardown; process-region channels borrow caller-owned arenas.

    .. versionadded:: 0.7.0

    .. versionchanged:: 0.9.0
       This extension contract is publicly available from ``spdl.pipeline``.
    """

    def open_writer(self) -> ArenaWriterProtocol:
        """Open the writer endpoint for the worker side.

        Should be called once per arena, in the worker process, before
        the first item is offloaded. The returned writer is the last
        step in the worker pipeline. A process-region channel opens it in its
        sending process, which may instead be the parent.

        Returns:
            A fresh :py:class:`ArenaWriterProtocol` bound to this arena.
        """
        ...

    def open_reader(self) -> ArenaReaderProtocol:
        """Open the reader endpoint for the parent side.

        Should be called once per arena, in the parent process, before
        the first item is consumed. The returned reader is the first
        step in the parent pipeline. A process-region channel opens it in its
        receiving process; consumer-local pinning is the backend's policy.

        Returns:
            A fresh :py:class:`ArenaReaderProtocol` bound to this arena.
        """
        ...

    def shutdown_arena(self) -> None:
        """Request shutdown and wake producers blocked on arena capacity.

        Called before joining the sending process, so a producer waiting inside
        :py:meth:`ArenaWriterProtocol.begin_unit` or
        :py:meth:`ArenaWriterProtocol.write_binary` can observe shutdown and
        leave its wait. IPC callers use this operation through the protocol,
        not a backend-specific private hook.

        This operation is repeatable and must not release mappings, unregister
        live storage or unlink shared names. Existing byte exports and pending
        transfers retain their storage. A shutdown request is not proof that
        the worker has exited or that storage is safe to retire; complete
        :py:meth:`close` and :py:meth:`unlink` only after safe teardown.

        Returns:
            None.

        .. versionchanged:: 0.9.0
           Custom arena backends must expose the pre-join shutdown operation
           explicitly as part of this structural contract.
        """
        ...

    def close(self) -> None:
        """Release this process's handle to the shared-memory segment.

        Called by the parent during teardown after the worker has been
        confirmed dead. Does not destroy the OS-level segment; see
        :py:meth:`unlink` for that. Safe to call once per arena. Backends that
        provide terminal cancellation through ``close`` may also use it before
        worker join, but final physical retirement must wait for worker and
        native-export/transfer quiescence. Live or uncertain storage must remain
        retained rather than being unmapped or overwritten.

        Returns:
            None.
        """
        ...

    def unlink(self) -> None:
        """Destroy the OS-level shared-memory segment.

        Called by the owning parent during teardown after safe
        :py:meth:`close` retirement. If close fails with uncertain storage or
        worker state, retain ownership and names until safe retry is possible;
        unlink is not a substitute for proving quiescence. Must be called once
        per arena by its owner and only after every process that mapped the
        segment has released its handle. Removing names must not invalidate
        surviving mapped aliases.

        Returns:
            None.
        """
        ...
