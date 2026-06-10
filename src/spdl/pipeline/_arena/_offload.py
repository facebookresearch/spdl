# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Offload large fields of an object into the arena and restore them.

``_offload`` pickles an object with a custom ``persistent_id`` that diverts
offloadable leaves (large ``bytes``, NumPy arrays, Torch tensors, and
``spdl.io`` Packets) into the shared-memory writer, leaving a small
:py:class:`_ShmMarker` in the pickle
stream. ``_restore`` reverses it with a matching ``persistent_load``. Nested
collections (dict/list/dataclass) are handled automatically by pickle.

The returned envelope is ``<8-byte unit span><pickled small object>``; the span
lets the reader return the whole unit's region in one bulk call regardless of
how the binaries are laid out.
"""

from __future__ import annotations

import io
import pickle
import struct

from ._marker import _ShmMarker
from ._protocol import ArenaReaderProtocol, ArenaWriterProtocol
from ._registry import _OffloadRegistry

__all__ = ["_DEFAULT_OFFLOAD_THRESHOLD", "_offload", "_restore"]

_DEFAULT_OFFLOAD_THRESHOLD: int = 4096

_SPAN: struct.Struct = struct.Struct("<Q")


def _offload(
    obj: object,
    writer: ArenaWriterProtocol,
    registry: _OffloadRegistry,
    threshold: int = _DEFAULT_OFFLOAD_THRESHOLD,
) -> bytes:
    """Offload large fields of ``obj`` and return the small picklable envelope.

    Walks ``obj`` with a custom pickler whose ``persistent_id`` diverts each
    offloadable leaf — large ``bytes``, NumPy arrays, Torch tensors, etc., as
    determined by ``registry`` — into ``writer``'s shared-memory region and
    leaves a small :py:class:`_ShmMarker` in the pickle stream. Nested
    collections (``dict`` / ``list`` / dataclass / ...) are traversed by pickle
    automatically, so callers do not need to know the shape of ``obj``.

    The writer is bracketed with ``begin_unit`` / ``commit_unit`` so all
    binaries from a single ``_offload`` call form one unit; on any exception
    (unpicklable object, arena reservation failure, ...) the in-progress unit
    is rolled back via ``abort_unit`` and the original exception re-raised.

    The returned envelope is ``<8-byte unit span><pickled small object>``. The
    span — the total byte range of the unit in the arena — lets the matching
    :py:func:`_restore` return the whole unit's region in one bulk call without
    having to re-walk the binaries.

    Args:
        obj: The object to offload. May be any picklable value; offloadable
            leaves are diverted into the arena, the rest is pickled inline.
        writer: Arena writer endpoint produced by
            :py:meth:`ArenaProtocol.open_writer`. Receives the offloaded
            binaries.
        registry: Decides which leaves get offloaded and how to expose their
            bytes — typically the result of :py:func:`_default_registry`.
        threshold: Minimum size in bytes for a leaf to be eligible for
            offloading. Smaller values are pickled inline. Defaults to
            :py:data:`_DEFAULT_OFFLOAD_THRESHOLD`.

    Returns:
        The small envelope to put on the inter-process queue: an 8-byte
        little-endian unit span followed by the pickled object (with
        :py:class:`_ShmMarker` instances in place of the offloaded leaves).
        Pass it to :py:func:`_restore` on the reader side.

    Example:
        >>> arena = SharedMemoryRingBuffer(capacity=64 * 1024 * 1024)
        >>> writer = arena.open_writer()
        >>> registry = _default_registry()
        >>> envelope = _offload({"img": large_bytes}, writer, registry)
        >>> queue.put(envelope)  # small payload, no large bytes pickled
    """
    writer.begin_unit()
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL)

    def persistent_id(o: object) -> _ShmMarker | None:
        handler = registry.handler_for(o, threshold)
        if handler is None:
            return None
        # get_buffer exposes the object's bytes without copying; write_binary
        # performs the single copy into shared memory.
        src, meta = handler.get_buffer(o)
        offset, nbytes = writer.write_binary(src)
        return _ShmMarker(offset, nbytes, handler.kind, meta)

    pickler.persistent_id = persistent_id  # type: ignore[method-assign]
    try:
        pickler.dump(obj)
    except BaseException:
        # Roll back the in-progress unit so the writer is not left mid-unit
        # (e.g. on an unpicklable object or an arena reservation failure).
        writer.abort_unit()
        raise
    span = writer.commit_unit()
    return _SPAN.pack(span) + buf.getvalue()


def _restore(
    blob: bytes,
    reader: ArenaReaderProtocol,
    registry: _OffloadRegistry,
) -> object:
    """Reconstruct an object produced by :py:func:`_offload`.

    Reads the 8-byte unit span prefix, unpickles the small body with a custom
    ``persistent_load`` that turns each :py:class:`_ShmMarker` back into the
    original object by reading the binary from ``reader`` and asking the
    matching handler in ``registry`` to rebuild it.

    Zero-copy views of shared memory keep their backing region alive through
    *anchors* returned by handlers. Anchors are collected for the duration of
    the call and handed to ``reader.end_unit`` together with the unit span;
    the backend reclaims the unit's region either immediately (ring / copied
    out) or once every anchor is garbage-collected (segment pool / zero-copy).
    ``end_unit`` runs in ``finally`` so accounting stays correct even if
    unpickling raises.

    Args:
        blob: The envelope produced by :py:func:`_offload` — an 8-byte
            little-endian unit span followed by a pickled object with
            :py:class:`_ShmMarker` instances in place of offloaded leaves.
        reader: Arena reader endpoint produced by
            :py:meth:`ArenaProtocol.open_reader`. Provides access to the
            offloaded binaries and signals end-of-unit.
        registry: Same registry shape used on the writer side. Each
            marker's ``kind`` is looked up here to find the handler that
            knows how to rebuild that type.

    Returns:
        The object originally passed to :py:func:`_offload`, with offloaded
        leaves either copied out (ring) or aliasing shared memory through
        anchors held by ``reader`` (segment pool). The exact restored type
        for ``bytes`` / ``bytearray`` / ``memoryview`` depends on the
        backend — see :py:class:`~spdl.pipeline._arena._registry._BytesHandler`.

    Example:
        >>> reader = arena.open_reader()
        >>> registry = _default_registry()
        >>> envelope = queue.get()
        >>> obj = _restore(envelope, reader, registry)
    """
    span = int(_SPAN.unpack_from(blob, 0)[0])
    unpickler = pickle.Unpickler(io.BytesIO(blob[_SPAN.size :]))
    # Lifetime anchors for zero-copy fields (e.g. the np.frombuffer array). The
    # backend keeps the underlying region alive until these are released.
    pinned: list[object] = []

    def persistent_load(marker: object) -> object:
        assert isinstance(marker, _ShmMarker)
        handler = registry.handler(marker.kind)
        view = reader.read_binary(marker.offset, marker.nbytes)
        obj, anchor = handler.from_buffer(view, marker.meta, reader.zero_copy)
        if anchor is not None:
            pinned.append(anchor)
        return obj

    unpickler.persistent_load = persistent_load  # type: ignore[method-assign]
    try:
        return unpickler.load()
    finally:
        # Tell the backend the unit is fully read. It reclaims the region now
        # (ring / copy-out) or once every anchor is released (segment pool /
        # zero-copy). Always runs, so accounting stays correct even on error.
        reader.end_unit(span, pinned)
