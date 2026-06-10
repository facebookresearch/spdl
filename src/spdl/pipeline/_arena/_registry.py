# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Registry deciding which objects are offloaded to the arena and how.

A handler decides whether an object should be offloaded (``matches``), exposes
its raw bytes as a zero-copy source buffer (``get_buffer``) so the arena can copy
it into shared memory exactly once, and rebuilds it from a destination buffer
(``from_buffer``).

``from_buffer`` receives the destination buffer plus a ``zero_copy`` flag: true
for the segment pool (whose buffer is a live, reusable view into shared memory),
false for the ring (which already copied the bytes out into a private per-unit
buffer). A handler restores a zero-copy view when ``zero_copy`` is true and an
owning / copied-out value when it is false — so the restored type can differ
between backends (see :py:class:`_BytesHandler`).

``from_buffer`` returns ``(object, anchor)``: the restored object, plus an
*anchor* whose lifetime equals "some view of the shared-memory segment is still
alive" (e.g. the ``np.frombuffer`` array, or a restored ``memoryview`` itself),
or ``None`` when the object does not alias a reusable segment — i.e. always on the
ring (copied out), and on the pool only for the off-switch copy paths. For the
segment-pool backend a non-``None`` anchor keeps the segment reserved until it is
garbage-collected. The anchor is returned separately because views and shallow
copies of a NumPy array / Torch tensor share the buffer but are different objects,
and — importantly — NumPy/Torch do not retain the exact ``memoryview`` passed in,
so it cannot be used as the anchor (a plain ``memoryview``, by contrast, *is* its
own anchor — it keeps the buffer exported and is weak-referenceable).

NumPy and Torch are reached through ``lazy_import`` so ``import spdl.pipeline``
works without them installed and they never become hard Buck dependencies. A
handler only touches its module after a cheap module-name check, so unrelated
objects never trigger the import.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Protocol

from spdl._internal.import_utils import lazy_import

__all__ = ["_OffloadHandlerProtocol", "_OffloadRegistry", "_default_registry"]

np: ModuleType = lazy_import("numpy")
torch: ModuleType = lazy_import("torch")


class _OffloadHandlerProtocol(Protocol):
    """Interface for a type that can be offloaded to the arena.

    A handler teaches the registry how to move one family of objects (e.g.
    ``bytes``, NumPy arrays, Torch tensors) through a shared-memory arena: whether
    a given object is worth offloading (:py:meth:`matches`), how to expose its
    bytes as a single source buffer for the one copy into shared memory
    (:py:meth:`get_buffer`), and how to rebuild it on the reader side
    (:py:meth:`from_buffer`). Third-party handlers implement this protocol and are
    passed to :py:class:`_OffloadRegistry`.
    """

    kind: str
    """Stable registry key identifying this handler (e.g. ``"bytes"``,
    ``"numpy"``). Stored on each :py:class:`_ShmMarker` so :py:func:`_restore` can
    look the handler back up; must be unique within a registry."""

    def matches(self, obj: object, threshold: int) -> bool:
        """Return whether this handler should offload ``obj``.

        Called for every leaf the pickler encounters, so it must be cheap and
        side-effect-free for objects it does not handle (e.g. probe the type name
        before touching a heavy, lazily-imported module).

        Args:
            obj: The candidate leaf object being pickled.
            threshold: Minimum payload size in bytes for offloading to be
                worthwhile; objects smaller than this should be left inline
                (return ``False``).

        Returns:
            ``True`` if this handler will offload ``obj`` (the registry then calls
            :py:meth:`get_buffer`), ``False`` to leave it to another handler or to
            inline pickling.
        """
        ...

    def get_buffer(
        self, obj: object
    ) -> tuple["bytes | bytearray | memoryview[bytes]", tuple[object, ...]]:
        """Expose ``obj``'s payload as one source buffer plus rebuild metadata.

        Should not copy: return a flat, contiguous view over the object's own
        bytes — the arena's ``write_binary`` performs the single copy into shared
        memory. Any non-byte information needed to reconstruct the object (dtype,
        shape, original type, ...) is returned as ``meta``.

        Args:
            obj: The object to offload; guaranteed to be one that this handler's
                :py:meth:`matches` accepted.

        Returns:
            A ``(buffer, meta)`` pair. ``buffer`` is a bytes-like, contiguous
            source for the single copy into the arena; ``meta`` is a picklable
            tuple carried on the :py:class:`_ShmMarker` and handed back to
            :py:meth:`from_buffer` to rebuild the object.
        """
        ...

    def from_buffer(
        self, buf: "memoryview[bytes]", meta: tuple[object, ...], zero_copy: bool
    ) -> tuple[object, object | None]:
        """Reconstruct the object from its payload bytes in the arena.

        Args:
            buf: The payload bytes read back from the arena. When ``zero_copy`` is
                ``True`` this aliases shared memory and stays valid only while a
                returned anchor is alive; when ``False`` it is a private copy.
            meta: The tuple produced by :py:meth:`get_buffer` for this object.
            zero_copy: ``True`` if ``buf`` is a live, reusable view into shared
                memory (segment pool); ``False`` if it was already copied out
                (ring). Handlers restore a zero-copy view in the former case and
                an owning value in the latter, so the restored type may differ
                between backends.

        Returns:
            An ``(object, anchor)`` pair: the restored object, plus an *anchor*
            whose lifetime keeps any zero-copy view of ``buf`` valid — collected
            by :py:func:`_restore` and handed to
            :py:meth:`ArenaReaderProtocol.end_unit` — or ``None`` when the
            restored object does not alias a reusable segment.
        """
        ...


class _BytesHandler:
    """Offload bytes-like objects: ``bytes``, ``bytearray``, and ``memoryview``.

    All three are written into the arena identically. Restore is
    backend-dependent:

    * Ring buffer (copy backend): the bytes are copied out and the *original* type
      is reconstructed — ``bytes`` -> ``bytes``, ``bytearray`` -> ``bytearray``,
      ``memoryview`` -> ``memoryview`` (over the private per-unit copy). Type (and,
      for ``bytes``/``bytearray``, mutability) matches the input.
    * Segment pool (zero-copy backend): the copy is deferred and **all three**
      return a ``memoryview`` aliasing shared memory. That memoryview is its own
      lifetime anchor (it keeps the buffer exported and is weak-referenceable), so
      the pool holds the segment until it is released. The restored value is
      therefore a (writable) ``memoryview`` regardless of the input type — wrap it
      in ``bytes(...)`` / ``bytearray(...)`` for an owning copy. This discrepancy
      is the price of the opt-in zero-copy path.

    ``meta`` carries the original type so the ring can rebuild it.
    """

    kind: str = "bytes"

    def matches(self, obj: object, threshold: int) -> bool:
        if not isinstance(obj, (bytes, bytearray, memoryview)):
            return False
        nbytes = obj.nbytes if isinstance(obj, memoryview) else len(obj)
        return nbytes >= threshold

    def get_buffer(
        self, obj: object
    ) -> tuple["bytes | bytearray | memoryview[bytes]", tuple[object, ...]]:
        # Expose the input's bytes as a flat, contiguous byte view — no copy for
        # the common contiguous case (write_binary performs the single copy into
        # shared memory); a non-contiguous memoryview is flattened into a copy.
        # meta carries the original type so the ring backend can reconstruct it.
        mv = memoryview(obj)
        src = mv.cast("B") if mv.contiguous else memoryview(mv.tobytes())
        return src, (type(obj),)

    def from_buffer(
        self, buf: "memoryview[bytes]", meta: tuple[object, ...], zero_copy: bool
    ) -> tuple[object, object | None]:
        if zero_copy:
            # Segment pool: defer the copy — hand back a zero-copy memoryview into
            # shared memory. It is its own lifetime anchor, so the pool holds the
            # segment until the view is released. (bytes/bytearray inputs come back
            # as memoryview here; that is the accepted zero-copy discrepancy.)
            return buf, buf
        # Ring: the bytes were copied out into a private per-unit buffer — rebuild
        # the original type so the restored value matches what was sent. No anchor:
        # nothing aliases a reusable segment.
        (orig_type,) = meta
        if orig_type is bytearray:
            return bytearray(buf), None
        if orig_type is memoryview:
            return buf, None  # ``buf`` already views the private per-unit copy
        return bytes(buf), None


class _NumpyHandler:
    kind: str = "numpy"

    def matches(self, obj: object, threshold: int) -> bool:
        # Cheap module-name probe first so unrelated objects never realize the
        # lazily-imported numpy module; then duck-type (the lazy module is typed
        # ``Any``, so ``isinstance`` cannot narrow here).
        if type(obj).__module__.split(".", 1)[0] != "numpy":
            return False
        if type(obj).__name__ != "ndarray":
            return False
        arr: Any = obj
        return int(arr.nbytes) >= threshold

    def get_buffer(
        self, obj: object
    ) -> tuple["bytes | bytearray | memoryview[bytes]", tuple[object, ...]]:
        arr: Any = np.ascontiguousarray(obj)
        # memoryview over the array's own buffer — no copy (the single copy into
        # shared memory happens in the arena's write_binary).
        return memoryview(arr).cast("B"), (arr.dtype.str, tuple(arr.shape))

    def from_buffer(
        self, buf: "memoryview[bytes]", meta: tuple[object, ...], zero_copy: bool
    ) -> tuple[object, object | None]:
        dtype_str, shape = meta
        # np.frombuffer aliases ``buf`` (no copy); for the segment pool this is a
        # live view into shared memory, for the ring it views the copied-out
        # bytes that read_binary already produced. The anchor matters only for the
        # pool: ``arr`` is its lifetime anchor (every reshape/slice/view of the
        # result keeps it — and thus the segment — alive, whereas the reshaped
        # object can be dropped while a view of it survives). On the ring the bytes
        # are already a private copy the array keeps alive, so no anchor is needed.
        arr: Any = np.frombuffer(buf, dtype=np.dtype(dtype_str))
        return arr.reshape(shape), (arr if zero_copy else None)


class _TorchHandler:
    kind: str = "torch"

    def matches(self, obj: object, threshold: int) -> bool:
        if type(obj).__module__.split(".", 1)[0] != "torch":
            return False
        if type(obj).__name__ != "Tensor":
            return False
        t: Any = obj
        if not t.is_cpu:
            return False
        return int(t.element_size()) * int(t.nelement()) >= threshold

    def get_buffer(
        self, obj: object
    ) -> tuple["bytes | bytearray | memoryview[bytes]", tuple[object, ...]]:
        t: Any = obj.detach().contiguous().cpu()  # type: ignore[attr-defined]
        # t.numpy() shares the tensor's storage (no copy for a contiguous CPU
        # tensor); memoryview over it stays a view.
        return memoryview(t.numpy()).cast("B"), (str(t.dtype), tuple(t.shape))

    def from_buffer(
        self, buf: "memoryview[bytes]", meta: tuple[object, ...], zero_copy: bool
    ) -> tuple[object, object | None]:
        dtype_str, shape = meta
        dtype = getattr(torch, str(dtype_str).split(".")[-1])
        # As for NumPy, ``t`` is the pool's lifetime anchor (views keep the base
        # tensor alive); the ring copied the bytes out, so no anchor is needed.
        t: Any = torch.frombuffer(buf, dtype=dtype)
        return t.reshape(shape), (t if zero_copy else None)


class _PacketsHandler:
    """Offload ``spdl.io`` ``AudioPackets`` / ``VideoPackets`` / ``ImagePackets``.

    Packets wrap compressed media buffers — exactly the large binaries this
    feature targets — but expose no array-style buffer protocol. Their bytes are
    only reachable through ``serialize_packets`` (the C++ ``__getstate__``, the
    same path pickle takes), and that call is also the only way to learn their
    size. To avoid serializing twice, ``matches`` (which must size the object)
    caches the bytes for the ``get_buffer`` that immediately follows it.

    On restore the bytes are handed to the class's public ``deserialize_view``
    static method (carried in ``meta``), which builds a Packets whose low-level
    ``AVPacket``s point *directly into* the arena buffer — a zero-copy view, the
    same treatment NumPy/Torch get. The restored object is returned as its own
    lifetime anchor, so the segment-pool backend keeps the segment until the
    Packets are released; the ring backend already copied the bytes out, so the
    view aliases that copy instead. Using only ``__getstate__`` /
    ``deserialize_view`` keeps this handler free of any ``spdl.io`` import: the
    Packets class travels through the pickled marker.
    """

    kind: str = "packets"

    _NAMES: frozenset[str] = frozenset(("AudioPackets", "VideoPackets", "ImagePackets"))

    def __init__(self) -> None:
        # (id(obj), serialized bytes) of the most recent matched object, so the
        # get_buffer call that follows matches reuses the serialization.
        self._cache: tuple[int, bytes] | None = None

    def matches(self, obj: object, threshold: int) -> bool:
        # Cheap type-name probe first (import-free), then confirm it is really an
        # spdl.io Packets before touching ``__getstate__``: a foreign class that
        # merely shares the name could return non-bytes from ``__getstate__`` and
        # corrupt the arena write. The NumPy/Torch handlers disambiguate on the
        # module name, but the Packets extension module is FFmpeg-version- and
        # build-specific (``_spdl_ffmpeg*``), so it is not a reliable signal here;
        # instead require the ``deserialize_view`` static method this handler
        # depends on to restore — present only on genuine spdl.io Packets.
        cls = type(obj)
        if cls.__name__ not in self._NAMES:
            return False
        if not hasattr(cls, "deserialize_view"):
            return False
        data: bytes = obj.__getstate__()  # type: ignore[attr-defined]
        if len(data) < threshold:
            # Below the offload threshold: leave it inline and do NOT cache —
            # get_buffer will not run, so caching would only pin these bytes
            # until the next matches() call (a leak for a stream of small
            # Packets).
            return False
        self._cache = (id(obj), data)
        return True

    def get_buffer(
        self, obj: object
    ) -> tuple["bytes | bytearray | memoryview[bytes]", tuple[object, ...]]:
        if self._cache is not None and self._cache[0] == id(obj):
            data = self._cache[1]
        else:
            # Defensive: matches always runs first and caches, but re-serialize
            # rather than rely on it (the bytes own their memory either way).
            data = obj.__getstate__()  # type: ignore[attr-defined]
        self._cache = None
        # memoryview over the bytes — no copy; write_binary performs the single
        # copy into shared memory. meta carries the class so restore can call
        # its ``deserialize_view`` (pickled by reference in the marker).
        return memoryview(data), (type(obj),)

    def from_buffer(
        self, buf: "memoryview[bytes]", meta: tuple[object, ...], zero_copy: bool
    ) -> tuple[object, object | None]:
        (cls,) = meta
        if zero_copy:
            # Segment pool: zero-copy view — the restored Packets alias ``buf``
            # directly (the class keeps ``buf`` alive via nb::keep_alive). The
            # Packets object is its own lifetime anchor, so the pool holds the
            # segment until they are released.
            obj = cls.deserialize_view(buf)  # type: ignore[attr-defined]
            return obj, obj
        # Ring: the bytes were copied out — build an owning Packets, no anchor.
        return cls.deserialize(bytes(buf)), None  # type: ignore[attr-defined]


class _OffloadRegistry:
    """Ordered set of handlers used to offload and restore objects.

    Consulted on both sides of the process boundary: the offloading pickler asks
    :py:meth:`handler_for` which handler (if any) claims each leaf, and
    :py:func:`_restore` uses :py:meth:`handler` to look a handler back up by the
    ``kind`` stored on the marker. Order matters — the first matching handler
    wins — so more specific handlers should be listed before more general ones.

    Args:
        handlers: Handlers to consult, in priority order. Each must satisfy
            :py:class:`_OffloadHandlerProtocol` and carry a ``kind`` unique within
            this registry. :py:meth:`handler_for` tries them in this order;
            :py:meth:`handler` indexes them by ``kind``.
    """

    def __init__(self, handlers: list[_OffloadHandlerProtocol]) -> None:
        self._handlers = handlers
        self._by_kind: dict[str, _OffloadHandlerProtocol] = {
            h.kind: h for h in handlers
        }

    def handler_for(
        self, obj: object, threshold: int
    ) -> _OffloadHandlerProtocol | None:
        """Return the first handler that wants to offload ``obj``, else ``None``.

        Args:
            obj: The candidate leaf object being pickled.
            threshold: Minimum payload size in bytes for offloading to be
                worthwhile, passed through to each handler's
                :py:meth:`~_OffloadHandlerProtocol.matches`.

        Returns:
            The first handler (in registration order) whose ``matches`` returns
            ``True``, or ``None`` if no handler claims ``obj`` (it is then pickled
            inline).
        """
        for h in self._handlers:
            if h.matches(obj, threshold):
                return h
        return None

    def handler(self, kind: str) -> _OffloadHandlerProtocol:
        """Return the handler registered under ``kind``.

        Args:
            kind: The registry key stored on a :py:class:`_ShmMarker`, identifying
                the handler that produced it.

        Returns:
            The handler whose ``kind`` equals ``kind``.

        Raises:
            KeyError: If no handler with that ``kind`` is registered.
        """
        return self._by_kind[kind]


def _default_registry() -> _OffloadRegistry:
    """Registry covering large bytes-like objects (``bytes``, ``bytearray``,
    ``memoryview``), NumPy arrays, Torch tensors, and
    ``spdl.io`` Audio/Video/Image Packets."""
    return _OffloadRegistry(
        [_BytesHandler(), _NumpyHandler(), _TorchHandler(), _PacketsHandler()]
    )
