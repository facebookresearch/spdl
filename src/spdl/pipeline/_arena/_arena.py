# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Per-process wrapper bundling an arena with its endpoints and registry."""

from __future__ import annotations

from ._offload import _offload, _restore
from ._protocol import ArenaProtocol, ArenaReaderProtocol, ArenaWriterProtocol
from ._registry import _default_registry, _OffloadRegistry

__all__ = ["_Arena"]


class _Arena:
    """Bundles an arena with its lazily-created writer / reader / registry.

    The worker side offloads through :py:attr:`writer` + :py:attr:`registry`; the
    parent side restores through :py:attr:`reader` + :py:attr:`registry`. Each
    endpoint and the registry is created on first access and cached (delayed
    initialization), so a process only opens the side it actually uses — the
    worker never opens a reader, the parent never opens a writer.

    Construct one per process from the shared arena (or skip it entirely when no
    arena is in use), so a single ``if arena is not None`` covers "is an arena in
    use?" without juggling the endpoint and registry as separate optionals.
    """

    def __init__(self, arena: ArenaProtocol) -> None:
        self._arena = arena
        self._writer: ArenaWriterProtocol | None = None
        self._reader: ArenaReaderProtocol | None = None
        self._registry: _OffloadRegistry | None = None

    @property
    def writer(self) -> ArenaWriterProtocol:
        """The arena's writer endpoint (opened once, on first access)."""
        if self._writer is None:
            self._writer = self._arena.open_writer()
        return self._writer

    @property
    def reader(self) -> ArenaReaderProtocol:
        """The arena's reader endpoint (opened once, on first access)."""
        if self._reader is None:
            self._reader = self._arena.open_reader()
        return self._reader

    @property
    def registry(self) -> _OffloadRegistry:
        """The offload registry (created once, on first access)."""
        if self._registry is None:
            self._registry = _default_registry()
        return self._registry

    def offload(self, obj: object) -> bytes:
        """Offload ``obj``'s large fields into the arena; return the envelope.

        Worker-side convenience over :py:func:`_offload` bound to this object's
        :py:attr:`writer` and :py:attr:`registry`.

        Args:
            obj: The (picklable) item to offload.

        Returns:
            The small envelope to put on the inter-process queue; pass it to
            :py:meth:`restore` on the parent side.
        """
        return _offload(obj, self.writer, self.registry)

    def restore(self, blob: bytes) -> object:
        """Reconstruct an object produced by :py:meth:`offload`.

        Parent-side convenience over :py:func:`_restore` bound to this object's
        :py:attr:`reader` and :py:attr:`registry`.

        Args:
            blob: The envelope produced by :py:meth:`offload`.

        Returns:
            The restored object (offloaded fields aliasing shared memory or
            copied out, per the backend).
        """
        return _restore(blob, self.reader, self.registry)
