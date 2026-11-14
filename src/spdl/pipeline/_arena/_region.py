# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-local region endpoints for ordinary arena transport."""

from __future__ import annotations

import atexit
import queue
from dataclasses import dataclass
from typing import Any

from ._arena import _Arena
from ._protocol import ArenaProtocol
from ._registry import _default_registry, _OffloadHandlerProtocol, _OffloadRegistry

__all__ = [
    "_ArenaEnvelope",
    "_ArenaSendingQueue",
    "_ArenaInputQueue",
    "_ArenaOutputQueue",
    "_RegionChannel",
]


@dataclass(frozen=True)
class _ArenaEnvelope:
    worker: int
    sequence: int
    blob: bytes


class _RegionBytesHandler:
    kind = "bytes"

    def __init__(self, original: _OffloadHandlerProtocol) -> None:
        self._original = original

    def matches(self, obj: object, threshold: int) -> bool:
        return self._original.matches(obj, threshold)

    def get_buffer(self, obj: object) -> Any:
        return self._original.get_buffer(obj)

    def from_buffer(
        self, buf: "memoryview[bytes]", meta: tuple[object, ...], zero_copy: bool
    ) -> tuple[object, object | None]:
        if meta[0] in (bytes, bytearray):
            return self._original.from_buffer(buf, meta, False)
        return self._original.from_buffer(buf, meta, zero_copy)


class _RegionRegistry(_OffloadRegistry):
    def __init__(self) -> None:
        self._original = _default_registry()
        self._bytes = _RegionBytesHandler(self._original.handler("bytes"))

    def handler_for(
        self, obj: object, threshold: int
    ) -> _OffloadHandlerProtocol | None:
        handler = self._original.handler_for(obj, threshold)
        return (
            self._bytes if handler is not None and handler.kind == "bytes" else handler
        )

    def handler(self, kind: str) -> _OffloadHandlerProtocol:
        return self._bytes if kind == "bytes" else self._original.handler(kind)


class _RegionChannel:
    """One persistent endpoint for a worker's ordered arena transfers."""

    def __init__(self, worker: int, arena: ArenaProtocol) -> None:
        self._worker = worker
        self.arena = _Arena(arena, registry=_RegionRegistry())
        self._sequence = 0
        self.failed = False

    def encode(self, chunk: list[Any]) -> _ArenaEnvelope:
        if self.failed:
            raise RuntimeError("Arena region channel is no longer usable")
        blob = self.arena.offload(chunk)
        envelope = _ArenaEnvelope(self._worker, self._sequence, blob)
        self._sequence += 1
        return envelope

    def decode(self, envelope: Any, discard: bool = False) -> list[Any] | None:
        if self.failed:
            raise RuntimeError("Arena region channel is no longer usable")
        if (
            type(envelope) is not _ArenaEnvelope
            or type(envelope.worker) is not int
            or envelope.worker != self._worker
            or type(envelope.sequence) is not int
            or envelope.sequence != self._sequence
            or type(envelope.blob) is not bytes
        ):
            self.failed = True
            raise RuntimeError(
                "Invalid arena region envelope: worker or transfer sequence mismatch"
            )
        self._sequence += 1
        try:
            if discard:
                self.arena.discard(envelope.blob)
                return None
            chunk = self.arena.restore(envelope.blob)
            if type(chunk) is not list:
                raise RuntimeError("Arena region transfer must restore a list")
            return chunk
        except BaseException:
            self.failed = True
            raise


class _ArenaSendingQueue:
    """Encode once and retain the exact committed descriptor across queue Full retries."""

    def __init__(self, q: Any, kind: int, channel: _RegionChannel) -> None:
        self._q, self._kind, self._channel = q, kind, channel
        self._pending: tuple[Any, tuple[int, _ArenaEnvelope]] | None = None
        self._pending_retained = False
        self._pending_cleanup_error: BaseException | None = None
        self._accepted_cleanup_message: Any | None = None
        self._accepted_cleanup_failed = False

    def _retain_pending(self) -> None:
        if not self._pending_retained:
            atexit.register(self._release_retired_after_close)
            self._pending_retained = True

    def _release_pending_root(self) -> None:
        if self._pending_retained:
            atexit.unregister(self._release_retired_after_close)
            self._pending_retained = False
            self._pending_cleanup_error = None
            self._accepted_cleanup_message = None

    def _release_retired_after_close(self) -> None:
        # Only discard references, never borrow the caller's close/unlink role.
        # Unknown custom terminal state stays retained rather than guessed safe.
        if (
            self._pending is None
            and not self._channel.failed
            and not self._accepted_cleanup_failed
        ):
            self._release_pending_root()
            return
        arena = self._channel.arena.arena
        if (
            (self._channel.failed or self._accepted_cleanup_failed)
            and getattr(arena, "_owns_names", False) is True
            and getattr(arena, "_closed", False) is True
        ):
            self._pending = None
            self._release_pending_root()

    def put(self, obj: Any, block: bool = True, timeout: float | None = None) -> None:
        if obj[0] != self._kind:
            self._q.put(obj, block=block, timeout=timeout)
            return
        if self._accepted_cleanup_failed:
            raise RuntimeError(
                "Previous publication was already accepted; "
                "cleanup failure retired this sender"
            )
        if self._channel.failed:
            raise RuntimeError("Arena region channel is no longer usable")
        if self._pending is None:
            self._retain_pending()
            try:
                self._pending = (obj, (self._kind, self._channel.encode(obj[1])))
            except BaseException:
                self._channel.failed = True
                raise
        elif self._pending[0] is not obj:
            raise RuntimeError("Concurrent arena region publications are not supported")
        try:
            self._q.put(self._pending[1], block=block, timeout=timeout)
        except queue.Full:
            raise
        except BaseException:
            self._channel.failed = True
            raise
        else:
            # Checkpoint known acceptance before cleanup can raise control flow.
            self._accepted_cleanup_message = obj
            self._pending = None
            try:
                self._release_pending_root()
            except Exception as error:
                # The queue already accepted the descriptor. Keep cleanup retry
                # state without turning accepted delivery into a publication error.
                self._accepted_cleanup_failed = True
                self._pending_cleanup_error = error
            except BaseException as error:  # noqa: B036 - Preserve post-acceptance control flow.
                self._accepted_cleanup_failed = True
                self._pending_cleanup_error = error
                raise
            else:
                self._accepted_cleanup_message = None

    def put_nowait(self, obj: Any) -> None:
        self.put(obj, block=False)

    def _discard_pending(self, obj: Any) -> None:
        if self._pending is not None and self._pending[0] is obj:
            # The queue never accepted this committed span. Its descriptor/root
            # survives cancellation until safe terminal caller cleanup; reclaim
            # cannot jump ahead of earlier queued or exported units.
            self._channel.failed = True


class _ArenaInputQueue:
    def __init__(self, q: Any, kind: int, channel: _RegionChannel) -> None:
        self._q, self._kind, self._channel = q, kind, channel

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        kind, payload = self._q.get(block=block, timeout=timeout)
        if kind == self._kind:
            payload = self._channel.decode(payload)
        return kind, payload

    def get_discarded(self) -> Any:
        kind, payload = self._q.get()
        if kind == self._kind:
            self._channel.decode(payload, discard=True)
            payload = None
        return kind, payload


class _ArenaOutputQueue:
    """Decode each worker's ordered ordinary arena transfers."""

    def __init__(self, q: Any, channels: list[_RegionChannel]) -> None:
        self._q, self._channels = q, channels

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        return self._q.get(block=block, timeout=timeout)

    def decode_result(self, envelope: Any, discard: bool = False) -> list[Any] | None:
        if (
            type(envelope) is not _ArenaEnvelope
            or type(envelope.worker) is not int
            or not 0 <= envelope.worker < len(self._channels)
        ):
            raise RuntimeError("Invalid arena region envelope: unknown worker")
        return self._channels[envelope.worker].decode(envelope, discard)
