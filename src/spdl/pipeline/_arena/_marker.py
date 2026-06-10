# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Marker referencing an offloaded binary stored in a shared-memory arena."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["_ShmMarker"]


@dataclass(frozen=True)
class _ShmMarker:
    """Replaces an offloaded binary in the pickled envelope on the queue.

    An instance is produced by the pickler's ``persistent_id`` on the writer
    side and consumed by ``persistent_load`` on the reader side.
    """

    offset: int
    """Physical byte offset of the payload within the arena's data region."""

    nbytes: int
    """Number of bytes of the payload."""

    kind: str
    """Registry key identifying how to reconstruct the object (e.g. ``"bytes"``)."""

    meta: tuple[object, ...]
    """Extra data needed to reconstruct (e.g. ``(dtype, shape)`` for arrays)."""
