# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Shared-memory arena for moving large payloads across processes."""

from ._arena import _Arena
from ._pool import SharedMemorySegmentPool
from ._protocol import (
    _is_arena_backend,
    ArenaProtocol,
    ArenaReaderProtocol,
    ArenaWriterProtocol,
)
from ._region import (
    _ArenaEnvelope,
    _ArenaInputQueue,
    _ArenaOutputQueue,
    _ArenaSendingQueue,
    _RegionChannel,
)
from ._ring import SharedMemoryRingBuffer

__all__ = [
    "ArenaProtocol",
    "ArenaReaderProtocol",
    "ArenaWriterProtocol",
    "SharedMemoryRingBuffer",
    "SharedMemorySegmentPool",
    "_Arena",
    "_ArenaEnvelope",
    "_ArenaInputQueue",
    "_ArenaOutputQueue",
    "_ArenaSendingQueue",
    "_RegionChannel",
    "_is_arena_backend",
]
