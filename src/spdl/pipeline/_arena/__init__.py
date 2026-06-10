# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared-memory arena for moving large payloads across processes."""

from ._arena import _Arena
from ._pool import SharedMemorySegmentPool
from ._protocol import ArenaProtocol, ArenaReaderProtocol, ArenaWriterProtocol

__all__ = [
    "ArenaProtocol",
    "ArenaReaderProtocol",
    "ArenaWriterProtocol",
    "SharedMemorySegmentPool",
    "_Arena",
]
