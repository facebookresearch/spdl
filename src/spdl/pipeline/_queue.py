# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
from typing import TypeVar

__all__ = [
    "AsyncQueue",
]

T = TypeVar("T")


class AsyncQueue(asyncio.Queue[T]):
    pass
