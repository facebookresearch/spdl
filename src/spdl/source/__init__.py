# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Iterables for traversing datasets and utilities for transforming them."""

# pyre-strict

from ._type import IterableWithShuffle

__all__ = ["IterableWithShuffle"]


def __dir__() -> list[str]:
    return __all__
