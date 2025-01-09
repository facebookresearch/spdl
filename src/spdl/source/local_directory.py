# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Iterable for traversing local file directory."""

__all__ = ["LocalDirectory"]

import random
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

from ._type import IterableWithShuffle

# pyre-strict

################################################################################
# LocalDirectory
################################################################################


def _traverse(root: Path, pattern: str) -> list[Path]:
    flist = list(root.glob(pattern))
    flist.sort()
    return flist


class LocalDirectory(IterableWithShuffle[Path]):
    def __init__(self, root: str | PathLike[str], pattern: str) -> None:
        """

        .. note::

           See `<https://docs.python.org/3/library/pathlib.html#pattern-language>`_ for the
           list of supported patterns.

        """
        self.root = Path(root)
        self.pattern = pattern
        self._flist: list[Path] = _traverse(self.root, self.pattern)

    def shuffle(self, seed: int) -> None:
        random.Random(seed).shuffle(self._flist)

    def __iter__(self) -> Iterator[Path]:
        yield from self._flist
