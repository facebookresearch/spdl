# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import overload, Protocol

# Importing `spdl.io.lib` instead of `spdl.io.lilb._archive`
# so as to delay the import of C++ extension module
from . import lib as _libspdl

# pyre-strict


class SupportsRead(Protocol):
    def read(self, n: int = -1) -> bytes: ...


@overload
def iter_tarfile(src: SupportsRead) -> Iterator[tuple[str, bytes]]: ...


@overload
def iter_tarfile(src: bytes) -> Iterator[tuple[str, int, int]]: ...


def iter_tarfile(
    src: bytes | SupportsRead,
) -> Iterator[tuple[str, int, int]] | Iterator[tuple[str, bytes]]:
    """**[Experimental]** Parse a TAR file and yields file path and its contents.

    Args:
        src: Source data, A ``bytes`` object containing all the data or
            a file-like object
            (only a read method i.e. ``read(n: int) -> bytes`` is required).

    Yields
        If the source is ``bytes``, then it yields a series of tuples
        consist of the file name, the offset, and the size.
        The content is not returned so that the caller can decide whether
        to load the individual data. (make a copy)

        If the source is a file-like object, then it yields a series of
        tuples consist of the file name and its contents (in ``bytes``).

    .. admonition:: Example - Parsing an in-memory TAR file.

       .. code-block::

          with open(path, "rb") as f:
              data = f.read()

          for filepath, offset, size in iter_tarfile(tar_data):
              contents = tar_data[offset : offset + size]
              print(f"File: {filepath}, ({size} bytes)")
              print(f"Preview: {contents[:30]}")

    .. admonition:: Example - Parsing a TAR file from a file-like object.

       .. code-block::

          with open(path, "rb") as f:
              for filepath, contents in iter_tarfile(f):
                  print(f"File: {filepath}, ({len(contents)} bytes)")
                  print(f"Preview: {contents[:30]}")
    """
    return _libspdl._archive.parse_tar(src)
