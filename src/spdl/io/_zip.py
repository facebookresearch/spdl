# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections.abc import Mapping

import spdl.lib
from numpy.lib import format as np_format

__all__ = [
    "NpzFile",
    "load_npz",
]


class ZipFile:
    def __init__(self, handle):
        self._handle = handle

    def read(self, n: int) -> bytes:
        buffer = bytes(bytearray(n))
        num_read = self._handle.read(buffer)
        return buffer[:num_read]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getattr__(self, name: str):
        if hasattr(self._handle, name):
            return getattr(self._handle, name)
        raise AttributeError(f"{name} not found")


class ZipArchive:
    def __init__(self, handle, data: bytes):
        self._handle = handle
        self._data = data  # to keep the reference while the handle is alive

    def __getattr__(self, name: str):
        if hasattr(self._handle, name):
            return getattr(self._handle, name)
        raise AttributeError(f"{name} not found")

    def open(self, name: str):
        return ZipFile(self._handle.open(name))

    def read(self, name: str):
        with self.open(name) as file:
            return file.read()


def zip_archive(data: bytes) -> ZipArchive:
    handle = spdl.lib._zip.zip_archive(data)
    return ZipArchive(handle, data)


class NpzFile(Mapping):
    """NpzFile()
    A class mimics the behavior of :py:class:`numpy.lib.npyio.NpzFile`.

    It is a thin wrapper around a zip archive, and implements
    :py:class:`collections.abc.Mapping` interface.

    See :py:func:`load_npz` for the usage.
    """

    def __init__(self, archive):
        self._archive = archive
        self._files = self._archive.namelist()
        self.files = [f.removesuffix(".npy") for f in self._files]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._archive.close()

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        """Provide dictionary-like access to array data.

        One difference from the regular dictionary access is that
        it also supports accessing the item without ``.npy`` suffix
        in the key. This matches the behavior of :py:class:`numpy.lib.npyio.NpzFile`.
        """
        if key in self._files:
            pass
        elif key in self.files:
            key = f"{key}.npy"
        else:
            raise KeyError(f"{key} is not a file in the archive")

        with self._archive.open(key) as file:
            return np_format.read_array(file)

    def __contains__(self, key):
        return key in self._files or key in self.files

    def __repr__(self):
        return f"NpzFile object with {len(self)} entries."


def load_npz(data: bytes) -> NpzFile:
    """**[Experimental]** Load a numpy archive file (``npz``).

    It is almost a drop-in replacement for :py:func:`numpy.load` function,
    but it only supports the basic use cases.

    This function uses the C++ implementation of the zip archive reader, which
    releases the GIL. So it is more efficient than the official NumPy implementation
    for supported cases.

    Args:
        data: The data to load.

    Example

       >>> x = np.arange(10)
       >>> y = np.sin(x)
       >>>
       >>> with tempfile.TemporaryFile() as f:
       ...     np.savez(f, x=x, y=y)
       ...     f.seek(0)
       ...     data = spdl.io.load_npz(f.read())
       ...
       >>> assert np.array_equal(data["x"], x)
       >>> assert np.array_equal(data["y"], y)

    """
    return NpzFile(zip_archive(data))
