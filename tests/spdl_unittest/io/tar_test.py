# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import tarfile
import unittest

import spdl.io

# pyre-strict


def _create_test_tar(files: list[tuple[str, bytes]]) -> bytes:
    """Helper method to create a TAR archive in memory."""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for filename, content in files:
            info = tarfile.TarInfo(name=filename)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    tar_buffer.seek(0)
    return tar_buffer.getvalue()


class TestIterTarfile(unittest.TestCase):
    """Test class for iter_tarfile functionality."""

    def _test(self, ref_data: list[tuple[str, bytes]], tar_data: bytes) -> None:
        # In-memory parsing
        outputs = list(spdl.io.iter_tarfile(tar_data))
        self.assertEqual(len(outputs), len(ref_data))
        for (ref_path, ref_content), (path, content_view) in zip(
            ref_data, outputs, strict=True
        ):
            # content_view is now a memoryview, convert to bytes for comparison
            content = bytes(content_view)
            self.assertEqual(path, ref_path)
            self.assertEqual(content, ref_content)

        # file-like object
        outputs = list(spdl.io.iter_tarfile(io.BytesIO(tar_data)))  # pyre-ignore
        self.assertEqual(len(outputs), len(ref_data))
        for (ref_path, ref_content), (path, content) in zip(
            ref_data, outputs, strict=True
        ):
            self.assertEqual(path, ref_path)
            self.assertEqual(content, ref_content)

    def test_iter_tar_in_memory_bytes(self) -> None:
        """Test iter_tar with in-memory bytes input."""
        ref_data = [
            ("file1.txt", b"Hello, World!"),
            ("file2.txt", b"This is a test file."),
            ("subdir/file3.txt", b"File in subdirectory."),  # path with subdirectory
            ("test.txt", b"Content from BytesIO"),
            ("data.bin", bytes(range(256))),  # All possible byte values
            ("empty.bin", b""),
        ]
        tar_data = _create_test_tar(ref_data)

        self._test(ref_data, tar_data)

    def test_iter_tar_empty_archive(self) -> None:
        """Test iter_tar with an empty TAR archive."""
        ref_data = []

        tar_data = _create_test_tar(ref_data)
        self._test(ref_data, tar_data)

    def test_iter_tar_single_file(self) -> None:
        """Test iter_tar with a single file."""
        ref_data = [("single.txt", b"Single file content")]
        tar_data = _create_test_tar(ref_data)

        self._test(ref_data, tar_data)
