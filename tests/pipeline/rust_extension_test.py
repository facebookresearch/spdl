# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect
import unittest
from unittest.mock import patch

from spdl.pipeline import lib


class RustExtensionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.interface = importlib.import_module("spdl.pipeline.lib._libspdl_pipeline")
        self.buffer = importlib.import_module("spdl.pipeline.lib._buffer")

    def test_native_buffer_export(self) -> None:
        """The importable Python interface delegates to the compiled buffer exporter."""
        native = self.buffer._load_rust()
        self.assertTrue(inspect.isbuiltin(native._export))
        view = self.interface._export(memoryview(bytearray(b"buffer")), lambda: None)
        self.assertTrue(native._is_exporter(view.obj))
        self.assertEqual(bytes(view), b"buffer")
        view.release()

    def test_import_is_lazy(self) -> None:
        """Obtaining a Python interface proxy does not load the native binary."""
        get_interface = lib.__getattr__
        with patch("spdl._internal.import_utils.importlib.import_module") as load:
            get_interface("_libspdl_pipeline")
        load.assert_not_called()

    def test_missing_extension_fails_on_use(self) -> None:
        """The Python module exists without Rust, but native use preserves the error."""
        missing = ModuleNotFoundError(
            "Native pipeline binding is absent", name="spdl.pipeline.lib._native"
        )
        with patch.object(self.buffer, "_load_rust", side_effect=missing):
            with self.assertRaises(RuntimeError) as caught:
                self.interface._export(memoryview(bytearray(b"buffer")), lambda: None)
        self.assertIs(caught.exception.__cause__, missing)

    def test_import_errors_are_not_masked(self) -> None:
        """A broken binary keeps its original error without another implementation."""
        failure = ImportError("Native binding dependency failed")
        with patch.object(self.buffer, "_load_rust", side_effect=failure):
            with self.assertRaises(ImportError) as caught:
                self.interface._export(memoryview(bytearray(b"buffer")), lambda: None)
        self.assertIs(caught.exception, failure)

    def test_foreign_buffer_probe_does_not_load_rust(self) -> None:
        """Ordinary Python buffers do not trigger a native ownership probe."""
        with patch.object(self.buffer, "_load_rust") as load:
            self.assertFalse(self.buffer._is_exporter(bytearray(b"ordinary")))
        load.assert_not_called()

    def test_missing_storage_binding_is_explicit(self) -> None:
        """Native storage operations require Rust rather than using a fake exporter."""
        missing = ModuleNotFoundError(
            "Native pipeline binding is absent", name="spdl.pipeline.lib._native"
        )
        with patch.object(self.buffer, "_load_rust", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "core Rust binding") as caught:
                self.buffer._export(memoryview(bytearray(b"storage")), lambda: None)
        self.assertIs(caught.exception.__cause__, missing)
