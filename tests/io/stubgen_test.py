# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

from spdl.io.lib.stubgen import (
    _normalize_existing_stubs,
    _normalize_stub_content,
    generate_stub_content,
)


class TestStubGeneration(unittest.TestCase):
    def test_normalizes_opaque_native_adaptor(self) -> None:
        """Opaque C++ adaptor names do not become invalid Python annotations."""
        source = (
            'def make_demuxer(_adaptor: "spdl::core::SourceAdaptor" | None)'
            " -> _spdl_ffmpeg7.Demuxer: ...\n"
        )
        result = _normalize_stub_content(source)
        self.assertEqual(
            result,
            "def make_demuxer(_adaptor: object | None) -> _libspdl.Demuxer: ...\n",
        )
        ast.parse(result)
        self.assertEqual(_normalize_stub_content(result), result)

    def test_tar_reader_protocol_is_not_generic_object(self) -> None:
        """TAR overloads distinguish memoryviews from binary file readers."""
        source = (
            "from __future__ import annotations\n"
            "from typing import overload\n"
            "@overload\n"
            "def parse_tar(arg: memoryview, /) -> InMemoryTarParser: ...\n"
            "@overload\n"
            "def parse_tar(arg: object, /) -> FileObjectTarParser: ...\n"
        )
        result = _normalize_stub_content(source)
        self.assertIn("arg: _BinaryReaderProtocol", result)
        self.assertIn("def read(self, size: int, /) -> bytes", result)
        self.assertTrue(result.startswith("from __future__ import annotations\n"))
        ast.parse(result)
        self.assertEqual(_normalize_stub_content(result), result)

    def test_existing_normalization_preserves_header(self) -> None:
        """Schema-only regeneration retains generated headers and is idempotent."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            header = "# Copyright\n# @generated\n\n\n"
            body = (
                'def make_demuxer(_adaptor: "spdl::core::SourceAdaptor" | None)'
                " -> object: ...\n"
            )
            for name in ("_libspdl", "_libspdl_cuda", "_archive", "_wav"):
                (inputs / f"{name}.pyi").write_text(header + body)
            _normalize_existing_stubs(inputs, outputs)
            expected = header + _normalize_stub_content(body)
            self.assertEqual((outputs / "_libspdl.pyi").read_text(), expected)
            self.assertEqual((inputs / "_libspdl.pyi").read_text(), header + body)
            _normalize_existing_stubs(outputs, outputs)
            self.assertEqual((outputs / "_libspdl.pyi").read_text(), expected)

    def test_unregistered_native_decoder_is_not_erased(self) -> None:
        """A missing native registration remains a visible type-schema error."""
        source = (
            "def make_decoder(codec: ImageCodec)"
            ' -> "spdl::core::Decoder<(spdl::core::MediaType)2>": ...\n'
        )
        self.assertEqual(_normalize_stub_content(source), source)

    @unittest.skipUnless(
        importlib.util.find_spec("nanobind") is not None,
        "Stub generation requires the optional nanobind developer tool.",
    )
    def test_retains_private_native_interfaces(self) -> None:
        """Stub generation retains private helpers used by Python wrappers."""
        module = ModuleType("stubgen_test_native")
        exec(
            "def _extract_packets_at_indices(packets: object) -> object: ...\n"
            "class Decoder:\n"
            "    def _reset(self) -> None: ...\n",
            module.__dict__,
        )
        result = generate_stub_content(module)
        self.assertIn("def _extract_packets_at_indices", result)
        self.assertIn("def _reset", result)
        ast.parse(result)
