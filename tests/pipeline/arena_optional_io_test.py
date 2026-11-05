# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check the Python-only pipeline boundary in fresh interpreters."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from spdl.pipeline._arena._ownership import _export_buffer
from spdl.pipeline._arena._pinning import _allocate_pinned


class ArenaOptionalIOTest(unittest.TestCase):
    def test_pipeline_and_pageable_ring_work_without_optional_modules(self) -> None:
        """Ordinary execution and ring copy-out never import IO, NumPy or Torch."""
        script = textwrap.dedent("""
            import importlib.abc, sys
            class RejectOptional(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'spdl.pipeline.fb':
                        raise ModuleNotFoundError(
                            'Meta logging is not part of the OSS fixture', name=fullname
                        )
                    if (
                        fullname == 'spdl.io' or fullname.startswith('spdl.io.')
                        or fullname.split('.')[0] in ('torch', 'numpy')
                    ):
                        raise ModuleNotFoundError('optional module blocked', name=fullname)
            sys.meta_path.insert(0, RejectOptional())
            from spdl.pipeline import (
                PipelineBuilder, SharedMemoryRingBuffer, SharedMemorySegmentPool
            )
            pipeline = PipelineBuilder().add_source([1, 2, 3]).add_sink(2).build(num_threads=1)
            with pipeline.auto_stop():
                assert list(pipeline.get_iterator(timeout=10)) == [1, 2, 3]
            ring = SharedMemoryRingBuffer(64, acquire_timeout=0)
            try:
                writer, reader = ring.open_writer(), ring.open_reader()
                writer.begin_unit()
                offset, size = writer.write_binary(b'abcd')
                span = writer.commit_unit()
                view = reader.read_binary(offset, size)
                assert bytes(view) == b'abcd'
                reader.end_unit(span, [])
                del view
            finally:
                ring.close(); ring.unlink()
            pool = SharedMemorySegmentPool(64, 1, acquire_timeout=0)
            try:
                writer, reader = pool.open_writer(), pool.open_reader()
                writer.begin_unit()
                offset, size = writer.write_binary(b'abcd')
                span = writer.commit_unit()
                try:
                    reader.read_binary(offset, size)
                except RuntimeError as error:
                    assert 'install spdl.io' in str(error)
                else:
                    raise AssertionError('native pool export silently succeeded')
                reader.end_unit(span, [])
                assert pool.reclaimed == 1
            finally:
                pool.close(); pool.unlink()
            assert not any(
                n == 'spdl.io' or n.startswith('spdl.io.')
                or n.split('.')[0] in ('torch', 'numpy') for n in sys.modules
            )
            print('PYTHON_ONLY_BOUNDARY_PASSED')
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PYTHON_ONLY_BOUNDARY_PASSED", result.stdout)

    def test_missing_helper_and_original_dependency_errors_remain_distinct(
        self,
    ) -> None:
        """Only missing IO/helper names are relabeled; ABI and NumPy errors survive."""
        for operation in (
            lambda: _export_buffer(memoryview(bytearray(b"safe")), lambda: None),
            lambda: _allocate_pinned(4),
        ):
            with self.subTest(operation=operation):
                missing = ModuleNotFoundError(
                    "no native helper", name="spdl.io.lib._buffer"
                )
                with patch("importlib.import_module", side_effect=missing):
                    with self.assertRaisesRegex(
                        RuntimeError, "install spdl.io"
                    ) as raised:
                        operation()
                    self.assertIs(raised.exception.__cause__, missing)
                for error in (
                    ModuleNotFoundError("no numpy", name="numpy"),
                    OSError("ABI loader failed"),
                    ImportError("undefined native symbol"),
                ):
                    with patch("importlib.import_module", side_effect=error):
                        with self.assertRaises(type(error)) as raised:
                            operation()
                        self.assertIs(raised.exception, error)
