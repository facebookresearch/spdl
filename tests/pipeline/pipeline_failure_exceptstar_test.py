# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for PipelineFailure using ``except*`` syntax (Python 3.11+ only).

This file is separated from pipeline_builder_test.py because ``except*`` is a
syntactic construct that causes SyntaxError on Python < 3.11 at import time,
which would prevent the entire module from loading.
"""

import sys
import unittest
from collections.abc import Iterator

if sys.version_info < (3, 11):
    raise unittest.SkipTest("except* syntax requires Python 3.11+")

from spdl.pipeline import PipelineBuilder


def passthrough(x: int) -> int:
    return x


class TestPipelineFailureExceptStar(unittest.TestCase):
    def test_pipeline_failure_except_star(self) -> None:
        def failing_range(n: int) -> Iterator[int]:
            yield from range(n)
            raise ValueError("Iterator failed")

        pipeline = (
            PipelineBuilder()
            .add_source(failing_range(3))
            .pipe(passthrough)
            .add_sink(1000)
            .build(num_threads=1)
        )

        caught = []
        try:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=30))
        except* ValueError as eg:
            caught.extend(eg.exceptions)

        self.assertGreaterEqual(len(caught), 1)
        self.assertIsInstance(caught[0], ValueError)
