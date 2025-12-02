# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections.abc import Iterable, Mapping
from typing import TypeVar

from spdl.pipeline import build_pipeline
from spdl.pipeline.defs import (
    Aggregate,
    Disaggregate,
    Pipe,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)

T = TypeVar("T")
U = TypeVar("U")


# pyre-strict


class PipelineDefTest(unittest.TestCase):
    def test_source_repr(self) -> None:
        """`repr` of SourceConfig should not generate a huge string."""

        src = SourceConfig(list(range(10000)))

        self.assertGreater(len(repr(src.source)), 10000)
        self.assertLess(len(repr(src)), 10000)

    def test_pipe_args_repr(self) -> None:
        """`repr` of Pipe should not generate a huge string."""

        lst = list(range(10000))
        self.assertGreater(len(repr(lst)), 10000)
        pipe = Pipe(lst)
        self.assertLess(len(repr(pipe._args.op)), 10000)
        self.assertLess(len(repr(pipe)), 10000)

        dct = {i: i for i in range(10000)}
        self.assertGreater(len(repr(dct)), 10000)
        pipe = Pipe(dct)
        self.assertLess(len(repr(pipe._args.op)), 10000)
        self.assertLess(len(repr(pipe)), 10000)

    def _test_build_pipeline(self, cfg: PipelineConfig[T], expected: list[T]) -> None:
        print(cfg)
        pipeline = build_pipeline(cfg, num_threads=1)

        with pipeline.auto_stop():
            ite = pipeline.get_iterator(timeout=3)
            self.assertEqual(list(ite), expected)

    def test_build_pipeline_simple(self) -> None:
        """PipelineConfig and build_pipeline works without pipes."""
        src = range(10)
        cfg = PipelineConfig(
            src=SourceConfig(src),
            pipes=[],
            sink=SinkConfig(3),
        )

        self._test_build_pipeline(cfg, list(src))

    def test_build_pipeline_aggregate(self) -> None:
        """Aggregate works"""
        cfg = PipelineConfig(
            src=SourceConfig(range(8)),
            pipes=[
                Aggregate(3, drop_last=False),
            ],
            sink=SinkConfig(3),
        )

        expected = [[0, 1, 2], [3, 4, 5], [6, 7]]
        self._test_build_pipeline(cfg, expected)

    def test_build_pipeline_aggregate_drop_last(self) -> None:
        """Aggregate works"""
        cfg = PipelineConfig(
            src=SourceConfig(range(8)),
            pipes=[
                Aggregate(3, drop_last=True),
            ],
            sink=SinkConfig(3),
        )

        expected = [[0, 1, 2], [3, 4, 5]]
        self._test_build_pipeline(cfg, expected)

    def test_build_pipeline_disaggregate(self) -> None:
        """Disaggregate works"""
        cfg = PipelineConfig(
            src=SourceConfig([[0, 1, 2, 3]]),
            pipes=[
                Disaggregate(),
            ],
            sink=SinkConfig(3),
        )

        expected = [0, 1, 2, 3]
        self._test_build_pipeline(cfg, expected)

    def test_build_pipeline_pipe_identity(self) -> None:
        """Pipe works with identity"""
        cfg = PipelineConfig(
            src=SourceConfig(range(5)),
            pipes=[
                Pipe(lambda x: x),
            ],
            sink=SinkConfig(3),
        )

        expected = list(range(5))
        self._test_build_pipeline(cfg, expected)

    def test_build_pipeline_pipe_double(self) -> None:
        """Pipe works with simple lambda"""
        cfg = PipelineConfig(
            src=SourceConfig(range(5)),
            pipes=[
                Pipe(lambda x: 2 * x),
            ],
            sink=SinkConfig(3),
        )

        expected = [2 * i for i in range(5)]
        self._test_build_pipeline(cfg, expected)

    def test_build_pipeline_pipe_sum(self) -> None:
        """Pipe works with aggregated data"""
        cfg = PipelineConfig(
            src=SourceConfig(range(8)),
            pipes=[
                Aggregate(3),
                Pipe(sum),
            ],
            sink=SinkConfig(3),
        )

        expected = [3, 12, 13]
        self._test_build_pipeline(cfg, expected)

    def test_build_pipeline_pipe_list(self) -> None:
        """Pipe works with list"""
        mapping = [i * i for i in range(8)]
        cfg = PipelineConfig(
            src=SourceConfig(range(8)),
            pipes=[Pipe(mapping)],
            sink=SinkConfig(3),
        )
        self._test_build_pipeline(cfg, mapping)

    def test_build_pipeline_pipe_map(self) -> None:
        """Pipe works with map (dict)"""
        mapping = {i: i * i for i in range(8)}
        cfg = PipelineConfig(
            src=SourceConfig(range(8)),
            pipes=[Pipe(mapping)],
            sink=SinkConfig(3),
        )

        self._test_build_pipeline(cfg, list(mapping.values()))
