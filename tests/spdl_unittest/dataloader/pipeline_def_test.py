# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def test_source_repr():
    """`repr` of SourceConfig should not generate a huge string."""

    src = SourceConfig(list(range(10000)))

    assert len(repr(src.source)) > 10000
    assert len(repr(src)) < 10000


def test_pipe_args_repr():
    """`repr` of Pipe should not generate a huge string."""

    lst = list(range(10000))
    assert len(repr(lst)) > 10000
    pipe = Pipe(lst)
    assert len(repr(pipe._args.op)) < 10000
    assert len(repr(pipe)) < 10000

    dct = {i: i for i in range(10000)}
    assert len(repr(dct)) > 10000
    pipe = Pipe(dct)
    assert len(repr(pipe._args.op)) < 10000
    assert len(repr(pipe)) < 10000


def _test_build_pipeline(cfg: PipelineConfig, expected: list) -> None:
    print(cfg)
    pipeline = build_pipeline(cfg, num_threads=1)

    with pipeline.auto_stop():
        ite = pipeline.get_iterator(timeout=3)
        assert list(ite) == expected


def test_build_pipeline_simple():
    """PipelineConfig and build_pipeline works without pipes."""
    src = range(10)
    cfg = PipelineConfig(
        src=SourceConfig(src),
        pipes=[],
        sink=SinkConfig(3),
    )

    _test_build_pipeline(cfg, list(src))


def test_build_pipeline_aggregate():
    """Aggregate works"""
    cfg = PipelineConfig(
        src=SourceConfig(range(8)),
        pipes=[
            Aggregate(3, drop_last=False),
        ],
        sink=SinkConfig(3),
    )

    expected = [[0, 1, 2], [3, 4, 5], [6, 7]]
    _test_build_pipeline(cfg, expected)


def test_build_pipeline_aggregate_drop_last():
    """Aggregate works"""
    cfg = PipelineConfig(
        src=SourceConfig(range(8)),
        pipes=[
            Aggregate(3, drop_last=True),
        ],
        sink=SinkConfig(3),
    )

    expected = [[0, 1, 2], [3, 4, 5]]
    _test_build_pipeline(cfg, expected)


def test_build_pipeline_disaggregate():
    """Disaggregate works"""
    cfg = PipelineConfig(
        src=SourceConfig([[0, 1, 2, 3]]),
        pipes=[
            Disaggregate(),
        ],
        sink=SinkConfig(3),
    )

    expected = [0, 1, 2, 3]
    _test_build_pipeline(cfg, expected)


def test_build_pipeline_pipe_identity():
    """Pipe works with identity"""
    cfg = PipelineConfig(
        src=SourceConfig(range(5)),
        pipes=[
            Pipe(lambda x: x),
        ],
        sink=SinkConfig(3),
    )

    expected = list(range(5))
    _test_build_pipeline(cfg, expected)


def test_build_pipeline_pipe_double():
    """Pipe works with simple lambda"""
    cfg = PipelineConfig(
        src=SourceConfig(range(5)),
        pipes=[
            Pipe(lambda x: 2 * x),
        ],
        sink=SinkConfig(3),
    )

    expected = [2 * i for i in range(5)]
    _test_build_pipeline(cfg, expected)


def test_build_pipeline_pipe_sum():
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
    _test_build_pipeline(cfg, expected)


def test_build_pipeline_pipe_list():
    """Pipe works with list"""
    cfg = PipelineConfig(
        src=SourceConfig(range(8)),
        pipes=[
            Pipe([i * i for i in range(8)]),
        ],
        sink=SinkConfig(3),
    )

    expected = [i * i for i in range(8)]
    _test_build_pipeline(cfg, expected)


def test_build_pipeline_pipe_map():
    """Pipe works with map (dict)"""
    cfg = PipelineConfig(
        src=SourceConfig(range(8)),
        pipes=[
            Pipe({i: i * i for i in range(8)}),
        ],
        sink=SinkConfig(3),
    )

    expected = [i * i for i in range(8)]
    _test_build_pipeline(cfg, expected)
