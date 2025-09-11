# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import AsyncIterator
from unittest.mock import patch

from spdl.pipeline._profile import _build_pipeline, _fetch_inputs, profile_pipeline
from spdl.pipeline.defs import (
    Aggregate,
    Disaggregate,
    Pipe,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)


def test_fetch_inputs():
    """_fetch_inputs collects input items"""
    src = SourceConfig(range(10))

    inputs = _fetch_inputs(src, num_items=3)
    assert inputs == list(range(3))


def test_fetch_inputs_async():
    """_fetch_inputs collects input items"""

    async def arange(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    src = SourceConfig(arange(10))

    inputs = _fetch_inputs(src, num_items=3)
    assert inputs == list(range(3))


def test_profile_pipes():
    """"""

    def foo(i: int) -> int:
        return 2 * i

    def bar(items: list[int]) -> list[int]:
        return [sum(items)]

    def bazz(i: int) -> int:
        return i * i

    N, m = 25, 3

    cfg = PipelineConfig(
        src=SourceConfig(range(N)),
        pipes=[
            Pipe(foo),
            Aggregate(m),
            Pipe(bar),
            Disaggregate(),
            Pipe(bazz),
        ],
        sink=SinkConfig(3),
    )

    class Intercept_:
        def __init__(self) -> None:
            self.i = 0

            # Expected inputs created through profiling
            src_ = list(range(N))
            foo_ = [foo(i) for i in src_]
            agg_ = [[foo(i) for i in range(i, min(N, i + m))] for i in src_[::m]]
            bar_ = [bar(i) for i in agg_]
            dis_ = [i[0] for i in bar_]
            bazz_ = [bazz(i) for i in dis_]

            self.inputs = [src_, foo_, agg_, bar_, dis_, bazz_]

        def __call__(self, inputs, pipe, concurrency):
            assert inputs == self.inputs[self.i]
            ret = _build_pipeline(inputs, pipe, concurrency)
            assert ret.pipes[0]._args.op is cfg.pipes[self.i]._args.op
            self.i += 1
            return ret

    mock = Intercept_()
    with patch("spdl.pipeline._profile._build_pipeline", mock):
        profile_pipeline(cfg)

    assert mock.i == 5
