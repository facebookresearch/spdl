# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from spdl.pipeline._profile import (
    _build_pipeline_config,
    _fetch_inputs,
    profile_pipeline,
    ProfileHook,
    ProfileResult,
)
from spdl.pipeline.defs import (
    Aggregate,
    Disaggregate,
    MergeConfig,
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


def test_profile_pipeline():
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
            ret = _build_pipeline_config(inputs, pipe, concurrency)
            assert ret.pipes[0]._args.op is cfg.pipes[self.i]._args.op
            self.i += 1
            return ret

    mock = Intercept_()
    with patch("spdl.pipeline._profile._build_pipeline_config", mock):
        profile_pipeline(cfg)

    assert mock.i == 5


def test_profile_pipeline_callback():
    """Test that profile_pipeline calls the callback for each pipe stage."""

    def simple_op(i: int) -> int:
        return i + 1

    cfg = PipelineConfig(
        src=SourceConfig(range(10)),
        pipes=[
            Pipe(simple_op),
        ],
        sink=SinkConfig(1),
    )

    callback_mock = MagicMock()
    results = profile_pipeline(cfg, num_inputs=5, callback=callback_mock)

    callback_mock.assert_called_once()
    called_args = callback_mock.call_args[0]
    assert len(called_args) == 1
    called_result = called_args[0]

    assert isinstance(called_result, ProfileResult)
    assert called_result.name == "simple_op"
    assert len(called_result.stats) > 0

    assert len(results) == 1
    assert results[0].name == called_result.name
    assert len(results[0].stats) == len(called_result.stats)


def test_profile_pipeline_no_callback():
    """Test that profile_pipeline works correctly when no callback is provided."""

    def simple_op(i: int) -> int:
        return i * 2

    cfg = PipelineConfig(
        src=SourceConfig(range(5)),
        pipes=[
            Pipe(simple_op),
        ],
        sink=SinkConfig(1),
    )

    results = profile_pipeline(cfg, num_inputs=3, callback=None)

    assert len(results) == 1
    assert results[0].name == "simple_op"
    assert len(results[0].stats) > 0


class TestProfileHookTest(unittest.TestCase):
    """Test class for ProfileHook functionality."""

    def test_profile_pipeline_custom_hook_methods_called(self):
        """Test that when a custom ProfileHook is provided, its stage_profile_hook and pipeline_profile_hook methods are called."""

        def simple_op(i: int) -> int:
            return i + 10

        cfg = PipelineConfig(
            src=SourceConfig(range(5)),
            pipes=[
                Pipe(simple_op),
            ],
            sink=SinkConfig(1),
        )

        # Create a custom ProfileHook mock
        stage_hook_mock = MagicMock()
        pipeline_hook_mock = MagicMock()

        class MockProfileHook(ProfileHook):
            @contextmanager
            def stage_profile_hook(self) -> Iterator[None]:
                stage_hook_mock()
                try:
                    yield
                finally:
                    stage_hook_mock()

            @contextmanager
            def pipeline_profile_hook(self) -> Iterator[None]:
                pipeline_hook_mock()
                try:
                    yield
                finally:
                    pipeline_hook_mock()

        custom_hook = MockProfileHook()

        results = profile_pipeline(cfg, num_inputs=3, hook=custom_hook)

        self.assertTrue(len(results) > 0)
        self.assertEqual(pipeline_hook_mock.call_count, 2)
        # stage_profile_hook should be called twice per concurrency level (start and end)
        # Default concurrency levels are [32, 16, 8, 4, 1] = 5 levels, so 10 calls total
        self.assertEqual(stage_hook_mock.call_count, 10)

    def test_profile_pipeline_skips_when_local_rank_not_zero(self):
        """Test that profiling is skipped if LOCAL_RANK is not '0'."""

        def simple_op(i: int) -> int:
            return i * 3

        cfg = PipelineConfig(
            src=SourceConfig(range(5)),
            pipes=[
                Pipe(simple_op),
            ],
            sink=SinkConfig(1),
        )

        with patch("spdl.pipeline._profile._get_local_rank", return_value=1):
            results = profile_pipeline(cfg, num_inputs=5)

        self.assertEqual(results, [])

    def test_profile_pipeline_runs_when_local_rank_zero(self):
        """Test that profiling runs normally when LOCAL_RANK is '0'."""

        def simple_op(i: int) -> int:
            return i * 2

        cfg = PipelineConfig(
            src=SourceConfig(range(5)),
            pipes=[
                Pipe(simple_op),
            ],
            sink=SinkConfig(1),
        )

        with patch("spdl.pipeline._profile._get_local_rank", return_value=0):
            results = profile_pipeline(cfg, num_inputs=3)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].name, "simple_op")
        self.assertGreater(len(results[0].stats), 0)


def test_profile_pipeline_with_merge_config_and_post_merge_stages():
    """Test that profile_pipeline profiles all stages including those in MergeConfig and post-merge stages."""

    def double(i: int) -> int:
        return i * 2

    def triple(i: int) -> int:
        return i * 3

    def add_ten(i: int) -> int:
        return i + 10

    def square(i: int) -> int:
        return i * i

    plc1 = PipelineConfig(
        src=SourceConfig(range(5)),
        pipes=[
            Pipe(double, name="double"),
        ],
        sink=SinkConfig(1),
    )

    plc2 = PipelineConfig(
        src=SourceConfig(range(10, 15)),
        pipes=[
            Pipe(triple, name="triple"),
        ],
        sink=SinkConfig(1),
    )

    main_cfg = PipelineConfig(
        src=MergeConfig([plc1, plc2]),
        pipes=[
            Pipe(add_ten, name="add_ten"),
            Pipe(square, name="square"),
        ],
        sink=SinkConfig(1),
    )

    results = profile_pipeline(main_cfg, num_inputs=3)

    assert len(results) == 4

    assert results[0].name == "double"
    assert results[1].name == "triple"
    assert results[2].name == "add_ten"
    assert results[3].name == "square"

    for result in results:
        assert len(result.stats) > 0
        for stat in result.stats:
            assert hasattr(stat, "concurrency")
            assert hasattr(stat, "qps")
            assert hasattr(stat, "occupancy_rate")
