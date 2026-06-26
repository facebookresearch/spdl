# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for PathVariants feature."""

# pyre-strict

import asyncio
import unittest
from collections.abc import AsyncIterator, Iterable

from spdl.pipeline import build_pipeline
from spdl.pipeline._components._node import PipelineFailure
from spdl.pipeline.defs import (
    Aggregate,
    Disaggregate,
    Merge,
    PathVariants,
    Pipe,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)


def _run_pipeline(
    config: PipelineConfig[object], num_threads: int = 2, timeout: int = 30
) -> list[object]:
    """Helper to build, run, and collect results from a pipeline."""
    pipeline = build_pipeline(config, num_threads=num_threads)
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=timeout))


async def _slow_source(items: Iterable[int], delay: float = 0.1) -> AsyncIterator[int]:
    """Yield items with a delay between them.

    Each item is fully processed before the next one is dispatched,
    making cross-path ordering deterministic (matching input order).
    """
    for item in items:
        yield item
        await asyncio.sleep(delay)


class PathVariantsBasicTest(unittest.TestCase):
    """Basic functional tests for PathVariants."""

    def test_basic_routing(self) -> None:
        """Even items to path 0 (double), odd items to path 1 (add 100)."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(6))),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Pipe(lambda x: x * 2)],  # path 0: evens
                        [Pipe(lambda x: x + 100)],  # path 1: odds
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # Slow source ensures each item is processed before the next arrives,
        # so output order matches input order:
        # 0→path0→0, 1→path1→101, 2→path0→4, 3→path1→103, 4→path0→8, 5→path1→105
        self.assertEqual(results, [0, 101, 4, 103, 8, 105])

    def test_async_router(self) -> None:
        """Async router function works correctly."""

        async def async_router(x: int) -> int:
            return x % 2

        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(6))),
            pipes=[
                PathVariants(
                    router=async_router,
                    paths=[
                        [Pipe(lambda x: x * 2)],
                        [Pipe(lambda x: x + 100)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        self.assertEqual(results, [0, 101, 4, 103, 8, 105])

    def test_async_callable_router(self) -> None:
        """Class with async __call__ works as router."""

        class AsyncRouter:
            async def __call__(self, x: int) -> int:
                return x % 2

        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(6))),
            pipes=[
                PathVariants(
                    router=AsyncRouter(),
                    paths=[
                        [Pipe(lambda x: x * 2)],
                        [Pipe(lambda x: x + 100)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        self.assertEqual(results, [0, 101, 4, 103, 8, 105])

    def test_all_to_one_path(self) -> None:
        """Router always returns 0 — all items go to path 0."""
        config = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[
                PathVariants(
                    router=lambda x: 0,
                    paths=[
                        [Pipe(lambda x: x * 10)],
                        [Pipe(lambda x: x + 1000)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        self.assertEqual(results, [10, 20, 30])

    def test_multiple_paths_different_processing(self) -> None:
        """3 paths with different transforms."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(9))),
            pipes=[
                PathVariants(
                    router=lambda x: x % 3,
                    paths=[
                        [Pipe(lambda x: x * 2)],  # path 0: ×2
                        [Pipe(lambda x: x + 10)],  # path 1: +10
                        [Pipe(lambda x: -x)],  # path 2: negate
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=20),
        )
        results = _run_pipeline(config)
        # Slow source ensures interleaved input order:
        # 0→path0→0, 1→path1→11, 2→path2→-2,
        # 3→path0→6, 4→path1→14, 5→path2→-5,
        # 6→path0→12, 7→path1→17, 8→path2→-8
        self.assertEqual(results, [0, 11, -2, 6, 14, -5, 12, 17, -8])

    def test_identity_path_passthrough(self) -> None:
        """A path with an identity pipe passes items through unchanged."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source([1, 2, 3, 4])),
            pipes=[
                PathVariants(
                    router=lambda x: 0 if x <= 2 else 1,
                    paths=[
                        [Pipe(lambda x: x * 100)],  # path 0: transform
                        [Pipe(lambda x: x)],  # path 1: passthrough
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # 1→path0→100, 2→path0→200, 3→path1→3, 4→path1→4
        self.assertEqual(results, [100, 200, 3, 4])

    def test_paths_with_different_stage_counts(self) -> None:
        """Paths with different numbers of pipe stages."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(6))),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Pipe(lambda x: x * 10)],  # path 0: 1 stage
                        [
                            Pipe(lambda x: (x, x + 100)),
                            Pipe(lambda t: t[1]),
                        ],  # path 1: 2 stages
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # Slow source ensures input-order interleaving even with different
        # stage counts: 0→0, 1→(1,101)→101, 2→20, 3→(3,103)→103, 4→40, 5→105
        self.assertEqual(results, [0, 101, 20, 103, 40, 105])

    def test_path_with_aggregate(self) -> None:
        """Path containing Aggregate inside."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(6))),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Aggregate(3)],  # path 0: batch evens by 3
                        [Pipe(lambda x: x + 100)],  # path 1: add 100 to odds
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # Aggregate waits for 3 evens (0,2,4). With slow source, odds pass
        # through immediately while aggregate buffers:
        # 0→agg, 1→101, 2→agg, 3→103, 4→agg→[0,2,4], 5→105
        self.assertEqual(results, [101, 103, [0, 2, 4], 105])

    def test_path_with_aggregate_and_disaggregate(self) -> None:
        """Path containing Aggregate + Disaggregate inside."""
        config = PipelineConfig(
            src=SourceConfig(range(6)),
            pipes=[
                PathVariants(
                    router=lambda x: 0,
                    paths=[
                        [Aggregate(2), Disaggregate()],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        self.assertEqual(results, [0, 1, 2, 3, 4, 5])

    def test_nested_path_variants(self) -> None:
        """PathVariants inside a path of another PathVariants."""
        inner_variants = PathVariants(
            router=lambda x: 0 if x < 50 else 1,
            paths=[
                [Pipe(lambda x: x + 1000)],  # inner path 0
                [Pipe(lambda x: x + 2000)],  # inner path 1
            ],
        )
        config = PipelineConfig(
            src=SourceConfig(_slow_source([1, 2, 51, 52])),
            pipes=[
                PathVariants(
                    router=lambda x: 0,  # all to path 0
                    paths=[
                        [inner_variants],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # 1→inner path0→1001, 2→inner path0→1002,
        # 51→inner path1→2051, 52→inner path1→2052
        self.assertEqual(results, [1001, 1002, 2051, 2052])

    def test_immediately_nested_path_variants(self) -> None:
        """PathVariants as the first and only config in each path of an outer
        PathVariants — the inner router reads directly from the outer router's
        per-path queue."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(12))),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,  # outer: evens vs odds
                    paths=[
                        # path 0 (evens): immediately nest another PathVariants
                        [
                            PathVariants(
                                router=lambda x: 0 if x < 6 else 1,
                                paths=[
                                    [Pipe(lambda x: x * 10)],  # small evens
                                    [Pipe(lambda x: x * 100)],  # large evens
                                ],
                            ),
                        ],
                        # path 1 (odds): immediately nest another PathVariants
                        [
                            PathVariants(
                                router=lambda x: 0 if x < 6 else 1,
                                paths=[
                                    [Pipe(lambda x: -x)],  # small odds
                                    [Pipe(lambda x: -(x * 10))],  # large odds
                                ],
                            ),
                        ],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # Items interleaved across outer paths, then inner paths:
        # 0→even→small→0, 1→odd→small→-1, 2→even→small→20, 3→odd→small→-3,
        # 4→even→small→40, 5→odd→small→-5, 6→even→large→600, 7→odd→large→-70,
        # 8→even→large→800, 9→odd→large→-90, 10→even→large→1000, 11→odd→large→-110
        self.assertEqual(
            results,
            [0, -1, 20, -3, 40, -5, 600, -70, 800, -90, 1000, -110],
        )

    def test_path_variants_before_and_after_pipes(self) -> None:
        """Pipes before and after PathVariants in the main pipeline."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source([1, 2, 3, 4])),
            pipes=[
                Pipe(lambda x: x * 10),  # pre-processing
                PathVariants(
                    router=lambda x: 0 if x < 25 else 1,
                    paths=[
                        [Pipe(lambda x: x + 1)],
                        [Pipe(lambda x: x + 2)],
                    ],
                ),
                Pipe(lambda x: x * -1),  # post-processing
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # pre: [10,20,30,40]
        # path0 (<25): 10->11, 20->21
        # path1 (>=25): 30->32, 40->42
        # post: [-11,-21,-32,-42]
        self.assertEqual(results, [-11, -21, -32, -42])

    def test_path_variants_with_merge_source(self) -> None:
        """Merge as source, then PathVariants in pipes."""
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        plc2 = PipelineConfig(
            src=SourceConfig([3, 4]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Pipe(lambda x: x * 100)],
                        [Pipe(lambda x: x * -1)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # evens: 2,4 -> 200,400
        # odds: 1,3 -> -1,-3
        self.assertCountEqual(results, [200, 400, -1, -3])

    def test_path_variants_with_async_pipe(self) -> None:
        """Async pipe inside a path."""

        async def async_double(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        config = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[
                PathVariants(
                    router=lambda x: 0,
                    paths=[
                        [Pipe(async_double)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        self.assertEqual(results, [2, 4, 6])

    def test_path_variants_with_concurrent_pipe(self) -> None:
        """Pipe with concurrency > 1 inside a path."""
        config = PipelineConfig(
            src=SourceConfig(_slow_source(range(10))),
            pipes=[
                PathVariants(
                    router=lambda x: 0,
                    paths=[
                        [Pipe(lambda x: x + 1, concurrency=3)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=20),
        )
        results = _run_pipeline(config)
        self.assertEqual(results, list(range(1, 11)))


class PathVariantsValidationTest(unittest.TestCase):
    """Validation tests for PathVariants config."""

    def test_validation_empty_paths(self) -> None:
        """PathVariants with no paths raises ValueError."""
        with self.assertRaises(ValueError):
            PathVariants(router=lambda x: 0, paths=[])

    def test_validation_empty_path(self) -> None:
        """PathVariants with an empty path raises ValueError."""
        with self.assertRaises(ValueError):
            PathVariants(
                router=lambda x: 0,
                paths=[[Pipe(lambda x: x)], []],
            )

    def test_validation_non_callable_router(self) -> None:
        """Non-callable router raises ValueError."""
        with self.assertRaises(ValueError):
            # pyre-ignore[6]: Intentionally passing non-callable to test validation.
            PathVariants(router=42, paths=[[Pipe(lambda x: x)]])

    def test_validation_source_in_path(self) -> None:
        """SourceConfig in a path raises ValueError at construction time."""
        with self.assertRaises(ValueError):
            PathVariants(
                router=lambda x: 0,
                # pyre-ignore[6]: Intentionally passing SourceConfig.
                paths=[[SourceConfig([3, 4])]],
            )

    def test_validation_sink_in_path(self) -> None:
        """SinkConfig in a path raises ValueError at construction time."""
        with self.assertRaises(ValueError):
            PathVariants(
                router=lambda x: 0,
                # pyre-ignore[6]: Intentionally passing SinkConfig.
                paths=[[SinkConfig(buffer_size=10)]],
            )

    def test_validation_source_in_second_path(self) -> None:
        """SourceConfig in a later path position raises ValueError."""
        with self.assertRaises(ValueError):
            PathVariants(
                router=lambda x: 0,
                # pyre-ignore[6]: Intentionally passing SourceConfig.
                paths=[
                    [Pipe(lambda x: x)],
                    [Pipe(lambda x: x), SourceConfig([1, 2])],
                ],
            )

    def test_validation_sink_in_middle_of_path(self) -> None:
        """SinkConfig in the middle of a path raises ValueError."""
        with self.assertRaises(ValueError):
            PathVariants(
                router=lambda x: 0,
                # pyre-ignore[6]: Intentionally passing SinkConfig.
                paths=[
                    [Pipe(lambda x: x), SinkConfig(buffer_size=10), Pipe(lambda x: x)],
                ],
            )


class PathVariantsErrorHandlingTest(unittest.TestCase):
    """Error handling and edge case tests for PathVariants."""

    def test_router_returns_negative_index(self) -> None:
        """Router returns -1 — pipeline fails with PipelineFailure."""
        config = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[
                PathVariants(
                    router=lambda x: -1,
                    paths=[
                        [Pipe(lambda x: x)],
                        [Pipe(lambda x: x)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)

    def test_router_returns_index_equal_to_num_paths(self) -> None:
        """Router returns N (== len(paths)) — pipeline fails."""
        config = PipelineConfig(
            src=SourceConfig([1]),
            pipes=[
                PathVariants(
                    router=lambda x: 2,  # only 2 paths, index 2 is out of range
                    paths=[
                        [Pipe(lambda x: x)],
                        [Pipe(lambda x: x)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)

    def test_router_returns_index_greater_than_num_paths(self) -> None:
        """Router returns N+5 — pipeline fails."""
        config = PipelineConfig(
            src=SourceConfig([1]),
            pipes=[
                PathVariants(
                    router=lambda x: 7,
                    paths=[
                        [Pipe(lambda x: x)],
                        [Pipe(lambda x: x)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)

    def test_one_path_fails_other_continues(self) -> None:
        """One path's pipe raises; the other path's items still processed.

        With default max_failures=-1, individual task failures are tolerated.
        The pipeline completes successfully and the non-failing path's results
        are collected. The failing path's items are dropped.
        """

        def fail_on_odd(x: int) -> int:
            if x % 2 == 1:
                raise ValueError(f"odd item {x}")
            return x * 10

        config = PipelineConfig(
            src=SourceConfig(range(6)),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Pipe(lambda x: x * 10)],  # path 0: evens succeed
                        [Pipe(fail_on_odd)],  # path 1: odds fail (dropped)
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        # Path 0 items succeed: 0,2,4 -> 0,20,40
        # Path 1 items fail and are dropped
        self.assertEqual(results, [0, 20, 40])

    def test_one_path_fails_with_max_failures(self) -> None:
        """One path's pipe raises with max_failures=0; pipeline fails."""

        def fail_always(x: int) -> int:
            raise ValueError("fail")

        config = PipelineConfig(
            src=SourceConfig(range(4)),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Pipe(lambda x: x * 10)],
                        [Pipe(fail_always, max_failures=0)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)

    def test_all_paths_fail(self) -> None:
        """All paths fail with max_failures=0 — pipeline raises PipelineFailure."""

        def always_fail(x: int) -> int:
            raise ValueError("boom")

        config = PipelineConfig(
            src=SourceConfig([1, 2]),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Pipe(always_fail, max_failures=0)],
                        [Pipe(always_fail, max_failures=0)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)

    def test_empty_source_with_path_variants(self) -> None:
        """Empty source — clean shutdown with no results."""
        config = PipelineConfig(
            src=SourceConfig([]),
            pipes=[
                PathVariants(
                    router=lambda x: 0,
                    paths=[
                        [Pipe(lambda x: x * 2)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        results = _run_pipeline(config)
        self.assertEqual(results, [])

    def test_path_failure_cancels_router(self) -> None:
        """When a path fails with max_failures=0, the router is cancelled and
        the pipeline shuts down cleanly without hanging."""

        def fail_immediately(x: int) -> int:
            raise ValueError("boom")

        config = PipelineConfig(
            # Use enough items so the router is still active when path fails.
            src=SourceConfig(range(100)),
            pipes=[
                PathVariants(
                    router=lambda x: x % 2,
                    paths=[
                        [Pipe(lambda x: x, max_failures=0)],  # path 0: succeeds
                        [Pipe(fail_immediately, max_failures=0)],  # path 1: fails
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        # The pipeline must raise PipelineFailure (not hang).
        # If the router is not cancelled on path failure, this would deadlock
        # because the router keeps trying to push items to the dead path's queue.
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)

    def test_path_failure_cancels_router_all_to_failing_path(self) -> None:
        """All items routed to the failing path — router cancelled, no hang."""

        def fail_immediately(x: int) -> int:
            raise ValueError("boom")

        config = PipelineConfig(
            src=SourceConfig(range(100)),
            pipes=[
                PathVariants(
                    router=lambda x: 1,  # all items to failing path
                    paths=[
                        [Pipe(lambda x: x)],
                        [Pipe(fail_immediately, max_failures=0)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)

    def test_router_raises_exception(self) -> None:
        """Router function itself raises — pipeline fails cleanly."""

        def bad_router(x: int) -> int:
            raise RuntimeError("router error")

        config = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[
                PathVariants(
                    router=bad_router,
                    paths=[
                        [Pipe(lambda x: x)],
                    ],
                ),
            ],
            sink=SinkConfig(buffer_size=10),
        )
        with self.assertRaises(PipelineFailure):
            _run_pipeline(config)


class PathVariantsReprTest(unittest.TestCase):
    """Repr tests for PathVariants."""

    def test_repr(self) -> None:
        """Verify repr is readable and includes path info."""
        cfg = PathVariants(
            router=lambda x: x % 2,
            paths=[
                [Pipe(lambda x: x * 2, name="double")],
                [Pipe(lambda x: x + 1, name="add_one")],
            ],
        )
        r = repr(cfg)
        self.assertIn("PathVariants", r)
        self.assertIn("path0", r)
        self.assertIn("path1", r)


if __name__ == "__main__":
    unittest.main()
