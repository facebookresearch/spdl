# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import sys
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from spdl.pipeline import Pipeline, PipelineBuilder
from spdl.pipeline.defs import (
    _MainProcess,
    InterpreterPoolExecutorConfig,
    MAIN_PROCESS,
    Pipe,
    PlacementConfig,
    ProcessPoolExecutorConfig,
)


def add_one(x: int) -> int:
    return x + 1


def times_two(x: int) -> int:
    return x * 2


async def aadd_one(x: int) -> int:
    return x + 1


def _run(pipeline: Pipeline[Any], timeout: float = 60.0) -> list[Any]:
    with pipeline.auto_stop():
        return list(pipeline.get_iterator(timeout=timeout))


class ToMethodTest(unittest.TestCase):
    """The PipelineBuilder.to() region-marker method."""

    def test_appends_executor_config(self) -> None:
        """to() appends an PlacementConfig marker carrying the given target."""
        config = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one)
            .to(MAIN_PROCESS)
            .add_sink()
            .get_config()
        )
        markers = [p for p in config.pipes if isinstance(p, PlacementConfig)]
        self.assertEqual(len(markers), 2)
        self.assertEqual(markers[0].target, ProcessPoolExecutorConfig(max_workers=2))
        self.assertIs(markers[1].target, MAIN_PROCESS)

    def test_returns_self(self) -> None:
        """to() returns the builder for chaining."""
        b = PipelineBuilder().add_source(range(4))
        self.assertIs(b.to(MAIN_PROCESS), b)

    def test_rejects_live_executor(self) -> None:
        """Passing a live Executor is a TypeError pointing at pipe(executor=...)."""
        b = PipelineBuilder().add_source(range(4))
        with ThreadPoolExecutor(max_workers=1) as ex:
            with self.assertRaisesRegex(TypeError, "not a live Executor"):
                b.to(ex)  # type: ignore[arg-type]

    def test_rejects_wrong_type(self) -> None:
        """Passing a non-spec, non-sentinel target is a TypeError."""
        b = PipelineBuilder().add_source(range(4))
        with self.assertRaisesRegex(
            TypeError, "target must be a ProcessPoolExecutorConfig"
        ):
            b.to("subprocess")  # type: ignore[arg-type]


class ToBufferSizeTest(unittest.TestCase):
    """``to(..., buffer_size=N)``, which sets the region's transfer granularity."""

    def test_defaults_to_one(self) -> None:
        """Omitting buffer_size means one item per transfer."""
        config = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig(max_workers=1))
            .pipe(add_one)
            .to(MAIN_PROCESS)
            .add_sink()
            .get_config()
        )
        markers = [p for p in config.pipes if isinstance(p, PlacementConfig)]
        self.assertEqual([m.buffer_size for m in markers], [1, 1])

    def test_recorded_on_marker(self) -> None:
        """buffer_size is carried on the PlacementConfig that opens the region."""
        config = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig(max_workers=1), buffer_size=8)
            .pipe(add_one)
            .to(MAIN_PROCESS)
            .add_sink()
            .get_config()
        )
        markers = [p for p in config.pipes if isinstance(p, PlacementConfig)]
        self.assertEqual([m.buffer_size for m in markers], [8, 1])

    def test_per_region(self) -> None:
        """Adjacent regions each keep their own buffer_size."""
        config = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig(max_workers=1), buffer_size=4)
            .pipe(add_one)
            .to(ProcessPoolExecutorConfig(max_workers=1), buffer_size=16)
            .pipe(times_two)
            .to(MAIN_PROCESS)
            .add_sink()
            .get_config()
        )
        markers = [p for p in config.pipes if isinstance(p, PlacementConfig)]
        self.assertEqual([m.buffer_size for m in markers], [4, 16, 1])

    def test_not_carried_on_the_pool_spec(self) -> None:
        """The knob belongs to the boundary, not to the worker pool behind it."""
        self.assertFalse(hasattr(ProcessPoolExecutorConfig(), "buffer_size"))
        self.assertFalse(hasattr(InterpreterPoolExecutorConfig(), "buffer_size"))

    def test_accepted_on_main_process(self) -> None:
        """Closing a region is a boundary too -- results cross back there, so it takes a size.

        Each marker sizes the handoff at its own position, so a region's entry and exit are set
        independently.
        """
        config = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig(max_workers=1), buffer_size=32)
            .pipe(add_one)
            .to(MAIN_PROCESS, buffer_size=2)
            .add_sink()
            .get_config()
        )
        markers = [p for p in config.pipes if isinstance(p, PlacementConfig)]
        self.assertEqual([m.buffer_size for m in markers], [32, 2])

    def test_rejects_non_positive(self) -> None:
        """A buffer_size below 1 cannot form a transfer and is rejected at the call site."""
        b = PipelineBuilder().add_source(range(4))
        for bad in (0, -1):
            with self.subTest(buffer_size=bad):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    b.to(ProcessPoolExecutorConfig(max_workers=1), buffer_size=bad)

    def test_validated_at_build_for_hand_built_config(self) -> None:
        """A PlacementConfig assembled without the builder is still validated."""
        b = PipelineBuilder().add_source(range(4))
        # Bypass `to()` so only the build-time scan can catch it.
        b._process_args.append(
            PlacementConfig(
                target=ProcessPoolExecutorConfig(max_workers=1), buffer_size=0
            )
        )
        b._process_args.append(Pipe(add_one))
        b.to(MAIN_PROCESS).add_sink()
        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
            b.get_config()


class ToValidationTest(unittest.TestCase):
    """Validation performed on to() regions at get_config()/build()."""

    def test_unclosed_region_before_sink_raises(self) -> None:
        """A region left open before the sink is rejected."""
        b = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig())
            .pipe(add_one)
            .add_sink()
        )
        with self.assertRaisesRegex(ValueError, "must be closed with"):
            b.get_config()

    def test_input_order_inside_region_raises(self) -> None:
        """A stage with output_order='input' inside a region is rejected."""
        b = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig())
            .pipe(add_one, output_order="input")
            .to(MAIN_PROCESS)
            .add_sink()
        )
        with self.assertRaisesRegex(ValueError, "output_order='input'"):
            b.get_config()

    def test_output_order_input_in_path_variants_region_raises(self) -> None:
        """output_order='input' nested in a path_variants branch in a region is rejected."""
        b = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig())
            .path_variants(lambda _x: 0, [[Pipe(add_one, output_order="input")]])
            .to(MAIN_PROCESS)
            .add_sink()
        )
        with self.assertRaisesRegex(ValueError, "output_order='input'"):
            b.get_config()

    def test_empty_region_raises(self) -> None:
        """Opening a region with no stages before the next marker is rejected."""
        b = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig())
            .to(MAIN_PROCESS)
            .add_sink()
        )
        with self.assertRaisesRegex(ValueError, "has no stages"):
            b.get_config()

    def test_adjacent_nonempty_regions_are_valid(self) -> None:
        """Back-to-back non-empty regions (different targets, no MAIN between) are allowed."""
        config = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig(max_workers=1))
            .pipe(add_one)
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(times_two)
            .to(MAIN_PROCESS)
            .add_sink()
            .get_config()  # must not raise: both regions have stages
        )
        markers = [p for p in config.pipes if isinstance(p, PlacementConfig)]
        self.assertEqual(len(markers), 3)

    def test_closed_region_is_valid(self) -> None:
        """A properly closed region passes validation and preserves the marker sequence."""
        config = (
            PipelineBuilder()
            .add_source(range(4))
            .to(ProcessPoolExecutorConfig())
            .pipe(add_one)
            .to(MAIN_PROCESS)
            .add_sink()
            .get_config()  # must not raise: the region is closed before the sink
        )
        markers = [p for p in config.pipes if isinstance(p, PlacementConfig)]
        self.assertEqual(
            [type(m.target) for m in markers], [ProcessPoolExecutorConfig, _MainProcess]
        )

    if sys.version_info < (3, 14):

        def test_subinterpreter_region_rejected_before_314(self) -> None:
            """On Python < 3.14, a subinterpreter region is rejected at get_config()."""
            b = (
                PipelineBuilder()
                .add_source(range(4))
                .to(InterpreterPoolExecutorConfig())
                .pipe(add_one)
                .to(MAIN_PROCESS)
                .add_sink()
            )
            with self.assertRaisesRegex(RuntimeError, "requires Python 3.14"):
                b.get_config()


class ToEndToEndTest(unittest.TestCase):
    """The full public path: build and run a pipeline with a subprocess region."""

    def test_subprocess_region_runs(self) -> None:
        """A .to(ProcessPoolExecutorConfig()) region fuses and runs, matching the inline result."""
        n = 16
        pipeline = (
            PipelineBuilder()
            .add_source(range(n))
            .to(ProcessPoolExecutorConfig(max_workers=2))
            .pipe(add_one)
            .pipe(times_two)
            .to(MAIN_PROCESS)
            .add_sink()
            .build(num_threads=4)
        )
        self.assertEqual(sorted(_run(pipeline)), sorted((x + 1) * 2 for x in range(n)))


class AsyncOpExecutorRejectedTest(unittest.TestCase):
    """An async op may not be given an executor.

    Regions place stages by marker, so an async op inside a region carries no executor; the
    per-stage ``executor=`` on an async op (previously accepted as a fusion-group tag) is
    rejected again.
    """

    def test_pipe_async_with_executor_raises(self) -> None:
        """pipe(async_op, executor=...) is rejected regardless of executor type.

        The validation only checks that an executor was supplied (no type-specific branch), so an
        isolating (process) and a non-isolating (thread) pool are rejected identically.
        """
        for factory in (ProcessPoolExecutor, ThreadPoolExecutor):
            with self.subTest(executor=factory.__name__):
                b = PipelineBuilder().add_source(range(4))
                with factory(max_workers=1) as ex:
                    with self.assertRaises(ValueError):
                        b.pipe(aadd_one, executor=ex)

    def test_pipe_config_async_with_executor_raises(self) -> None:
        """The rejection is enforced at the config layer (Pipe factory)."""
        with ProcessPoolExecutor(max_workers=1) as ex:
            with self.assertRaises(ValueError):
                Pipe(aadd_one, executor=ex)
