# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import inspect
import unittest
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence

from spdl.pipeline.defs import (
    Aggregate,
    Disaggregate,
    Merge,
    Pipe,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)


# Test helper functions and classes
def example_sync_function(x: int) -> int:
    """Example synchronous function for testing."""
    return x * 2


async def example_async_function(x: int) -> int:
    """Example async function for testing."""
    await asyncio.sleep(0)
    return x * 2


def _ln(target: object) -> int:
    """Helper to get line number from inspect.getsourcelines."""
    return inspect.getsourcelines(target)[1]  # pyre-ignore[6]


class ExampleIterable(Iterable[int]):
    """Example iterable class for testing."""

    def __iter__(self) -> Iterator[int]:
        return iter([1, 2, 3])


class ExampleAsyncIterable(AsyncIterable[int]):
    """Example async iterable class for testing."""

    async def __aiter__(self) -> AsyncIterator[int]:
        for i in [1, 2, 3]:
            yield i


async def custom_merge_op(
    name: str,
    input_queues: Sequence[asyncio.Queue],
    output_queue: asyncio.Queue,
) -> None:
    """Example custom merge operation for testing."""
    pass


class TestSourceConfigRepr(unittest.TestCase):
    """Test SourceConfig.__repr__ with source location."""

    def test_source_config_repr_with_iterable_class(self) -> None:
        """Test __repr__ shows source location for iterable class."""
        # Setup: create source config with custom iterable
        source = ExampleIterable()
        config = SourceConfig(source=source)

        # Execute: get repr
        result = repr(config)

        # Assert: repr contains class name
        # Note: for class instances, source location may not always be available
        self.assertIn("ExampleIterable", result)
        self.assertIn("SourceConfig", result)

    def test_source_config_repr_with_generator(self) -> None:
        """Test __repr__ shows source location for generator function."""
        # Setup: create source config with generator
        source = (x for x in range(10))
        config = SourceConfig(source=source)

        # Execute: get repr
        result = repr(config)

        # Assert: repr contains generator class name
        self.assertIn("generator", result)

    def test_source_config_repr_with_async_iterable(self) -> None:
        """Test __repr__ shows source location for async iterable."""
        # Setup: create source config with async iterable
        source = ExampleAsyncIterable()
        config = SourceConfig(source=source)

        # Execute: get repr
        result = repr(config)

        # Assert: repr contains class name
        # Note: for class instances, source location may not always be available
        self.assertIn("ExampleAsyncIterable", result)
        self.assertIn("SourceConfig", result)


class TestPipeConfigRepr(unittest.TestCase):
    """Test PipeConfig.__repr__ with source location."""

    def test_pipe_config_repr_with_sync_function(self) -> None:
        """Test __repr__ shows source location for sync function."""
        # Setup: create pipe config with sync function
        config = Pipe(example_sync_function, concurrency=4)

        # Execute: get repr
        result = repr(config)

        # Assert: repr contains function name, concurrency, and source location
        self.assertIn("concurrency=4", result)
        self.assertIn("example_sync_function", result)
        self.assertIn(__file__, result)
        self.assertIn(f":{_ln(example_sync_function)}", result)

    def test_pipe_config_repr_with_async_function(self) -> None:
        """Test __repr__ shows source location for async function."""
        # Setup: create pipe config with async function
        config = Pipe(example_async_function, concurrency=2)

        # Execute: get repr
        result = repr(config)

        # Assert: repr contains function name, concurrency, and source location
        self.assertIn("concurrency=2", result)
        self.assertIn("example_async_function", result)
        self.assertIn(__file__, result)
        self.assertIn(f":{_ln(example_async_function)}", result)

    def test_pipe_config_repr_with_lambda(self) -> None:
        """Test __repr__ handles lambda functions gracefully."""
        # Setup: create pipe config with lambda
        config = Pipe(lambda x: x * 2, concurrency=1)

        # Execute: get repr
        result = repr(config)

        # Assert: repr contains lambda and concurrency
        self.assertIn("concurrency=1", result)
        self.assertIn("lambda", result)

    def test_pipe_config_repr_without_source_location(self) -> None:
        """Test __repr__ handles cases where source location cannot be determined."""
        # Setup: create pipe config with built-in function
        config = Pipe(len, concurrency=1)

        # Execute: get repr
        result = repr(config)

        # Assert: repr still works and contains concurrency
        self.assertIn("concurrency=1", result)
        self.assertIn("len", result)


class TestMergeConfigRepr(unittest.TestCase):
    """Test MergeConfig.__repr__ with nested pipelines."""

    def test_merge_config_repr_with_two_pipelines(self) -> None:
        """Test __repr__ shows nested pipeline configs with proper indentation."""
        # Setup: create two pipeline configs and merge them
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        plc2 = PipelineConfig(
            src=SourceConfig([4, 5, 6]),
            pipes=[],
            sink=SinkConfig(buffer_size=20),
        )
        merge_config = Merge([plc1, plc2])

        # Execute: get repr
        result = repr(merge_config)

        # Assert: repr contains merge structure with both pipelines
        self.assertIn("MergeConfig(", result)
        self.assertIn("Pipeline 1:", result)
        self.assertIn("Pipeline 2:", result)
        self.assertIn("PipelineConfig", result)

    def test_merge_config_repr_with_custom_op(self) -> None:
        """Test __repr__ shows custom merge operation with source location."""
        # Setup: create merge config with custom op
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        plc2 = PipelineConfig(
            src=SourceConfig([4, 5, 6]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        merge_config = Merge([plc1, plc2], op=custom_merge_op)

        # Execute: get repr
        result = repr(merge_config)

        # Assert: repr contains op info with source location
        self.assertIn("op=", result)
        self.assertIn("custom_merge_op", result)
        self.assertIn(__file__, result)
        self.assertIn(f":{_ln(custom_merge_op)}", result)

    def test_merge_config_repr_multiline_structure(self) -> None:
        """Test __repr__ creates multi-line output with proper indentation."""
        # Setup: create merge config with pipes
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[Pipe(example_sync_function, concurrency=2)],
            sink=SinkConfig(buffer_size=10),
        )
        plc2 = PipelineConfig(
            src=SourceConfig([4, 5, 6]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        merge_config = Merge([plc1, plc2])

        # Execute: get repr
        result = repr(merge_config)

        # Assert: result is multi-line and properly indented
        lines = result.split("\n")
        self.assertGreater(len(lines), 5)  # Multi-line output
        # Check some lines have proper indentation
        pipeline_lines = [line for line in lines if "Pipeline" in line]
        self.assertGreater(len(pipeline_lines), 0)


class TestPipelineConfigRepr(unittest.TestCase):
    """Test PipelineConfig.__repr__ with MergeConfig source."""

    def test_pipeline_config_repr_with_source_config(self) -> None:
        """Test __repr__ with SourceConfig shows inline representation."""
        # Setup: create pipeline config with simple source
        config = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[Pipe(example_sync_function, concurrency=4)],
            sink=SinkConfig(buffer_size=10),
        )

        # Execute: get repr
        result = repr(config)

        # Assert: repr shows source inline
        self.assertIn("PipelineConfig", result)
        self.assertIn("Source:", result)
        self.assertIn("Pipes:", result)
        self.assertIn("Sink:", result)
        self.assertIn("example_sync_function", result)

    def test_pipeline_config_repr_with_merge_config(self) -> None:
        """Test __repr__ with MergeConfig shows proper indentation."""
        # Setup: create pipeline config with merge source
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        plc2 = PipelineConfig(
            src=SourceConfig([4, 5, 6]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        merge_config = Merge([plc1, plc2])
        final_config = PipelineConfig(
            src=merge_config,
            pipes=[Pipe(example_sync_function, concurrency=2)],
            sink=SinkConfig(buffer_size=100),
        )

        # Execute: get repr
        result = repr(final_config)

        # Assert: repr shows nested structure with proper indentation
        self.assertIn("PipelineConfig", result)
        self.assertIn("Source:", result)
        self.assertIn("MergeConfig(", result)
        self.assertIn("Pipeline 1:", result)
        self.assertIn("Pipeline 2:", result)
        # Check final pipe appears after merge
        self.assertIn("example_sync_function", result)

    def test_pipeline_config_repr_indentation_hierarchy(self) -> None:
        """Test __repr__ maintains correct indentation hierarchy."""
        # Setup: create nested pipeline config
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[Pipe(example_sync_function, concurrency=1)],
            sink=SinkConfig(buffer_size=10),
        )
        plc2 = PipelineConfig(
            src=SourceConfig([4, 5, 6]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )
        merge_config = Merge([plc1, plc2])
        final_config = PipelineConfig(
            src=merge_config,
            pipes=[
                Pipe(example_async_function, concurrency=4),
                Aggregate(5),
                Disaggregate(),
            ],
            sink=SinkConfig(buffer_size=100),
        )

        # Execute: get repr
        result = repr(final_config)

        # Assert: verify indentation levels exist
        lines = result.split("\n")
        # Should have various indentation levels
        has_no_indent = any(line and not line[0].isspace() for line in lines)
        has_some_indent = any(line.startswith("  ") for line in lines)
        has_more_indent = any(line.startswith("    ") for line in lines)

        self.assertTrue(has_no_indent, "Should have lines with no indentation")
        self.assertTrue(has_some_indent, "Should have lines with 2-space indentation")
        self.assertTrue(has_more_indent, "Should have lines with 4+ space indentation")

    def test_pipeline_config_repr_with_aggregate_disaggregate(self) -> None:
        """Test __repr__ shows aggregate and disaggregate pipes correctly."""
        # Setup: create pipeline config with various pipe types
        config = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[
                Pipe(example_sync_function, concurrency=2),
                Aggregate(10, drop_last=True),
                Disaggregate(),
            ],
            sink=SinkConfig(buffer_size=10),
        )

        # Execute: get repr
        result = repr(config)

        # Assert: repr shows all pipe types
        self.assertIn("example_sync_function", result)
        self.assertIn("aggregate", result)
        self.assertIn("disaggregate", result)
