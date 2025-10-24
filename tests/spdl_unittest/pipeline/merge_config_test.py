# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MergeConfig class."""

# pyre-unsafe

import asyncio
import unittest
from collections.abc import Sequence

from spdl.pipeline import build_pipeline, create_task, is_eof
from spdl.pipeline.defs import (
    Merge,
    Pipe,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)


class MergeConfigTest(unittest.TestCase):
    """Test MergeConfig functionality."""

    def test_merge_config_with_two_simple_pipelines(self) -> None:
        """Test MergeConfig merges outputs from two simple pipelines."""
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

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 6)
        self.assertCountEqual(results, [1, 2, 3, 4, 5, 6])

    def test_merge_config_with_processed_pipelines(self) -> None:
        """Test MergeConfig merges outputs from pipelines with processing."""
        double_pipe = Pipe(lambda x: x * 2)
        add_ten_pipe = Pipe(lambda x: x + 10)

        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[double_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        plc2 = PipelineConfig(
            src=SourceConfig([4, 5, 6]),
            pipes=[add_ten_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 6)
        # Pipeline 1: [1, 2, 3] -> [2, 4, 6] (doubled)
        # Pipeline 2: [4, 5, 6] -> [14, 15, 16] (added 10)
        self.assertCountEqual(results, [2, 4, 6, 14, 15, 16])

    def test_merge_config_with_multiple_pipelines(self) -> None:
        """Test MergeConfig can merge outputs from three pipelines."""
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        plc2 = PipelineConfig(
            src=SourceConfig([10, 20]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        plc3 = PipelineConfig(
            src=SourceConfig([100, 200]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2, plc3]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=3)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 6)
        self.assertCountEqual(results, [1, 2, 10, 20, 100, 200])

    def test_merge_config_with_post_processing(self) -> None:
        """Test MergeConfig output can be further processed in main pipeline."""
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

        multiply_by_5_pipe = Pipe(lambda x: x * 5)

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[multiply_by_5_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 6)
        self.assertCountEqual(results, [5, 10, 15, 20, 25, 30])

    def test_merge_config_with_async_processing(self) -> None:
        """Test MergeConfig works with async processing functions."""

        async def async_double(x: int) -> int:
            await asyncio.sleep(0.01)  # Small delay to simulate async work
            return x * 2

        async_pipe = Pipe(async_double)

        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[async_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        plc2 = PipelineConfig(
            src=SourceConfig([4, 5, 6]),
            pipes=[async_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=5))

        self.assertEqual(len(results), 6)
        self.assertCountEqual(results, [2, 4, 6, 8, 10, 12])

    def test_merge_config_with_different_data_types(self) -> None:
        """Test MergeConfig works with different data types."""
        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        plc2 = PipelineConfig(
            src=SourceConfig(["a", "b", "c"]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 6)
        self.assertCountEqual(results, [1, 2, 3, "a", "b", "c"])

    def test_merge_config_with_empty_pipeline(self) -> None:
        """Test MergeConfig handles pipeline with empty source."""
        plc1 = PipelineConfig(
            src=SourceConfig([]),  # Empty source
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        plc2 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 3)
        self.assertCountEqual(results, [1, 2, 3])

    def test_merge_config_validation_empty_list(self) -> None:
        """Test MergeConfig validation fails with empty pipeline list."""
        with self.assertRaises(ValueError) as cm:
            Merge([])

        self.assertIn("at least one upstream pipeline", str(cm.exception))

    def test_merge_config_with_aggregation(self) -> None:
        """Test MergeConfig works with aggregation operations."""
        from spdl.pipeline.defs import Aggregate, Disaggregate

        aggregate_pipe = Aggregate(2)
        disaggregate_pipe = Disaggregate()

        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3, 4]),
            pipes=[aggregate_pipe, disaggregate_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        plc2 = PipelineConfig(
            src=SourceConfig([10, 20, 30, 40]),
            pipes=[aggregate_pipe, disaggregate_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 8)
        self.assertCountEqual(results, [1, 2, 3, 4, 10, 20, 30, 40])

    def test_merge_config_with_concurrency(self) -> None:
        """Test MergeConfig works with concurrent processing."""
        concurrent_pipe = Pipe(lambda x: x + 100, concurrency=3)

        plc1 = PipelineConfig(
            src=SourceConfig(list(range(10))),
            pipes=[concurrent_pipe],
            sink=SinkConfig(buffer_size=20),
        )

        plc2 = PipelineConfig(
            src=SourceConfig(list(range(50, 60))),
            pipes=[concurrent_pipe],
            sink=SinkConfig(buffer_size=20),
        )

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[],
            sink=SinkConfig(buffer_size=50),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=4)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 20)
        expected = list(range(100, 110)) + list(range(150, 160))
        self.assertCountEqual(results, expected)

    def test_merge_config_with_different_pipe_counts_and_post_processing(self) -> None:
        """Test MergeConfig merges pipelines with different numbers of pipes and applies post-processing."""

        # Pipeline 1: single pipe (multiply by 2)
        multiply_pipe = Pipe(lambda x: x * 2)

        plc1 = PipelineConfig(
            src=SourceConfig([1, 2, 3]),
            pipes=[multiply_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        # Pipeline 2: three pipes (add 10, multiply by 3, subtract 5)
        add_ten_pipe = Pipe(lambda x: x + 10)
        multiply_by_three_pipe = Pipe(lambda x: x * 3)
        subtract_five_pipe = Pipe(lambda x: x - 5)

        plc2 = PipelineConfig(
            src=SourceConfig([4, 5]),
            pipes=[add_ten_pipe, multiply_by_three_pipe, subtract_five_pipe],
            sink=SinkConfig(buffer_size=10),
        )

        # Add post-processing after merge: add 100 to all merged results
        post_process_pipe = Pipe(lambda x: x + 100)

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2]),
            pipes=[post_process_pipe],
            sink=SinkConfig(buffer_size=20),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        self.assertEqual(len(results), 5)

        # [1, 2, 3] --(multiply by 2)--> [2, 4, 6] --(add 100)--> [102, 104, 106]
        pipeline1_expected = [102, 104, 106]

        # [4, 5] --(add 10)-->        [14, 15]
        #        --(multiply by 3)--> [42, 45]
        #        --(subtract 5)-->    [37, 40]
        #        --(add 100)-->       [137, 140]
        pipeline2_expected = [137, 140]

        expected_all = pipeline1_expected + pipeline2_expected
        self.assertCountEqual(results, expected_all)

    def test_merge_config_with_custom_merge_op(self) -> None:
        """Test MergeConfig accepts and uses custom merge operation."""

        # Track which pipelines contributed items (for verification)
        collected_items = []

        async def custom_merge_op(
            name: str,
            input_queues: Sequence[asyncio.Queue],
            output_queue: asyncio.Queue,
        ) -> None:
            """Custom merge that adds a prefix to each item based on its source pipeline."""

            async def process_queue(queue_idx: int, in_q: asyncio.Queue) -> None:
                while True:
                    item = await in_q.get()
                    if is_eof(item):
                        return
                    # Add prefix based on source pipeline
                    prefixed_item = f"p{queue_idx}_{item}"
                    collected_items.append(prefixed_item)
                    await output_queue.put(prefixed_item)

            tasks = [
                create_task(process_queue(i, in_q), name=f"{name}:{i}")
                for i, in_q in enumerate(input_queues)
            ]
            await asyncio.wait(tasks)

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

        main_pipeline_config = PipelineConfig(
            src=Merge([plc1, plc2], op=custom_merge_op),
            pipes=[],
            sink=SinkConfig(buffer_size=10),
        )

        pipeline = build_pipeline(main_pipeline_config, num_threads=2)

        with pipeline.auto_stop():
            results = list(pipeline.get_iterator(timeout=3))

        # Verify we got all items with prefixes
        self.assertEqual(len(results), 6)
        # Items from pipeline 0 should have p0_ prefix
        self.assertIn("p0_1", results)
        self.assertIn("p0_2", results)
        self.assertIn("p0_3", results)
        # Items from pipeline 1 should have p1_ prefix
        self.assertIn("p1_4", results)
        self.assertIn("p1_5", results)
        self.assertIn("p1_6", results)


if __name__ == "__main__":
    unittest.main()
