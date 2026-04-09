#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Comprehensive example defining a complex pipeline with :py:mod:`spdl.pipeline.defs`.

This example showcases the usage of all configuration classes available in the
:py:mod:`spdl.pipeline.defs` module, including:

.. py:currentmodule:: spdl.pipeline.defs

- :py:class:`SourceConfig`: Configures data sources for pipelines
- :py:class:`PipeConfig`: Configures individual processing stages (via factory functions)
- :py:class:`SinkConfig`: Configures output buffering for pipelines
- :py:class:`PipelineConfig`: Top-level pipeline configuration combining all components
- :py:class:`Merge`: Merges outputs from multiple pipelines into a single stream
- :py:class:`PathVariantsConfig`: Routes items to different processing paths

The example also demonstrates the factory functions:

- :py:func:`Pipe`: Creates pipe configurations for general processing
- :py:func:`Aggregate`: Creates configurations for batching/grouping data
- :py:func:`Disaggregate`: Creates configurations for splitting batched data
- :py:func:`PathVariants`: Creates variant path routing configurations

.. note::

   This pipeline uses the merge mechanism, which is not supported by
   :py:class:`~spdl.pipeline.PipelineBuilder`.

.. mermaid::

   graph TD
       subgraph "Pipeline 1"
           S1[Source: range#40;5#41;] --> P1[Pipe: square]
           P1 --> AGG1[Aggregate: batch_size=2]
           AGG1 --> SNK1[Sink 1]
       end

       subgraph "Pipeline 2"
           S2[Source: range#40;10,15#41;] --> P2[Pipe: add_100]
           P2 --> SNK2[Sink 2]
       end

       subgraph "Main Pipeline"
           SNK1 --> M[Merge]
           SNK2 --> M
           M --> NORM[Pipe: normalize_to_lists]
           NORM --> DISAGG[Disaggregate]
           DISAGG --> R[Router: cache_router]
           R -->|cache hit| CACHE[Pipe: load_from_cache]
           R -->|cache miss| MUL[Pipe: multiply_by_10]
           MUL --> STORE[Pipe: store_in_cache]
           CACHE --> PV_MERGE[PathVariants merge]
           STORE --> PV_MERGE
           PV_MERGE --> FINAL_SINK[Final Sink]
       end

The data flow:

1. Pipeline 1:
   ``[0, 1, 2, 3, 4]`` → ``square`` → ``[[0, 1], [4, 9], [16]]``
2. Pipeline 2:
   ``[10, 11, 12, 13, 14]`` → ``add_100`` → ``[110, 111, 112, 113, 114]``
3. Merge combines outputs:
   ``[[0, 1], [4, 9], [16], 110, 111, 112, 113, 114]``
4. Normalize handles mixed data types:
   ``[[0, 1], [4, 9], [16], [110], [111], [112], [113], [114]]``
5. Disaggregate flattens:
   ``[0, 1, 4, 9, 16, 110, 111, 112, 113, 114]``
6. PathVariants routes each item based on cache membership:

   - Cache hits (pre-populated values) → ``load_from_cache`` (1 stage)
   - Cache misses → ``multiply_by_10`` → ``store_in_cache`` (2 stages:
     compute ``×10``, then store the result for future lookups)
"""

__all__ = [
    "main",
    "create_sub_pipeline_1",
    "create_sub_pipeline_2",
    "create_main_pipeline",
    "square",
    "add_100",
    "multiply_by_10",
    "store_in_cache",
    "normalize_to_lists",
    "cache_router",
    "load_from_cache",
    "run_pipeline_example",
]

import logging
from typing import Any

from spdl.pipeline import build_pipeline
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

# pyre-strict

_LG: logging.Logger = logging.getLogger(__name__)


def square(x: int) -> int:
    """Square the input number."""
    return x * x


def add_100(x: int) -> int:
    """Add 100 to the input number."""
    return x + 100


################################################################################
# Cache-based routing with PathVariants
################################################################################

# Cache of pre-computed results.  In a real pipeline this might be backed by
# a key-value store or an on-disk cache; here we use a plain dict.
_CACHE: dict[int, int] = {0: 0, 1: 10, 4: 40}  # pre-populated with a few ×10 values


def cache_router(item: int) -> int:
    """Route items based on cache membership.

    Returns 0 for cache hits (fast path) and 1 for cache misses (slow path).
    """
    return 0 if item in _CACHE else 1


def load_from_cache(item: int) -> int:
    """Retrieve a previously computed result from the cache (fast path)."""
    return _CACHE[item]


def multiply_by_10(item: int) -> tuple[int, int]:
    """Multiply the input by 10 (slow path, first stage).

    Returns a (key, result) tuple so the next stage can store it in the cache.
    """
    return (item, item * 10)


def store_in_cache(item: tuple[int, int]) -> int:
    """Store the computed result in the cache and return it (slow path, second stage)."""
    key, value = item
    _CACHE[key] = value
    return value


def create_sub_pipeline_1() -> PipelineConfig[list[int]]:
    """Create a sub-pipeline that squares numbers and aggregates them.

    .. code-block:: text

       range(5)
        → square
        → aggregate(2)
        → [[squared_pairs], [remaining]]

    Returns:
        Configuration for a pipeline that processes
        ``[0,1,2,3,4]`` into batches of squared values.
    """
    source_config = SourceConfig(range(5))
    square_pipe = Pipe(square)
    aggregate_pipe = Aggregate(2, drop_last=False)
    sink_config = SinkConfig(buffer_size=10)
    return PipelineConfig(
        src=source_config,
        pipes=[square_pipe, aggregate_pipe],
        sink=sink_config,
    )


def create_sub_pipeline_2() -> PipelineConfig[int]:
    """Create a sub-pipeline that adds 100 to numbers.

    .. code-block:: text

       range(10,15)
        → add_100
        → individual_values

    Returns:
        Configuration for a pipeline that processes
        ``[10,11,12,13,14]`` by adding ``100``.
    """
    source_config = SourceConfig(range(10, 15))
    add_pipe = Pipe(add_100, concurrency=2)
    sink_config = SinkConfig(buffer_size=5)

    return PipelineConfig(
        src=source_config,
        pipes=[add_pipe],
        sink=sink_config,
    )


def normalize_to_lists(item: Any) -> list[Any]:
    """Flatten lists or wrap individual items in a list for uniform handling."""
    if isinstance(item, list):
        return item
    else:
        return [item]


def create_main_pipeline(
    sub_pipeline_1: PipelineConfig[list[int]],
    sub_pipeline_2: PipelineConfig[int],
) -> PipelineConfig[int]:
    """Create the main pipeline that merges outputs from sub-pipelines.

    After merging, normalising and disaggregating, the pipeline uses
    :py:func:`PathVariants` to route each item through a cache-aware
    multiply-by-10 stage:

    - **Cache hit** → ``load_from_cache`` (1 pipe stage)
    - **Cache miss** → ``multiply_by_10`` → ``store_in_cache`` (2 pipe stages)

    Note that the two paths have different numbers of pipe stages,
    which PathVariants supports.

    .. code-block:: text

       Merge([sub1, sub2])
         → normalize_to_lists
         → disaggregate
         → PathVariants(cache_router)
             path 0 (hit):  load_from_cache
             path 1 (miss): multiply_by_10 → store_in_cache

    Args:
        sub_pipeline_1: First sub-pipeline configuration
        sub_pipeline_2: Second sub-pipeline configuration

    Returns:
        Main pipeline configuration that merges and processes the sub-pipeline outputs.
    """
    merge_config = Merge([sub_pipeline_1, sub_pipeline_2])
    normalize_pipe = Pipe(normalize_to_lists)
    disaggregate_pipe = Disaggregate()

    # Instead of a plain Pipe(multiply_by_10), use PathVariants to
    # demonstrate cache-based routing: items already in the cache are
    # served instantly, while cache misses compute the result and
    # populate the cache for future lookups.
    cached_multiply = PathVariants(
        router=cache_router,
        paths=[
            [Pipe(load_from_cache)],  # path 0: 1-stage (fast)
            [
                Pipe(multiply_by_10),
                Pipe(store_in_cache),
            ],  # path 1: 2-stage (compute + cache)
        ],
    )

    sink_config = SinkConfig(buffer_size=20)

    return PipelineConfig(
        src=merge_config,
        pipes=[normalize_pipe, disaggregate_pipe, cached_multiply],
        sink=sink_config,
    )


def run_pipeline_example() -> list[int]:
    """Execute the complete pipeline example and return results.

    Returns:
        List of processed integers from the merged pipeline execution.
    """
    _LG.info("Creating sub-pipeline configurations...")

    sub_pipeline_1 = create_sub_pipeline_1()
    sub_pipeline_2 = create_sub_pipeline_2()

    _LG.info("Sub-pipeline 1: %s", sub_pipeline_1)
    _LG.info("Sub-pipeline 2: %s", sub_pipeline_2)

    main_pipeline_config = create_main_pipeline(sub_pipeline_1, sub_pipeline_2)

    _LG.info("Main pipeline config: %s", main_pipeline_config)

    _LG.info("Building the pipeline.")
    pipeline = build_pipeline(main_pipeline_config, num_threads=4)

    _LG.info("Executing the pipeline.")
    results = []
    with pipeline.auto_stop():
        for item in pipeline:
            results.append(item)

    return results


def run() -> None:
    """Run example pipeline and check the result."""
    results = run_pipeline_example()

    _LG.info("Final results: %s", results)
    _LG.info("Number of items processed: %d", len(results))

    # Verify expected data flow
    expected_squared = [0, 1, 4, 9, 16]  # squares of 0-4
    expected_added = [110, 111, 112, 113, 114]  # 10-14 + 100
    expected_combined_count = len(expected_squared) + len(expected_added)

    if len(results) != expected_combined_count:
        raise RuntimeError(
            f"✗ Unexpected number of items: got {len(results)}, expected {expected_combined_count}"
        )
    _LG.info("✓ Pipeline processed expected number of items")


def main() -> None:
    """Main entry point demonstrating all pipeline configuration classes."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    run()


if __name__ == "__main__":
    main()
