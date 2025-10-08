# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from . import _build
from ._pipeline import Pipeline
from .defs._defs import (
    _PipeArgs,
    _PipeType,
    PipeConfig,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)

# pyre-strict

__all__ = [
    "profile_pipeline",
]


T = TypeVar("T")
U = TypeVar("U")

_LG: logging.Logger = logging.getLogger(__name__)


def _run(pipeline: Pipeline[T]) -> tuple[float, list[T]]:
    outputs = []
    t0 = time.monotonic()
    try:
        with pipeline.auto_stop():
            for item in pipeline:
                outputs.append(item)
    finally:
        elapsed = time.monotonic() - t0

    qps = len(outputs) / elapsed if elapsed > 0 else float("nan")
    return qps, outputs


def _fetch_inputs(src: SourceConfig[T], num_items: int) -> list[T]:
    pipeline = _build._build_pipeline(
        PipelineConfig(
            src=src,
            pipes=[],
            sink=SinkConfig(1),
        ),
        num_threads=1,
    )

    ret = []
    with pipeline.auto_stop():
        for item in pipeline:
            ret.append(item)
            if len(ret) >= num_items:
                break
    return ret


def _build_pipeline_config(
    src: list[T], pipe: PipeConfig[T, U], concurrency: int
) -> PipelineConfig[T, U]:
    return PipelineConfig(
        src=SourceConfig(src),
        pipes=[
            PipeConfig(
                name=pipe.name,
                _type=pipe._type,
                _args=_PipeArgs(
                    op=pipe._args.op,
                    executor=None,
                    concurrency=concurrency,
                    op_requires_eof=pipe._args.op_requires_eof,
                ),
            )
        ],
        sink=SinkConfig(2),
    )


@dataclass
class _ProfileStats:
    concurrency: int
    qps: float
    occupancy_rate: float


@dataclass
class _ProfileResult:
    name: str
    stats: list[_ProfileStats]


def _get_local_rank() -> int:
    # https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/distributed_c10d.py#L1445-L1468
    # LOCAL_RANK env var is how device mappingis communicated between PyTorch and
    # external system, so it should be a stable way to check the device without
    # importing PyTorch.
    return int(os.environ.get("LOCAL_RANK", "0"))


def profile_pipeline(
    cfg: PipelineConfig[T, U],
    num_inputs: int = 1000,
    *,
    callback: Callable[[_ProfileResult], None] | None = None,
) -> list[_ProfileResult]:
    """**[Experimental]** Profile pipeline by running pipes separately
    while changing the concurrency, measuring performance and logging results.

    This function benchmarks each pipeline stage independently across different
    concurrency levels (32, 16, 8, 4, 1) to identify optimal performance settings.
    It measures both throughput (QPS) and queue occupancy rates.

    Args:
        cfg: Pipeline configuration containing source, pipes, and sink definitions.
        num_inputs: The number of source items to use for profiling each stage.
        callback: Optional function that, if provided, will be called with the profiling
            result (``_ProfileResult``) for each pipeline stage after it is benchmarked.
            This allows for custom handling or logging of profiling results as they are produced.

    Returns:
        List of ``_ProfileResult`` objects, one per pipeline stage.
        Each result contains:

        - ``name``: The name of the pipe stage.
        - ``stats``: List of ``_ProfileStats`` for each concurrency level tested, where each stat includes:

          - ``concurrency``: The concurrency level used for this benchmark.
          - ``qps``: The number of items the stage processed per second.
          - ``occupancy_rate``: The percentage of time the queue was occupied (0.0 to 1.0).
    """
    if _get_local_rank() != 0:
        _LG.info(
            "Distributed training is enabled. Profiling is only performed on local rank 0. "
            "Exiting without profiling."
        )
        return []

    _LG.info("Fetching %d inputs.", num_inputs)
    inputs = _fetch_inputs(cfg.src, num_inputs)

    results = []
    for i, pipe in enumerate(cfg.pipes):
        _LG.info("Profiling Stage %d: %s", i, pipe.name)

        if pipe._type in (_PipeType.Aggregate, _PipeType.Disaggregate):
            concurrencies = [1]
        else:
            concurrencies = [32, 16, 8, 4, 1]

        stats = []
        cfg_ = _build_pipeline_config(inputs, pipe, max(concurrencies))
        for concurrency in concurrencies:
            pipeline = _build._build_pipeline(cfg_, num_threads=concurrency)
            qps_, outputs = _run(pipeline)
            occupancy_rate = (
                pipeline._output_queue._get_lap_stats().occupancy_rate  # pyre-ignore[16]
            )
            _LG.info(" - Concurrency: %d", concurrency)
            _LG.info(" - QPS: %.2f", qps_)
            _LG.info(" - Occupancy Rate: %.2f", occupancy_rate)

            stats.append(_ProfileStats(concurrency, qps_, occupancy_rate))

        inputs = outputs  # pyre-ignore[61]
        result = _ProfileResult(pipe.name, stats)
        if callback is not None:
            callback(result)
        results.append(result)

    return results
