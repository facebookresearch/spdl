# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "profile_pipeline",
]

import logging
import time
from dataclasses import dataclass
from typing import TypeVar

from ._build import build_pipeline
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

    qps = len(outputs) / elapsed
    return qps, outputs


def _fetch_inputs(src: SourceConfig[T], num_items: int) -> list[T]:
    pipeline = build_pipeline(
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


def _build_pipeline(
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
class _ProfileResult:
    name: str
    concurrency: list[int]
    qps: list[int]


def profile_pipeline(
    cfg: PipelineConfig[T, U], num_inputs: int = 1000
) -> list[_ProfileResult]:
    """**[Experimental]** Profile pipeline by running pipes separately
    while changing the oncurrency.

    Args:
        cfg: Pipeline configuration.
        num_inputs: The number of source items to use for profiling.

    Returns:
        list of dataclass: List of result objects.
            Each result has the following attributes.

            - ``name``: The name of the pipe.
            - ``concurrency``: The concurrency used to benchmark the performance.
            - ``qps``: The number of itmes each stage processed each second.
    """
    _LG.info("Fetching %d inputs.", num_inputs)
    inputs = _fetch_inputs(cfg.src, num_inputs)

    results = []
    for i, pipe in enumerate(cfg.pipes):
        _LG.info("Profiling Stage %d: %s", i, pipe.name)

        if pipe._type in (_PipeType.Aggregate, _PipeType.Disaggregate):
            concurrencies = [1]
        else:
            concurrencies = [32, 16, 8, 4, 1]

        qps = []
        cfg_ = _build_pipeline(inputs, pipe, max(concurrencies))
        for concurrency in concurrencies:
            _LG.info(" - Concurrency: %d", concurrency)
            pipeline = build_pipeline(cfg_, num_threads=concurrency)
            qps_, outputs = _run(pipeline)
            _LG.info(" - QPS: %.2f", qps_)
            qps.append(qps_)

        inputs = outputs  # pyre-ignore[61]
        results.append(_ProfileResult(pipe.name, concurrencies, qps))

    for result in results:
        _LG.info("Pipe: %s", result.name)
        _LG.info("Concurrency: %s", ",".join(f"{s:8d}" for s in result.concurrency))
        _LG.info("QPS:         %s", ",".join(f"{s:8.2f}" for s in result.qps))

    return results
