# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from . import _build, _config
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
    "ProfileResult",
    "_build_pipeline_diagnostic_mode",
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


class ProfileHook(ABC):
    """A hook object that can be used to execute custom code before and after each stage and pipeline profiling."""

    @abstractmethod
    @contextmanager
    def stage_profile_hook(self) -> Iterator[None]:
        """A context manager that is executed around each stage profiling."""
        ...

    @abstractmethod
    @contextmanager
    def pipeline_profile_hook(self) -> Iterator[None]:
        """A context manager that is executed at the beginning and the end of :py:func:`profile_pipeline` function."""
        ...


class _NoOpHook(ProfileHook):
    @contextmanager
    def stage_profile_hook(self) -> Iterator[None]:
        yield

    @contextmanager
    def pipeline_profile_hook(self) -> Iterator[None]:
        yield


@dataclass
class _ProfileStats:
    concurrency: int
    qps: float
    occupancy_rate: float


@dataclass
class ProfileResult:
    """A data class contains profiling result, returned by :py:func:`profile_pipeline`."""

    name: str
    """The name of the pipe stage."""

    stats: list["_ProfileStats"]
    """Dataclass objects for each concurrency level tested, where each stat includes:

      - ``concurrency``: The concurrency level used for this benchmark.
      - ``qps``: The number of items the stage processed per second.
      - ``occupancy_rate``: The percentage of time the queue was occupied (0.0 to 1.0)."""


_ProfileResult = ProfileResult  # temp


def no_op(_: ProfileResult) -> None:
    pass


_DEFAULT_HOOK = _NoOpHook()
_DEFAULT_CALLBACK = no_op


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
    callback: Callable[[ProfileResult], None] | None = None,
    hook: ProfileHook | None = None,
) -> list[ProfileResult]:
    """**[Experimental]** Profile pipeline by running pipes separately
    while changing the concurrency, measuring performance and logging results.

    This function benchmarks each pipeline stage independently across different
    concurrency levels (32, 16, 8, 4, 1) to identify optimal performance settings.
    It measures both throughput (QPS) and queue occupancy rates.

    Args:
        cfg: Pipeline configuration containing source, pipes, and sink definitions.
        num_inputs: The number of source items to use for profiling each stage.
        callback: Optional function that, if provided, will be called with the profiling
            result for each pipeline stage after it is benchmarked.
            This allows for custom handling or logging of profiling results as they are produced.
        hook: Optional hook object, which can be used to execute custom code before and after
            each stage and pipeline profiling.

    Returns:
        List of ProfileResult objects, one per pipeline stage.

    Example:
        .. code-block:: python

            from spdl.pipeline import PipelineConfig, SourceConfig, SinkConfig, Pipe, profile_pipeline

            # Define a simple data processing function
            def double_value(x):
                return x * 2

            def square_value(x):
                return x ** 2

            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                src=SourceConfig(range(1000)),  # Source with 1000 integers
                pipes=[
                    Pipe(double_value, concurrency=4, name="double"),
                    Pipe(square_value, concurrency=2, name="square")
                ],
                sink=SinkConfig(buffer_size=10)
            )

            # Profile the pipeline
            results = profile_pipeline(pipeline_config, num_inputs=500)

            # The results list contains ProfileResult objects for each pipe stage
            for result in results:
                print(f"Stage: {result.name}")
                for stat in result.stats:
                    print(f"  Concurrency {stat.concurrency}: "
                          f"QPS={stat.qps:.2f}, "
                          f"Occupancy={stat.occupancy_rate:.2f}")

            # Example output:
            # Stage: double
            #   Concurrency 32: QPS=1250.45, Occupancy=0.85
            #   Concurrency 16: QPS=1180.32, Occupancy=0.78
            #   Concurrency 8: QPS=1050.21, Occupancy=0.65
            #   Concurrency 4: QPS=850.12, Occupancy=0.45
            #   Concurrency 1: QPS=320.88, Occupancy=0.25
            # Stage: square
            #   Concurrency 32: QPS=2100.67, Occupancy=0.92
            #   Concurrency 16: QPS=1980.55, Occupancy=0.88
            #   Concurrency 8: QPS=1750.33, Occupancy=0.75
            #   Concurrency 4: QPS=1200.44, Occupancy=0.55
            #   Concurrency 1: QPS=450.22, Occupancy=0.30
    """
    hook_ = hook or _DEFAULT_HOOK
    callback_ = callback or _DEFAULT_CALLBACK

    with hook_.pipeline_profile_hook():
        if _get_local_rank() != 0:
            _LG.info(
                "Distributed training is enabled. Profiling is only performed on local rank 0. "
                "Exiting without profiling."
            )
            return []

        _LG.info("Fetching %d inputs.", num_inputs)
        if isinstance(cfg.src, SourceConfig):
            inputs = _fetch_inputs(cfg.src, num_inputs)
        else:
            raise ValueError("Pipeline with MergeConfig is not currently supported.")

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
                with hook_.stage_profile_hook():
                    qps_, outputs = _run(pipeline)
                occupancy_rate = (
                    pipeline._output_queue._get_lap_stats().occupancy_rate  # pyre-ignore[16]
                )
                _LG.info(" - Concurrency: %d", concurrency)
                _LG.info(" - QPS: %.2f", qps_)
                _LG.info(" - Occupancy Rate: %.2f", occupancy_rate)

                stats.append(_ProfileStats(concurrency, qps_, occupancy_rate))

            inputs = outputs  # pyre-ignore[61]
            result = ProfileResult(pipe.name, stats)
            callback_(result)
            results.append(result)

    return results


##############################################################################
# Self-Diagnostic Pipeline
##############################################################################


class _ProfilePipeline(Pipeline[U]):
    def __init__(self, pipeline_cfg: PipelineConfig[T, U], num_items: int) -> None:
        self._pipeline_cfg = pipeline_cfg
        self._num_items = num_items

    def __str__(self) -> str:
        return str(self._pipeline_cfg)

    def start(self, *, timeout: float | None = None, **kwargs: Any) -> None:
        pass

    def stop(self, *, timeout: float | None = None, **kwargs: Any) -> None:
        pass

    def get_item(self, *, timeout: float | None = None) -> U:  # noqa: ARG002
        profile_pipeline(self._pipeline_cfg, self._num_items)
        _LG.info("Profiling completed. Exiting.")
        raise SystemExit(0)


def _build_pipeline_diagnostic_mode(cfg: PipelineConfig[T, U]) -> Pipeline[U]:
    num_items = _config._diagnostic_mode_num_sources()
    return _ProfilePipeline(cfg, num_items)
