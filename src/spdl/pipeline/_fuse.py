# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Fuse each ``.to()`` execution region into one nested subprocess (or subinterpreter) pipeline.

A :py:class:`~spdl.pipeline.defs.PlacementConfig` marker (appended by
:py:meth:`~spdl.pipeline.PipelineBuilder.to`) designates where the stages that follow it run: a
subprocess/subinterpreter worker pool, or back in the main process. This module walks a config's
``pipes``, and replaces each maximal span of stages under a worker-pool target with a single
stage that executes the span as one nested :py:class:`~spdl.pipeline.Pipeline` inside the pool —
so the op→op handoff stays in the worker and intermediate values are never round-tripped (or
pickled) back to the main process.

Every stage in a region is carried into the worker, including
:py:class:`~spdl.pipeline.defs.AggregateConfig`,
:py:class:`~spdl.pipeline.defs.DisaggregateConfig`, and
:py:class:`~spdl.pipeline.defs.PathVariantsConfig`. Because the nested pipeline is built by the
normal :py:func:`~spdl.pipeline._build.build_pipeline`, no execution machinery is special-cased
here — only the segmentation on markers and the executor-stripping rewrite.

The rewrite is a pure function over the stage configs (aside from spawning the worker pools),
which keeps the fusion policy easy to reason about and to test in isolation.
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import sys
import threading
import warnings
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import Any

from spdl.pipeline._components import _get_global_id, _set_global_id
from spdl.pipeline._subprocess_pipeline_pool import (
    _InterpreterBackend,
    _PoolBackend,
    _ProcessBackend,
    _SubprocessPipelinePool,
)
from spdl.pipeline.defs._defs import (
    _MainProcess,
    _SubprocessPipelineConfig,
    InterpreterPoolExecutorConfig,
    MergeConfig,
    PathVariantsConfig,
    PipeConfig,
    PipelineConfig,
    PlacementConfig,
    ProcessPoolExecutorConfig,
    SinkConfig,
    SourceConfig,
)

__all__ = [
    "_fuse_marked_regions",
]

# Buffer size for the fused sub-pipeline's sink. Small: the worker drains it straight onto the
# result queue, so a deep buffer only adds latency.
_FUSED_SINK_BUFFER: int = 3


def _has_continuous_source(config: PipelineConfig[Any]) -> bool:
    """Whether any source in the (possibly merged) config re-iterates continuously."""
    src = config.src
    if isinstance(src, SourceConfig):
        return src.continuous
    if isinstance(src, MergeConfig):
        return any(_has_continuous_source(plc) for plc in src.pipeline_configs)
    return False


def _strip_executor(cfg: object) -> object:
    """Strip a stage's executor(s) so it runs on the worker's own pool.

    Recurses into a path-variants stage, stripping every branch pipe's executor (the branch ops
    then run on the worker's nested thread pool). A pickling boundary is crossed when the config
    ships to the worker, so any live executor — pool or thread — must be removed regardless.
    """
    if isinstance(cfg, PipeConfig):
        return replace(cfg, _args=replace(cfg._args, executor=None))
    if isinstance(cfg, PathVariantsConfig):
        new_paths = tuple(
            tuple(_strip_executor(stage) for stage in path) for path in cfg.paths
        )
        return replace(cfg, paths=new_paths)
    return cfg


def _is_async_op(op: object) -> bool:
    """Whether ``op`` runs on the event loop rather than the worker's thread pool."""
    return inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op)


def _stage_concurrency(cfg: object) -> int:
    """Total worker-thread demand of a fused stage: a pipe's ``concurrency``, or the sum across
    every branch pipe of a path-variants stage (recursively). Other stages contribute 0.

    An async pipe contributes 0: its ``concurrency`` bounds concurrent coroutines on the
    worker's event loop, which do not consume worker threads.
    """
    if isinstance(cfg, PipeConfig):
        return 0 if _is_async_op(cfg._args.op) else cfg._args.concurrency
    if isinstance(cfg, PathVariantsConfig):
        return sum(_stage_concurrency(s) for path in cfg.paths for s in path)
    return 0


def _stage_name(cfg: object) -> str:
    return getattr(cfg, "name", type(cfg).__name__)


def _worker_initializer(
    global_id: int,
    user_initializer: Any,
    user_initargs: tuple[Any, ...],
) -> None:
    """Run inside each worker before the sub-pipeline is built.

    Aligns the worker's pipeline-id base with the parent so per-stage stats are attributed
    consistently, then runs the user's executor initializer (if any), preserving the side
    effects (e.g. dataset setup) the user attached to their pool.
    """
    _set_global_id(global_id)
    if user_initializer is not None:
        user_initializer(*user_initargs)


def _warn_fork_with_threads(ctx: Any, stacklevel: int) -> None:
    if ctx.get_start_method() == "fork" and threading.active_count() > 1:
        warnings.warn(
            "Fusing subprocess pipeline stages with the 'fork' start method from a "
            "multi-threaded process can deadlock. Pass mp_context='spawn' or 'forkserver', "
            "or build the pipeline before other threads start.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def _build_fused_stage_core(
    stages: Sequence[object],
    *,
    backend: _PoolBackend,
    num_threads: int,
    max_workers: int,
    user_initializer: Any,
    user_initargs: tuple[Any, ...],
    report_stats_interval: float,
    continuous: bool,
) -> tuple[_SubprocessPipelineConfig, _SubprocessPipelinePool]:
    """Spawn a worker pool that runs ``stages`` as a nested pipeline; return the pool and its
    replacement stage. Called by :py:func:`_build_fused_stage_from_spec` once the pool
    parameters and backend have been resolved from a region's spec."""
    stripped = [_strip_executor(s) for s in stages]
    sub_config: PipelineConfig[Any] = PipelineConfig(
        src=SourceConfig(
            []
        ),  # placeholder; the worker sets the real source per session
        pipes=stripped,  # pyre-ignore[6]
        sink=SinkConfig(_FUSED_SINK_BUFFER),
    )
    build_kwargs = {
        "num_threads": num_threads,
        "report_stats_interval": report_stats_interval,
    }
    initializer = partial(
        _worker_initializer, _get_global_id(), user_initializer, user_initargs
    )
    pool = _SubprocessPipelinePool(
        backend,
        max_workers,
        sub_config,
        build_kwargs,
        initializer,
        (),
        continuous=continuous,
    )
    name = "subprocess_pipeline(" + "+".join(_stage_name(s) for s in stages) + ")"
    return _SubprocessPipelineConfig(name=name, handle=pool.make_handle()), pool


def _build_fused_stage_from_spec(
    stages: Sequence[object],
    spec: ProcessPoolExecutorConfig | InterpreterPoolExecutorConfig,
    backend: _PoolBackend,
    report_stats_interval: float,
    continuous: bool,
) -> tuple[_SubprocessPipelineConfig, _SubprocessPipelinePool]:
    """Build the worker pool and replacement stage for one ``.to()`` region, reading the pool
    parameters from ``spec``. ``num_threads`` for the nested pipeline is the sum of the region
    stages' concurrency, ``max_workers`` falls back to the CPU count, and
    ``report_stats_interval`` is inherited from
    :py:func:`~spdl.pipeline._build.build_pipeline`. ``backend`` (process or subinterpreter) is
    chosen by the caller from the spec type."""
    num_threads = max(1, sum(_stage_concurrency(s) for s in stages))
    max_workers = spec.max_workers or os.cpu_count() or 1
    return _build_fused_stage_core(
        stages,
        backend=backend,
        num_threads=num_threads,
        max_workers=max_workers,
        user_initializer=spec.initializer,
        user_initargs=spec.initargs,
        report_stats_interval=report_stats_interval,
        continuous=continuous,
    )


def _has_executor_markers(pipes: Sequence[object]) -> bool:
    """Whether ``pipes`` contains any ``.to()`` region marker."""
    return any(isinstance(p, PlacementConfig) for p in pipes)


def _fuse_marked_regions(
    config: PipelineConfig[Any],
    *,
    report_stats_interval: float = -1,
    stacklevel: int = 3,
) -> tuple[PipelineConfig[Any], list[_SubprocessPipelinePool]]:
    """Fuse each ``.to()`` region into one subprocess-pipeline stage.

    Walks ``config.pipes`` tracking the current execution target set by
    :py:class:`~spdl.pipeline.defs.PlacementConfig` markers (a pipeline starts on the main
    process). Every maximal span of stages under a subprocess/subinterpreter target — pipes *and*
    the aggregate/disaggregate/path-variants stages between them — is replaced by a single stage
    that runs the span as one nested pipeline inside a worker pool. Main-process spans are kept
    unchanged, and the marker nodes themselves are dropped. The spawned pools are returned for the
    caller to reap at teardown; the input ``config`` is not mutated.

    This is a no-op (returns ``config`` and no pools) when there are no markers, so it is safe to
    run unconditionally.

    Args:
        config: The pipeline configuration to rewrite.
        report_stats_interval: Fallback stats interval for a region whose spec does not set one.
        stacklevel: ``warnings.warn`` stack level measured at this function. The fork-with-threads
            warning is raised from the nested ``_flush`` closure (two frames deeper), so it uses
            ``stacklevel + 2``.

    Returns:
        A tuple ``(new_config, pools)``. ``pools`` is empty when no region is fused.
    """
    pipes = list(config.pipes)
    if not _has_executor_markers(pipes):
        return config, []

    continuous = _has_continuous_source(config)
    pools: list[_SubprocessPipelinePool] = []
    new_pipes: list[object] = []
    target: ProcessPoolExecutorConfig | InterpreterPoolExecutorConfig | _MainProcess = (
        _MainProcess()
    )
    region: list[object] = []

    def _flush() -> None:
        if not region:
            return
        backend: _PoolBackend
        if isinstance(target, ProcessPoolExecutorConfig):
            ctx = mp.get_context(target.mp_context)
            # +2, not +1: this runs inside the nested ``_flush`` closure, one frame deeper than
            # ``_fuse_marked_regions`` itself, so the warning still points at the user's call site.
            _warn_fork_with_threads(ctx, stacklevel + 2)
            backend = _ProcessBackend(ctx)
        elif isinstance(target, InterpreterPoolExecutorConfig):
            if sys.version_info < (3, 14):
                raise RuntimeError(
                    "Subinterpreter regions (`.to(InterpreterPoolExecutorConfig(...))`) require "
                    "Python 3.14 or later. Current version: "
                    f"{sys.version_info.major}.{sys.version_info.minor}"
                )
            backend = _InterpreterBackend()
        else:  # pragma: no cover -- a main-process target never accumulates a region
            raise AssertionError(f"Unexpected region target: {target!r}")
        fused, pool = _build_fused_stage_from_spec(
            region, target, backend, report_stats_interval, continuous
        )
        pools.append(pool)
        new_pipes.append(fused)
        region.clear()

    try:
        for p in pipes:
            if isinstance(p, PlacementConfig):
                _flush()  # close the span running under the previous target
                target = p.target
            elif isinstance(target, _MainProcess):
                new_pipes.append(p)
            else:
                region.append(p)
        _flush()  # close a region left open at the end of the pipes
    except BaseException:
        # Reap any pools spawned before the failure; the caller never receives them to reap.
        for pool in pools:
            pool.shutdown()
        raise
    return replace(config, pipes=new_pipes), pools  # pyre-ignore[6]
