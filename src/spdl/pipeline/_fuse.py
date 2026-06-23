# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Detect runs of consecutive pipe stages that can be fused into one subprocess sub-pipeline.

When several consecutive :py:func:`~spdl.pipeline.defs.Pipe` stages share the **same**
process-pool (or interpreter-pool) executor instance, SPDL round-trips data back to the main
process between every stage (each stage compiles to ``loop.run_in_executor(executor, op, item)``
in :py:mod:`spdl.pipeline._common._convert`). That inter-stage IPC is wasteful and fails
outright when an intermediate value is unpicklable.

This module finds the maximal *fusable runs* — spans of the config's ``pipes`` that can be
combined into a single nested :py:class:`~spdl.pipeline.Pipeline` executed inside the worker
pool, so the op→op handoff stays in-process. The actual config rewrite and worker-pool spawning
build on the runs found here.

The detection is a pure function over the stage configs (no process spawning), which keeps the
fusion policy easy to reason about and to test in isolation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import warnings
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from spdl.pipeline._common._convert import _is_process_pool
from spdl.pipeline._components import _get_global_id, _set_global_id
from spdl.pipeline._executor_proxy import _ensure_executor_unused
from spdl.pipeline._subprocess_pipeline_pool import _SubprocessPipelinePool
from spdl.pipeline.defs._defs import (
    _PipeType,
    _SubprocessPipelineConfig,
    MergeConfig,
    PipeConfig,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)

__all__ = [
    "_FusableRun",
    "_find_fusable_runs",
    "_fuse_subprocess_stages",
    "_is_interpreter_pool",
    "_is_isolating_pool",
]

# Buffer size for the fused sub-pipeline's sink. Small: the worker drains it straight onto the
# result queue, so a deep buffer only adds latency.
_FUSED_SINK_BUFFER: int = 3


# Declared before the branch so its type stays ``type[Executor] | None`` on every Python
# version; otherwise, on versions without ``InterpreterPoolExecutor`` the ``else`` assignment
# narrows it to ``None`` and the isinstance/issubclass checks below fail to type-check.
_INTERPRETER_POOL_CLASS: type[Executor] | None
if sys.version_info >= (3, 14):
    from concurrent.futures.interpreter import (  # pyre-ignore[21]
        InterpreterPoolExecutor,
    )

    _INTERPRETER_POOL_CLASS = InterpreterPoolExecutor
else:
    _INTERPRETER_POOL_CLASS = None


def _is_interpreter_pool(executor: Executor | type[Executor] | None) -> bool:
    """Check whether ``executor`` is or wraps an ``InterpreterPoolExecutor``.

    Mirrors :py:func:`spdl.pipeline._common._convert._is_process_pool`: matches the stdlib
    ``InterpreterPoolExecutor`` (Python 3.14+) directly, SPDL pool wrappers via their
    ``_pool_executor_class``, and a ``PriorityExecutorEntrypoint`` via its ``_owner``.

    Args:
        executor: The executor (or executor class, or ``None``) to inspect.

    Returns:
        ``True`` if the executor isolates work in a subinterpreter, else ``False``. Always
        ``False`` on Python versions without ``InterpreterPoolExecutor``.
    """
    if _INTERPRETER_POOL_CLASS is None or executor is None:
        return False
    if isinstance(executor, _INTERPRETER_POOL_CLASS):
        return True
    pool_cls = getattr(executor, "_pool_executor_class", None)
    if pool_cls is not None:
        return issubclass(pool_cls, _INTERPRETER_POOL_CLASS)
    owner = getattr(executor, "_owner", None)
    if owner is not None:
        return _is_interpreter_pool(owner)
    return False


def _is_isolating_pool(executor: Executor | type[Executor] | None) -> bool:
    """Check whether ``executor`` runs work in a separate process or subinterpreter.

    These are exactly the executors whose inter-stage handoff crosses an IPC / pickling
    boundary, and therefore the executors that fusion targets. Thread pools (which share
    address space) and ``None`` (the default thread pool) are not isolating.

    Args:
        executor: The executor (or executor class, or ``None``) to inspect.

    Returns:
        ``True`` if the executor is a process pool or interpreter pool, else ``False``.
    """
    return _is_process_pool(executor) or _is_interpreter_pool(executor)


@dataclass(frozen=True)
class _FusableRun:
    """A maximal span of ``pipes`` that can be fused into one subprocess sub-pipeline.

    The fused stages are ``pipes[start:stop]`` — two or more *adjacent* pool-pipes sharing
    :py:attr:`executor`. Only consecutive pool-pipes fuse; aggregate/disaggregate micro-stages
    are never absorbed (so their batching semantics are unchanged) and instead bound a run.
    """

    start: int
    """Index of the first fused stage in ``pipes`` (inclusive)."""

    stop: int
    """Index just past the last fused stage in ``pipes`` (exclusive)."""

    executor: Executor
    """The isolating-pool executor instance shared by every pool-pipe in the run."""


def _fusable_pool_executor(cfg: object) -> Executor | None:
    """Return the executor if ``cfg`` is a completion-ordered isolating-pool pipe.

    Returns ``None`` otherwise. Input-ordered (``output_order="input"``) pipes are not fusable:
    their global input order cannot be preserved across a pool of independent workers.
    """
    if not isinstance(cfg, PipeConfig):
        return None
    if cfg._type is not _PipeType.Pipe:
        return None
    executor = cfg._args.executor
    if executor is not None and _is_isolating_pool(executor):
        return executor
    return None


def _scan_run(
    pipes: Sequence[object], start: int, executor: Executor
) -> tuple[_FusableRun | None, int]:
    """Scan one fusable run of adjacent same-executor pool-pipes from ``pipes[start]``.

    Greedily consumes the immediately-following pool-pipes that share ``executor``, stopping at
    the first stage that is anything else — a different executor, a thread/None/async pipe, an
    aggregate/disaggregate micro-stage, a path-variant stage, an input-ordered pool-pipe, or the
    end. Only adjacent pool-pipes fuse, so a micro-stage between two pool-pipes splits them into
    separate (and therefore unfused) runs and keeps its main-process batching semantics.

    Returns:
        A tuple ``(run, next_index)``. ``run`` is the emitted :py:class:`_FusableRun`, or
        ``None`` for a lone pool-pipe with no same-executor neighbour. ``next_index`` is where
        the outer scan should resume.
    """
    j = start + 1
    n = len(pipes)
    while j < n and _fusable_pool_executor(pipes[j]) is executor:
        j += 1
    if j - start >= 2:
        return _FusableRun(start, j, executor), j
    return None, j


def _find_fusable_runs(pipes: Sequence[object]) -> list[_FusableRun]:
    """Find all maximal fusable runs in a pipeline config's ``pipes``.

    A run is a span of ≥2 adjacent completion-ordered :py:func:`~spdl.pipeline.defs.Pipe`
    stages sharing the same isolating-pool executor instance. A lone pool-pipe, and any
    aggregate/disaggregate micro-stage, break a run and are left unchanged in the main process.
    Returned runs are ordered by position and never overlap.

    Args:
        pipes: The ordered stage configs from a :py:class:`~spdl.pipeline.defs.PipelineConfig`.

    Returns:
        The list of fusable runs; empty when no fusion applies.
    """
    runs: list[_FusableRun] = []
    i = 0
    n = len(pipes)
    while i < n:
        executor = _fusable_pool_executor(pipes[i])
        if executor is None:
            i += 1
            continue
        run, i = _scan_run(pipes, i, executor)
        if run is not None:
            runs.append(run)
    return runs


################################################################################
# Config rewrite: replace each fusable run with one subprocess-pipeline stage.
################################################################################


def _has_continuous_source(config: PipelineConfig[Any]) -> bool:
    """Whether any source in the (possibly merged) config re-iterates continuously."""
    src = config.src
    if isinstance(src, SourceConfig):
        return src.continuous
    if isinstance(src, MergeConfig):
        return any(_has_continuous_source(plc) for plc in src.pipeline_configs)
    return False


def _strip_executor(cfg: object) -> object:
    """Strip a pool-pipe's executor so it runs on the worker's own pool."""
    if isinstance(cfg, PipeConfig):
        return replace(cfg, _args=replace(cfg._args, executor=None))
    return cfg


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


def _pool_params(executor: Executor) -> tuple[int, Any, tuple[Any, ...]]:
    """Read worker count and initializer off a fresh pool executor without using it."""
    _ensure_executor_unused(executor)
    max_workers = getattr(executor, "_max_workers", None) or os.cpu_count() or 1
    user_initializer = getattr(executor, "_initializer", None)
    user_initargs = getattr(executor, "_initargs", ()) or ()
    return max_workers, user_initializer, user_initargs


def _warn_fork_with_threads(ctx: Any) -> None:
    if ctx.get_start_method() == "fork" and threading.active_count() > 1:
        warnings.warn(
            "Fusing subprocess pipeline stages with the 'fork' start method from a "
            "multi-threaded process can deadlock. Pass mp_context='spawn' or 'forkserver', "
            "or build the pipeline before other threads start.",
            RuntimeWarning,
            stacklevel=3,
        )


def _build_fused_stage(
    stages: Sequence[object],
    executor: Executor,
    ctx: Any,
    report_stats_interval: float,
) -> tuple[_SubprocessPipelineConfig, _SubprocessPipelinePool]:
    """Build the worker pool and replacement stage for one fusable run."""
    stripped = [_strip_executor(s) for s in stages]
    num_threads = max(
        1, sum(s._args.concurrency for s in stages if isinstance(s, PipeConfig))
    )
    sub_config: PipelineConfig[Any] = PipelineConfig(
        src=SourceConfig(
            []
        ),  # placeholder; the worker sets the real source per session
        pipes=stripped,  # pyre-ignore[6]
        sink=SinkConfig(_FUSED_SINK_BUFFER),
    )
    max_workers, user_initializer, user_initargs = _pool_params(executor)
    build_kwargs = {
        "num_threads": num_threads,
        "report_stats_interval": report_stats_interval,
    }
    initializer = partial(
        _worker_initializer, _get_global_id(), user_initializer, user_initargs
    )
    pool = _SubprocessPipelinePool(
        ctx, max_workers, sub_config, build_kwargs, initializer, ()
    )
    name = "subprocess_pipeline(" + "+".join(_stage_name(s) for s in stages) + ")"
    return _SubprocessPipelineConfig(name=name, handle=pool.make_handle()), pool


def _fuse_subprocess_stages(
    config: PipelineConfig[Any],
    *,
    mp_context: str | None = None,
    report_stats_interval: float = -1,
) -> tuple[PipelineConfig[Any], list[_SubprocessPipelinePool]]:
    """Replace fusable runs of pool-pipe stages with single subprocess-pipeline stages.

    For each run found by :py:func:`_find_fusable_runs`, spawns a worker pool that runs the run
    as a nested pipeline and replaces the run with a :py:class:`_SubprocessPipelineConfig`. The
    spawned pools are returned so the caller can reap them at teardown. The input ``config`` is
    not mutated.

    Fusion is skipped for continuous-source pipelines (epoch-end markers would be mishandled by
    the fused source); such configs are returned unchanged.

    Args:
        config: The pipeline configuration to rewrite.
        mp_context: Multiprocessing start-method name, as accepted by
            :py:func:`multiprocessing.get_context`.
        report_stats_interval: Forwarded to the nested ``build_pipeline`` so per-stage stats are
            reported from inside the workers.

    Returns:
        A tuple ``(new_config, pools)``. ``pools`` is empty when no fusion applies.
    """
    if _has_continuous_source(config):
        return config, []
    runs = _find_fusable_runs(config.pipes)
    if not runs:
        return config, []

    ctx = mp.get_context(mp_context)
    _warn_fork_with_threads(ctx)

    pools: list[_SubprocessPipelinePool] = []
    by_start = {r.start: r for r in runs}
    pipes = list(config.pipes)
    new_pipes: list[object] = []
    i = 0
    while i < len(pipes):
        run = by_start.get(i)
        if run is None:
            new_pipes.append(pipes[i])
            i += 1
            continue
        fused, pool = _build_fused_stage(
            pipes[run.start : run.stop], run.executor, ctx, report_stats_interval
        )
        pools.append(pool)
        new_pipes.append(fused)
        i = run.stop
    return replace(config, pipes=new_pipes), pools  # pyre-ignore[6]
