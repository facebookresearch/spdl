# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Make a :py:class:`~spdl.pipeline.defs.PipelineConfig` with stdlib executors picklable.

The stdlib thread-based ``concurrent.futures`` executors
(:py:class:`~concurrent.futures.ThreadPoolExecutor` and, on Python 3.14+,
``InterpreterPoolExecutor``) are not picklable, so a :py:class:`PipelineConfig` that attaches
one of them to a pipe stage cannot be moved to a subprocess as-is.

For SPDL's purpose, what matters is that an executor of the same type with the same capacity
(``max_workers``) is recreated in the subprocess. So rather than moving the live executor, we
perform surgery on the config: each non-picklable stdlib executor is replaced with an
:py:class:`_ExecutorProxy` that records the executor's type and constructor arguments and
lazily reconstructs an equivalent executor on first use inside the subprocess. Their workers
(threads / subinterpreters) live inside the subprocess and are cleaned up when it exits.

``ProcessPoolExecutor`` is handled separately — its workers are hoisted into the main process;
see :py:mod:`spdl.pipeline._subprocess_worker_pool`.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from spdl.pipeline.defs import (
    MergeConfig,
    PathVariantsConfig,
    PipeConfig,
    PipelineConfig,
)

__all__ = [
    "_ensure_executor_unused",
    "_make_config_executors_picklable",
    "_rewrite_config_executors",
]


class _ExecutorProxy(Executor):
    """Picklable, lazily-constructed stand-in for a non-picklable stdlib executor.

    Records the executor's concrete type and constructor arguments instead of the live
    executor. The real executor is created lazily, on first use, in whichever process holds
    the proxy. This makes the proxy work uniformly regardless of the ``multiprocessing`` start
    method:

    - With ``spawn``/``forkserver``, the proxy is pickled (only the type and kwargs travel)
      and the real executor is built on first use in the subprocess.
    - With ``fork``, the proxy object is inherited as-is and the real executor is built on
      first use in the subprocess.

    The parent process never builds the pipeline, so the proxy never instantiates the real
    executor there (no stray threads/processes are created in the parent).
    """

    def __init__(self, executor_class: type[Executor], kwargs: dict[str, Any]) -> None:
        self._executor_class = executor_class
        self._kwargs = kwargs
        self._executor: Executor | None = None
        self._shutdown = False
        self._lock = threading.Lock()

    def _ensure_executor(self) -> Executor:
        # Double-checked locking: concurrent ``submit`` calls must build exactly one executor,
        # otherwise the extra executors (and their threads) leak unshut down.
        if self._executor is None:
            with self._lock:
                if self._shutdown:
                    # Honor the Executor contract: no new work after shutdown. Without this,
                    # a ``submit`` after ``shutdown`` on a never-built proxy would silently
                    # construct a fresh executor instead of raising.
                    raise RuntimeError("cannot schedule new futures after shutdown")
                if self._executor is None:
                    self._executor = self._executor_class(**self._kwargs)
        return self._executor

    def submit(  # pyre-ignore[14]
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        # If the executor was already built and then shut down, the underlying executor raises
        # on submit; the ``_shutdown`` guard in ``_ensure_executor`` covers the never-built
        # case.
        return self._ensure_executor().submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._lock:
            self._shutdown = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __getstate__(self) -> dict[str, Any]:
        # Never carry the live executor across the pickle boundary.
        return {"executor_class": self._executor_class, "kwargs": self._kwargs}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._executor_class = state["executor_class"]
        self._kwargs = state["kwargs"]
        self._executor = None
        self._shutdown = False
        self._lock = threading.Lock()


def _thread_pool_initializer(executor: Any) -> tuple[Any, tuple[Any, ...]]:
    # Python < 3.14 stores ``initializer``/``initargs`` directly on the executor. Python 3.14+
    # captures them inside a worker-context factory (``_create_worker_context``) instead, so
    # recover them by building a context (construction is side-effect-free: it just stashes the
    # values, without spawning any worker threads).
    if hasattr(executor, "_initializer"):
        return executor._initializer, executor._initargs
    factory = getattr(executor, "_create_worker_context", None)
    if factory is not None:
        ctx = factory()
        return getattr(ctx, "initializer", None), getattr(ctx, "initargs", ())
    return None, ()


def _thread_pool_kwargs(executor: Any) -> dict[str, Any]:
    initializer, initargs = _thread_pool_initializer(executor)
    return {
        "max_workers": executor._max_workers,
        "thread_name_prefix": executor._thread_name_prefix,
        "initializer": initializer,
        "initargs": initargs,
    }


def _interpreter_pool_initializer(executor: Any) -> tuple[Any, tuple[Any, ...]]:
    # InterpreterPoolExecutor does not keep ``initializer``/``initargs`` as attributes; it
    # stores them as ``initdata`` -- a ``(fn, args, kwargs)`` tuple, or ``None`` when no
    # initializer was given -- on the worker context produced by its ``_create_worker_context``
    # factory. Building that context is side-effect-free (no interpreter is created until the
    # context is later initialized), so recover the original ``(initializer, initargs)`` here
    # rather than silently dropping them.
    factory = getattr(executor, "_create_worker_context", None)
    if factory is None:
        return None, ()
    initdata = getattr(factory(), "initdata", None)
    if not initdata:
        return None, ()
    return initdata[0], initdata[1]


def _interpreter_pool_kwargs(executor: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_workers": executor._max_workers,
        "thread_name_prefix": executor._thread_name_prefix,
    }
    initializer, initargs = _interpreter_pool_initializer(executor)
    if initializer is not None:
        kwargs["initializer"] = initializer
        kwargs["initargs"] = initargs
    return kwargs


# Exact-type registry (not ``isinstance``) so that ``InterpreterPoolExecutor`` (a
# ``ThreadPoolExecutor`` subclass) and any user-defined subclasses are matched precisely or
# left untouched.
_EXECUTOR_ARG_EXTRACTORS: dict[type[Executor], Callable[[Any], dict[str, Any]]] = {
    ThreadPoolExecutor: _thread_pool_kwargs,
}

if sys.version_info >= (3, 14):
    from concurrent.futures.interpreter import (  # pyre-ignore[21]
        InterpreterPoolExecutor,
    )

    _EXECUTOR_ARG_EXTRACTORS[InterpreterPoolExecutor] = _interpreter_pool_kwargs


def _ensure_executor_unused(executor: Executor) -> None:
    """Reject a stdlib pool executor that has already been used.

    Moving a pipeline to a subprocess recreates the executor's workers as part of running it
    there (thread / interpreter pools are reconstructed in the subprocess; a
    :py:class:`~concurrent.futures.ProcessPoolExecutor`'s workers are hoisted into the main
    process). The executor must therefore be freshly constructed — one that has already
    spawned workers or had work submitted is being lifted mid-lifecycle, which is not the
    supported contract: the whole point is that execution, including the pool's worker startup,
    happens as part of the run. Construct the executor and hand it over without using it first.
    """
    if (
        getattr(
            executor, "_threads", None
        )  # Thread/Interpreter pool: spawned worker threads
        or getattr(
            executor, "_processes", None
        )  # Process pool: spawned worker processes
        or getattr(executor, "_queue_count", 0)  # Process pool: work already submitted
    ):
        raise ValueError(
            "run_pipeline_in_subprocess() requires a freshly constructed executor with no "
            "work submitted yet: its workers are (re)created when the pipeline runs in the "
            "subprocess. Construct the executor and pass it without using it first."
        )


def _maybe_proxy(executor: Executor | None) -> Executor | _ExecutorProxy | None:
    """Replace a non-picklable stdlib executor with a picklable proxy.

    Executors of other types (e.g. SPDL's ``Priority*PoolExecutor``, which are already
    picklable) and ``None`` are returned unchanged.
    """
    if executor is None:
        return None
    extractor = _EXECUTOR_ARG_EXTRACTORS.get(type(executor))
    if extractor is None:
        return executor
    _ensure_executor_unused(executor)
    return _ExecutorProxy(type(executor), extractor(executor))


def _rewrite_pipe(pipe: Any, convert: Callable[[Any], Any]) -> Any:
    if isinstance(pipe, PipeConfig):
        executor = pipe._args.executor
        new_executor = convert(executor)
        if new_executor is executor:
            return pipe
        return replace(pipe, _args=replace(pipe._args, executor=new_executor))
    if isinstance(pipe, PathVariantsConfig):
        new_paths = tuple(
            tuple(_rewrite_pipe(p, convert) for p in path) for path in pipe.paths
        )
        # Identity check (mirroring the MergeConfig branch in _rewrite_config_executors)
        # rather than ``==`` so a future value-based ``__eq__`` on a nested config element
        # cannot mask an actual rewrite.
        changed = any(
            np is not op
            for new_path, old_path in zip(new_paths, pipe.paths)
            for np, op in zip(new_path, old_path)
        )
        if not changed:
            return pipe
        return replace(pipe, paths=new_paths)
    return pipe


def _rewrite_config_executors(
    config: PipelineConfig[Any],
    convert: Callable[[Any], Any],
) -> PipelineConfig[Any]:
    """Return a copy of ``config`` with each pipe executor passed through ``convert``.

    Walks the config — including nested :py:class:`PathVariantsConfig` paths and
    :py:class:`MergeConfig` sub-configs — and replaces each pipe's executor with
    ``convert(executor)``. ``convert`` receives the executor (or ``None``) and must return the
    replacement (returning the same object signals "no change"). The input ``config`` is not
    mutated; a new config is returned (and shared unchanged where ``convert`` is a no-op).
    """
    new_src = config.src
    if isinstance(new_src, MergeConfig):
        new_configs = tuple(
            _rewrite_config_executors(plc, convert) for plc in new_src.pipeline_configs
        )
        # Identity check (mirroring the pipes branch) rather than ``!=`` so a future
        # value-based ``__eq__`` on a nested element cannot mask an actual rewrite.
        if any(nc is not oc for nc, oc in zip(new_configs, new_src.pipeline_configs)):
            # pyre-ignore[6]: MergeConfig annotates a 1-tuple but holds N configs.
            new_src = replace(new_src, pipeline_configs=new_configs)

    new_pipes = [_rewrite_pipe(p, convert) for p in config.pipes]

    src_changed = new_src is not config.src
    pipes_changed = any(np is not op for np, op in zip(new_pipes, config.pipes))
    if not src_changed and not pipes_changed:
        return config

    return replace(config, src=new_src, pipes=new_pipes)


def _make_config_executors_picklable(
    config: PipelineConfig[Any],
) -> PipelineConfig[Any]:
    """Return a copy of ``config`` with non-picklable stdlib executors replaced by proxies.

    Walks the config (including nested :py:class:`PathVariantsConfig` paths and
    :py:class:`MergeConfig` sub-configs) and swaps each stdlib
    :py:class:`~concurrent.futures.ThreadPoolExecutor` or (on Python 3.14+)
    ``InterpreterPoolExecutor`` attached to a pipe with an :py:class:`_ExecutorProxy`. The
    proxy reconstructs an equivalent executor (same type, same ``max_workers``) when unpickled
    in a subprocess.

    The input ``config`` is not mutated; a new config is returned (and shared unchanged where
    no surgery is needed).
    """
    return _rewrite_config_executors(config, _maybe_proxy)
