# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


__all__ = [
    "_build_pipeline",
    "build_pipeline",
    "run_pipeline_in_subinterpreter",
    "run_pipeline_in_subprocess",
    "get_default_build_callback",
    "set_default_build_callback",
]

import logging
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import partial
from typing import Any, Generic, TypeVar

from spdl.pipeline._bg_task import (
    BackgroundTaskFactory,
    get_default_background_tasks,
)
from spdl.pipeline._components import (
    _build_pipeline_coro,
    _get_global_id,
    _set_global_id,
    AsyncQueue,
    StageInfo,
    TaskHook,
)
from spdl.pipeline._executor_proxy import _make_config_executors_picklable
from spdl.pipeline._fuse import _fuse_subprocess_stages
from spdl.pipeline._iter_utils import iterate_in_subinterpreter, iterate_in_subprocess
from spdl.pipeline._subprocess_pipeline_pool import _shutdown_pipeline_pools
from spdl.pipeline._subprocess_worker_pool import (
    _hoist_process_pools,
    _IterableWithPoolShutdown,
    _shutdown_pools,
)
from spdl.pipeline.defs import MergeConfig, PipelineConfig, SourceConfig

from ._pipeline import Pipeline

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")

_LG: logging.Logger = logging.getLogger(__name__)

_DEFAULT_BUILD_CALLBACK: Callable[[PipelineConfig[Any]], None] | None = None


def get_default_build_callback() -> Callable[[PipelineConfig[Any]], None] | None:
    """Get the default build callback function.

    Returns:
        The default build callback function or None if not set.
    """
    return _DEFAULT_BUILD_CALLBACK


def set_default_build_callback(
    callback: Callable[[PipelineConfig[Any]], None] | None,
) -> None:
    """Set the default build callback function.

    Args:
        callback: A callback function that takes a PipelineConfig object,
            or None to unset the callback.
    """
    global _DEFAULT_BUILD_CALLBACK
    _DEFAULT_BUILD_CALLBACK = callback


# The following is how we intend pipeline to behave. If the implementation
# is inconsistent with the following, it is a bug.
#
# 1. Successful execution.
#
#    Assumption: The consumer keeps consuming the data from the output queue.
#
#    Each stage completes without failure.
#    The source will pass EOF, and each stage receiving EOF will shut down,
#    then pass EOF.
#    EOF is not propagated to the output queue.
#    (It is sink's responsibility to filter out EOF.)
#
# 2. Failures at some stage. One or more stages fail.
#
#    Assumption: The consumer keeps consuming the data from the output queue.
#
#    Resolution:
#      We want to keep the tasks downstream to failed ones alive, so that they
#      can finish the ongoing works.
#
#    Actions:
#      - The failed stage must pass EOF to the next queue, so that
#        the downstream stages can exit cleanly.
#      - Passing EOF to the next queue can be blocked if the queue is full.
#      - For a graceful exit, the assumption of the continued consumption must be met.
#      - The stages upstream to the failure must be cancelled.
#
# 3. Cancelled: The pipeline execution is cancelled.
#
#    Assumption: The consumer stopped consuming the data from the output queue.
#
#    Actions:
#      - All the stages must be cancelled.
#        (If tasks are blocked on async await, cancelling should do).
#
# Following the above intention, each stage must pass EOF to the output queue
# unless the task was cancelled. See `_queue_stage_hook`.
#


################################################################################
# Coroutine execution logics
################################################################################


def _build_pipeline(
    pipeline_cfg: PipelineConfig[U],
    /,
    *,
    num_threads: int,
    max_failures: int | Fraction = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None = None,
    stage_id: int = 0,
    background_tasks: list[BackgroundTaskFactory] | None = None,
    use_thread_output_queue: bool = False,
    fuse_subprocess_stages: bool = False,
) -> Pipeline[U]:
    if _DEFAULT_BUILD_CALLBACK is not None:
        try:
            _DEFAULT_BUILD_CALLBACK(pipeline_cfg)
        except Exception:
            _LG.exception("Build callback failed.")

    pools: list[Any] = []
    if fuse_subprocess_stages:
        # Fuse consecutive same-pool stages so each run executes as one nested pipeline inside a
        # worker pool, eliminating the inter-stage IPC. The pools are owned by the returned
        # Pipeline and reaped when it stops.
        # stacklevel=4: _fuse_subprocess_stages -> _build_pipeline -> build_pipeline -> user.
        pipeline_cfg, pools = _fuse_subprocess_stages(
            pipeline_cfg, report_stats_interval=report_stats_interval, stacklevel=4
        )

    desc = repr(pipeline_cfg)

    _LG.debug("%s", desc)

    # Merge per-pipeline background tasks with defaults
    all_bg_tasks: list[BackgroundTaskFactory] = []
    default_bg = get_default_background_tasks()
    if default_bg:
        all_bg_tasks.extend(default_bg)
    if background_tasks:
        all_bg_tasks.extend(background_tasks)

    coro, queue = _build_pipeline_coro(
        pipeline_cfg,
        max_failures=max_failures,
        report_stats_interval=report_stats_interval,
        queue_class=queue_class,
        task_hook_factory=task_hook_factory,
        stage_id=stage_id,
        background_tasks=all_bg_tasks or None,
        use_thread_output_queue=use_thread_output_queue,
    )

    executor = ThreadPoolExecutor(
        max_workers=num_threads,
        thread_name_prefix="spdl_worker_thread_",
    )
    return Pipeline(coro, queue, executor, desc=desc, pools=pools)


def build_pipeline(
    pipeline_cfg: PipelineConfig[U],
    /,
    *,
    num_threads: int,
    max_failures: int | Fraction = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None = None,
    stage_id: int = 0,
    background_tasks: list[BackgroundTaskFactory] | None = None,
    use_thread_output_queue: bool = False,
    fuse_subprocess_stages: bool = False,
) -> Pipeline[U]:
    """Build a pipeline from the config.

    .. note::

       If environment variable ``SPDL_PIPELINE_DIAGNOSTIC_MODE=1`` is set, then this
       function builds a Pipeline in self-diagnostic mode. In self-diagnostic mode,
       the pipeline will call ``profile_pipeline`` function and benchmark each stage
       with different concurrency. Once the profiling is done, then the program exits.

    .. seealso::

       :ref:`Example: Pipeline definitions <example-pipeline-definitions>`
          Illustrates how to build a complex pipeline.

       :py:func:`profile_pipeline`
          A function to profile a Pipeline stage by stage.

    Args:
        pipeline_cfg: The definition of the pipeline to build.

        num_threads: The number of threads in the thread pool commonly used among
            Pipeline stages.

            .. note::

               If a stage in the pipeline has dedicated executor, that stage will
               use it.

        max_failures: The maximum number (int) or rate (Fraction) of failures each pipe
            stage can have before the pipeline is halted.
            When an int is provided, it specifies the maximum count of failures.
            Setting ``-1`` (default) disables it.
            When a Fraction is provided (e.g., Fraction(1, 10) for 10%), it specifies
            the maximum failure rate (failures / invocations).

        report_stats_interval: When provided, report the pipeline performance stats
            every given interval. Unit: [sec]

            This is only effective if there is no custom hook or custom AsyncQueue
            provided for stages. The argument is passed to
            :py:class:`TaskStatsHook` and :py:class:`StatsQueue`.

            If a custom stage hook is provided and stats report is needed,
            you can instantiate :py:class:`TaskStatsHook` and include
            it in the hooks provided to :py:meth:`PipelineBuilder.pipe`.

            Similarly if you are providing a custom :py:class:`AsyncQueue` class,
            you need to implement the same logic by your self.

        queue_class: If provided, override the queue class used to connect stages.
            Must be a class (not an instance) inherits :py:class:`AsyncQueue`.

        task_hook_factory: If provided, used to create task hook objects, given a
            name of the stage. If ``None``, a default hook,
            :py:class:`TaskStatsHook` is used.
            To disable hooks, provide a function that returns an empty list.

        stage_id: The index of the initial stage  used for logging.

        background_tasks: Optional list of :py:class:`BackgroundTaskFactory` callables.
            Each factory is called to create a :py:class:`BackgroundTask` whose
            :py:meth:`~BackgroundTask.run` method runs alongside the pipeline stages.
            Tasks are cancelled when the pipeline completes. Their errors are logged
            but do not cause the pipeline to fail.

        use_thread_output_queue: If ``True``, replace the sink's output queue with a
            :py:class:`queue.Queue`-backed queue for the final handoff from the
            background event loop to the foreground consumer thread. This bypasses
            ``asyncio.run_coroutine_threadsafe``, reducing per-batch latency from
            ~200-400us to ~10us. Default: ``False``.

        fuse_subprocess_stages: If ``True``, fuse runs of two or more adjacent pipe stages that
            share the same process-pool (or interpreter-pool) executor instance into a single
            stage that executes the run as one nested pipeline inside a worker pool. This
            eliminates the inter-stage IPC that otherwise round-trips data back to this process
            between each stage (so intermediate values need not be picklable), while each fused
            stage keeps its own ``concurrency`` and per-stage stats. Only adjacent pool stages
            fuse: an ``aggregate``/``disaggregate`` between two pool stages is not fused (it
            keeps its main-process batching) and splits them into separate runs. Has no effect
            on continuous-source pipelines (a :py:class:`RuntimeWarning` is emitted if fusable
            stages are otherwise present). Default: ``False``.

            .. versionadded:: 0.6.0
               The ``fuse_subprocess_stages`` argument.
    """
    from . import _profile

    if _profile.is_diagnostic_mode_enabled():
        return _profile._build_pipeline_diagnostic_mode(pipeline_cfg)

    return _build_pipeline(
        pipeline_cfg,
        num_threads=num_threads,
        max_failures=max_failures,
        report_stats_interval=report_stats_interval,
        queue_class=queue_class,
        task_hook_factory=task_hook_factory,
        stage_id=stage_id,
        background_tasks=background_tasks,
        use_thread_output_queue=use_thread_output_queue,
        fuse_subprocess_stages=fuse_subprocess_stages,
    )


################################################################################
# run in subprocess
################################################################################


def _has_continuous_source(config: PipelineConfig[Any]) -> bool:
    """Check if any source in the pipeline config is continuous."""
    src = config.src
    if isinstance(src, SourceConfig):
        return src.continuous
    if isinstance(src, MergeConfig):
        return any(_has_continuous_source(plc) for plc in src.pipeline_configs)
    return False


class _Wrapper(Generic[U]):
    def __init__(
        self,
        config: PipelineConfig[U],
        num_threads: int,
        max_failures: int | Fraction,
        report_stats_interval: float,
        queue_class: type[AsyncQueue] | None,
        task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None = None,
        background_tasks: list[BackgroundTaskFactory] | None = None,
        use_thread_output_queue: bool = False,
    ) -> None:
        self.config = config
        self.num_threads = num_threads
        self.max_failures = max_failures
        self.report_stats_interval = report_stats_interval
        self.queue_class = queue_class
        self.task_hook_factory = task_hook_factory
        self.background_tasks = background_tasks
        self.use_thread_output_queue = use_thread_output_queue
        self._pipeline: Pipeline[U] | None = None
        if _has_continuous_source(config):
            self._pipeline = self._build()
            self._pipeline.start()

    def _build(self) -> Pipeline[U]:
        return build_pipeline(
            self.config,
            num_threads=self.num_threads,
            max_failures=self.max_failures,
            report_stats_interval=self.report_stats_interval,
            queue_class=self.queue_class,
            task_hook_factory=self.task_hook_factory,
            background_tasks=self.background_tasks,
            use_thread_output_queue=self.use_thread_output_queue,
        )

    def __iter__(self) -> Iterator[U]:
        if (pipeline := self._pipeline) is not None:
            # The pipeline is in continuous mode. It is already running.
            yield from pipeline
        else:
            pipeline = self._build()
            with pipeline.auto_stop():
                yield from pipeline


def _get_initializer(kwargs: Any) -> Sequence[Callable[[], None]]:
    initializer = [partial(_set_global_id, _get_global_id())]
    if "initializer" not in kwargs:
        return initializer

    init_ = kwargs.pop("initializer")
    if not isinstance(init_, Sequence):
        initializer.append(init_)
    else:
        initializer.extend(init_)
    return initializer


def run_pipeline_in_subprocess(
    config_or_builder: PipelineConfig[T],
    /,
    *,
    num_threads: int,
    max_failures: int | Fraction = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None = None,
    background_tasks: list[BackgroundTaskFactory] | None = None,
    use_thread_output_queue: bool = False,
    fuse_subprocess_stages: bool = False,
    **kwargs: Any,
) -> Iterable[T]:
    """Run the given Pipeline in a subprocess, and iterate on the result.

    The returned :py:class:`Iterable` supports multiple iterations. The
    subprocess is created once and reused — each call to ``iter()`` (or
    ``for ... in``) builds a fresh :py:class:`Pipeline` inside the same
    subprocess without spawning a new process. This avoids the overhead
    of subprocess creation (fork/spawn, initializer execution, and
    pickling) on every iteration.

    For multi-epoch training, create the iterable once before the epoch
    loop and iterate it each epoch::

        src = run_pipeline_in_subprocess(config, num_threads=4)
        for epoch in range(num_epochs):
            for batch in src:
                train(batch)

    If the given config has a continuous source (built with
    :py:meth:`PipelineBuilder.add_source(..., continuous=True)
    <spdl.pipeline.PipelineBuilder.add_source>`), the pipeline is built and
    started once inside the subprocess and then **reused** across epochs: it
    keeps running in the background between epochs instead of being torn down
    and rebuilt. This keeps the prefetch buffer warm and removes the per-epoch
    rebuild gap, which matters most when the training step is short (e.g. small
    models)::

        config = (
            PipelineBuilder()
            .add_source(dataset, continuous=True)
            .pipe(load, concurrency=4)
            .aggregate(batch_size)
            .add_sink(buffer_size=3)
            .get_config()
        )
        src = run_pipeline_in_subprocess(config, num_threads=4)
        for epoch in range(num_epochs):
            for batch in src:  # one epoch; subprocess pipeline stays warm
                train(batch)

    Each iteration yields exactly one epoch (the epoch boundary is handled
    internally), so the loop above iterates one epoch per ``for`` pass just as
    in the non-continuous case. The continuous setting only changes *how* the
    subprocess manages the pipeline between epochs. To run continuous
    GPU-transfer stages in the main process, an outer pipeline can wrap ``src``
    with ``add_source(src, continuous=True)`` (see the MTP pattern in the
    parallelism guide).

    .. note::

       Pipe stages configured with a stdlib
       :py:class:`concurrent.futures.ThreadPoolExecutor`,
       :py:class:`concurrent.futures.ProcessPoolExecutor` or (on Python 3.14+)
       ``InterpreterPoolExecutor`` are explicitly supported, even though these executors are
       not picklable.

       Such an executor must be **freshly constructed** — handed over without any work
       submitted yet — because its workers are (re)created as part of running the pipeline in
       the subprocess (the whole point of moving execution there). Passing one that has already
       spawned workers (i.e. been used) lifts it mid-lifecycle and raises :py:exc:`ValueError`.

       - ``ThreadPoolExecutor`` / ``InterpreterPoolExecutor``: their constructor arguments are
         serialized and an equivalent executor (same type, same ``max_workers``) is
         reconstructed inside the subprocess. Their workers (threads / subinterpreters) live
         inside the subprocess and are cleaned up when it exits.

       - ``ProcessPoolExecutor``: its worker processes are spawned in the **main** process (as
         children of the main process, not grandchildren via the pipeline subprocess) and the
         executor is replaced with a queue-backed proxy that the subprocess submits to. This
         keeps ownership of the worker processes in the main process, which reaps them when the
         returned iterable is garbage-collected, so they cannot be orphaned if the pipeline
         subprocess is force-killed. The worker count (``max_workers``) and
         ``initializer``/``initargs`` are preserved; other construction options (e.g.
         ``mp_context``, ``max_tasks_per_child``) are not honored.

         .. warning::

            Those worker processes are spawned with the start method named by ``mp_context``
            (default: the platform default start method — ``fork`` on Linux through Python
            3.13, ``forkserver`` from Python 3.14). Spawning them with ``fork`` from a process
            that already has other live threads can deadlock — ``fork``
            copies only the calling thread, so a lock held by another thread is never released
            in the child. If you attach a ``ProcessPoolExecutor`` and the main process is (or
            may become) multi-threaded, pass ``mp_context="spawn"`` or ``"forkserver"``, or
            build the pipeline before any other threads start. A :py:exc:`RuntimeWarning` is
            emitted when this risky combination is detected.

       SPDL's own :py:class:`~spdl.pipeline.PriorityThreadPoolExecutor` and related pool
       executors are already picklable and pass through unchanged.

    Args:
        config_or_builder: The definition of :py:class:`Pipeline`. Can be either a
            :py:class:`PipelineConfig` or :py:class:`PipelineBuilder`.

            .. warning::

               The support for :py:class:`PipelineBuilder` is deprecated, and will be removed in
               the future. Please call `get_config()` method and pass the config object.

        num_threads,max_failures,report_stats_interval,queue_class,task_hook_factory,background_tasks:
            Passed to :py:func:`build_pipeline`.
        fuse_subprocess_stages: If ``True``, fuse runs of two or more adjacent pipe stages that
            share the same process-pool (or interpreter-pool) executor instance into a single
            stage that runs the run as one nested pipeline inside a worker pool. The worker
            processes are spawned in (and owned by) the main process, exactly like a hoisted
            ``ProcessPoolExecutor``; the pipeline subprocess drives them through a queue handle.
            This removes the per-stage round-trip between the pipeline subprocess and the pool
            workers (so intermediate values need not be picklable). Has no effect on
            continuous-source pipelines (a :py:class:`RuntimeWarning` is emitted if fusable
            stages are otherwise present). Default: ``False``.

            .. versionadded:: 0.6.0
               The ``fuse_subprocess_stages`` argument.
        kwargs: Passed to :py:func:`iterate_in_subprocess`.

    Yields:
        The results yielded from the pipeline.

    .. versionchanged:: 0.6.0
       Pipe stages configured with a stdlib
       :py:class:`~concurrent.futures.ThreadPoolExecutor`,
       :py:class:`~concurrent.futures.ProcessPoolExecutor`, or (on Python 3.14+)
       ``InterpreterPoolExecutor`` are now supported. Thread/interpreter pools are
       reconstructed inside the subprocess; a ``ProcessPoolExecutor``'s worker processes are
       spawned in (and owned by) the main process.

    .. seealso::

       - :py:func:`iterate_in_subprocess` implements the logic for manipulating an iterable
         in a subprocess.
       - :ref:`parallelism-performance` for the context in which this function was created.
    """
    if not isinstance(config_or_builder, PipelineConfig):
        warnings.warn(
            "Passing a `PipelineBuilder` object directly to `run_pipeline_in_subprocess` is "
            "now deprecated. Please call `get_config()` method and pass the config object.",
            stacklevel=2,
        )

    config = (
        config_or_builder
        if isinstance(config_or_builder, PipelineConfig)
        else config_or_builder.get_config()  # pyre-ignore[16]
    )

    # Fuse runs of same-pool stages into nested-pipeline stages first, so a fused run executes
    # as one pipeline inside the (main-owned) worker pool instead of round-tripping per stage.
    # Runs before hoisting so only the *non-fused* ProcessPoolExecutor stages remain to be
    # hoisted. The fused stage's queue handle is picklable and rides into the pipeline
    # subprocess with the config, where the bridge stage drives the main-owned workers.
    fuse_pools: list[Any] = []
    if fuse_subprocess_stages:
        # stacklevel=3: _fuse_subprocess_stages -> run_pipeline_in_subprocess -> user.
        config, fuse_pools = _fuse_subprocess_stages(
            config,
            mp_context=kwargs.get("mp_context"),
            report_stats_interval=report_stats_interval,
            stacklevel=3,
        )

    # Spawn workers for any stdlib ``ProcessPoolExecutor`` in the main process (as children of
    # main, not grandchildren via the pipeline subprocess), then replace the executor with a
    # queue-backed proxy that the subprocess submits to. The remaining stdlib executors
    # (Thread/Interpreter pools, whose workers live inside the subprocess and die with it) are
    # made picklable via lazy-reconstruction proxies.
    config, pools = _hoist_process_pools(config, kwargs.get("mp_context"))
    try:
        config = _make_config_executors_picklable(config)

        initializer = _get_initializer(kwargs)
        iterable = iterate_in_subprocess(
            fn=partial(
                _Wrapper,
                config=config,
                num_threads=num_threads,
                max_failures=max_failures,
                report_stats_interval=report_stats_interval,
                queue_class=queue_class,
                task_hook_factory=task_hook_factory,
                background_tasks=background_tasks,
                use_thread_output_queue=use_thread_output_queue,
            ),
            initializer=initializer,
            **kwargs,
        )
    except BaseException:
        # If anything between the hoist and the iterable raises (e.g.
        # ``iterate_in_subprocess`` fails to start the subprocess), the iterable is never
        # returned to the caller — reap the pools here so their worker processes and pipe
        # fds do not leak.
        _shutdown_pools(pools)
        _shutdown_pipeline_pools(fuse_pools)
        raise
    # The hoisted ``ProcessPoolExecutor`` workers and the fused-stage worker pools are all
    # owned by this (the main) process and must outlive every iteration (they are reused across
    # epochs). Wrap the iterable so they are reaped via the wrapper's finalizer when it is torn
    # down — after the epoch loop, not at the end of each iteration. Both kinds share queues
    # with the pipeline subprocess, so the wrapper joins that subprocess before reaping them.
    all_pools: list[Any] = [*pools, *fuse_pools]
    if all_pools:
        iterable = _IterableWithPoolShutdown(iterable, all_pools)
    return iterable


################################################################################
# run in subinterpreter
################################################################################


def run_pipeline_in_subinterpreter(
    config: PipelineConfig[T],
    /,
    *,
    num_threads: int,
    max_failures: int | Fraction = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None = None,
    background_tasks: list[BackgroundTaskFactory] | None = None,
    use_thread_output_queue: bool = False,
    **kwargs: Any,
) -> Iterable[T]:
    """**[Experimental]** Run the given Pipeline in a subinterpreter, and iterate on the result.

    The returned :py:class:`Iterable` supports multiple iterations. The
    subinterpreter is created once and reused — each call to ``iter()``
    (or ``for ... in``) builds a fresh :py:class:`Pipeline` inside the
    same subinterpreter without creating a new one. This avoids the
    overhead of repeated subinterpreter creation and initializer
    execution on every iteration.

    For multi-epoch training, create the iterable once before the epoch
    loop and iterate it each epoch::

        src = run_pipeline_in_subinterpreter(config, num_threads=4)
        for epoch in range(num_epochs):
            for batch in src:
                train(batch)

    Args:
        config: The definition of :py:class:`Pipeline`.
        num_threads,max_failures,report_stats_interval,queue_class,task_hook_factory,background_tasks:
            Passed to :py:func:`build_pipeline`.
        kwargs: Passed to :py:func:`iterate_in_subinterpreter`.

    Yields:
        The results yielded from the pipeline.

    .. seealso::

       - :py:func:`iterate_in_subinterpreter` implements the logic for manipulating an iterable
         in a subinterpreter.
       - :ref:`parallelism-performance` for the context in which this function was created.
    """
    initializer = _get_initializer(kwargs)
    return iterate_in_subinterpreter(
        fn=partial(
            _Wrapper,
            config=config,
            num_threads=num_threads,
            max_failures=max_failures,
            report_stats_interval=report_stats_interval,
            queue_class=queue_class,
            task_hook_factory=task_hook_factory,
            background_tasks=background_tasks,
            use_thread_output_queue=use_thread_output_queue,
        ),
        initializer=initializer,
        **kwargs,
    )
