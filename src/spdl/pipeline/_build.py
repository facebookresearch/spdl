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
from spdl.pipeline._fuse import (
    _fuse_marked_regions,
    _fuse_subprocess_stages,
    _strip_async_executor_tags,
)
from spdl.pipeline._iter_utils import iterate_in_subinterpreter, iterate_in_subprocess
from spdl.pipeline._random_seed import _capture_rng_initializers
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
    # Both fusion passes eagerly spawn worker pools. Reap them together on failure: each pass
    # only reaps its own pools if it raises, so without this a failure in the second pass would
    # leak the pools the first already spawned -- this half-built pipeline is never returned to
    # the caller to be stopped.
    try:
        # Honor explicit `.to()` region markers first. This is a no-op when the config has no
        # markers, so it is always safe to run and independent of `fuse_subprocess_stages`.
        # stacklevel=4: _fuse_marked_regions -> _build_pipeline -> build_pipeline -> user.
        pipeline_cfg, region_pools = _fuse_marked_regions(
            pipeline_cfg, report_stats_interval=report_stats_interval, stacklevel=4
        )
        pools.extend(region_pools)
        if fuse_subprocess_stages:
            # Fuse consecutive same-pool stages so each run executes as one nested pipeline
            # inside a worker pool, eliminating the inter-stage IPC. The pools are owned by the
            # returned Pipeline and reaped when it stops.
            # stacklevel=4: _fuse_subprocess_stages -> _build_pipeline -> build_pipeline -> user.
            pipeline_cfg, id_pools = _fuse_subprocess_stages(
                pipeline_cfg, report_stats_interval=report_stats_interval, stacklevel=4
            )
            pools.extend(id_pools)
    except BaseException:
        _shutdown_pipeline_pools(pools)
        raise

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
            stage keeps its own ``concurrency`` and per-stage stats. A ``path_variants`` stage
            whose branches all use the same pool executor is fused too — the whole routing
            construct (router and branches) moves into the worker — and fuses on its own even
            when it is the only such stage. An ``aggregate``/``disaggregate`` between two pool
            stages is not fused (it keeps its main-process batching) and splits them into
            separate runs. An async op joins a fused run when tagged with the same executor as
            its neighbours (see :py:meth:`~spdl.pipeline.PipelineBuilder.pipe`), running on the
            worker's own event loop. Continuous sources are supported (the fused worker
            sub-pipelines stay warm across epochs and epoch boundaries are propagated across the
            pool). Default: ``False``.

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


def _get_initializer(
    kwargs: Any, *, shareable_rng_only: bool
) -> Sequence[Callable[[], None]]:
    initializer: list[Callable[[], None]] = [
        partial(_set_global_id, _get_global_id()),
    ]
    # Copy the main process' current global RNG state into the worker so that
    # RNG-dependent work inside the pipeline (e.g. augmentation in MTP mode)
    # continues from the state the user set, regardless of the start method.
    # Captured here in the main process; restored in the worker before iteration.
    # For subinterpreters only the stdlib ``random`` state is copied: initializers
    # cross the interpreter boundary as call arguments and so must be shareable,
    # which the captured numpy/torch state objects are not.
    initializer.extend(_capture_rng_initializers(shareable_only=shareable_rng_only))
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

       **Random number generators.** The current global RNG state of the main
       process is copied into the worker subprocess before iterating, so
       RNG-dependent work inside the pipeline (e.g. data augmentation) continues
       from exactly the state the main process was in — regardless of the
       multiprocessing start method (``fork`` / ``spawn`` / ``forkserver``). No
       opt-in is required.

       In particular, if you seed the global RNGs in the main process before
       creating the pipeline (``random.seed(k)``, ``numpy.random.seed(k)``,
       ``torch.manual_seed(k)``), the worker inherits that seeding seamlessly, so
       draws are reproducible across program runs just as they would be in-process.

       This copies the **global** generators only:

       - :py:mod:`random` (standard library),
       - NumPy's **legacy** global RNG (:py:func:`numpy.random.rand`,
         :py:func:`numpy.random.choice`, :py:func:`numpy.random.shuffle`, …), and
       - PyTorch's CPU generator.

       NumPy and PyTorch state is copied only when the program has already
       imported them; SPDL never imports them on your behalf.

       .. warning::

          The following sources are **not** copied, because the library cannot
          reach them:

          - A :py:class:`numpy.random.Generator` created with
            :py:func:`numpy.random.default_rng` (the modern NumPy API) is an
            independent object, not part of the legacy global state. If your code
            holds one, seed it explicitly. SPDL's own samplers already store an
            explicit seed, so they are reproducible in every mode.
          - PyTorch **CUDA** device RNG state (only the CPU generator is copied).
          - Hash randomization (``PYTHONHASHSEED``), which affects ``set``
            iteration order, is fixed at interpreter start — pin it in the launch
            environment if it matters.
          - Cryptographic / entropy sources (:py:func:`os.urandom`,
            :py:mod:`secrets`, :py:func:`uuid.uuid4`) are intentionally left
            untouched.

          Note also that task *completion order* (``output_order="completion"``)
          is independent of RNG state and can still differ between execution
          modes.

       .. versionchanged:: 0.6.0
          The worker subprocess now inherits the main process' global RNG state.

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
            workers (so intermediate values need not be picklable). A ``path_variants`` stage
            whose branches all use the same pool executor is fused too (router and branches move
            into the worker). An async op joins a fused run when tagged with the same executor as
            its neighbours (see :py:meth:`~spdl.pipeline.PipelineBuilder.pipe`), running on the
            worker's own event loop. Continuous sources are supported. Default: ``False``.

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

    # Every pass below eagerly spawns worker pools, so they all run inside one try/except:
    # ``_fuse_marked_regions`` spawns ``fuse_pools`` up front, and if any *later* pass
    # (``_fuse_subprocess_stages``, ``_hoist_process_pools``) or the iterable creation
    # raises, both ``fuse_pools`` and the hoisted ``pools`` must be reaped -- this half-built
    # iterable is never returned to the caller to be stopped. Mirrors ``_build_pipeline``.
    fuse_pools: list[Any] = []
    pools: list[Any] = []
    try:
        # Fuse each `.to()` region into a nested-pipeline stage here, in the main process (a
        # no-op when the config has no markers), so its worker pool is main-owned -- spawned
        # here, not inside the daemonic pipeline subprocess (a daemon cannot have children),
        # and not orphaned if that subprocess is force-killed. Runs before hoisting so only
        # the *non-region* ProcessPoolExecutor stages remain to be hoisted. The fused stage's
        # queue handle is picklable and rides into the subprocess with the config (where the
        # inner build sees no markers and is a no-op), and the bridge stage there drives the
        # main-owned workers.
        # stacklevel=3: _fuse_marked_regions -> run_pipeline_in_subprocess -> user.
        config, region_pools = _fuse_marked_regions(
            config,
            report_stats_interval=report_stats_interval,
            stacklevel=3,
        )
        fuse_pools.extend(region_pools)
        # Also fuse runs of same-pool stages tagged with an identical executor into one nested
        # pipeline inside a worker pool, eliminating the inter-stage IPC (before hoisting, so
        # only unfused ProcessPoolExecutor stages remain).
        if fuse_subprocess_stages:
            # stacklevel=3: _fuse_subprocess_stages -> run_pipeline_in_subprocess -> user.
            config, id_pools = _fuse_subprocess_stages(
                config,
                mp_context=kwargs.get("mp_context"),
                report_stats_interval=report_stats_interval,
                stacklevel=3,
            )
            fuse_pools.extend(id_pools)

        # Clear executor tags left on any unfused async op: they are subprocess fusion-group
        # hints, not real pools, and the executor-hoisting/pickling passes below are op-agnostic
        # -- an async op's process-pool tag would otherwise spawn an idle pool it never uses.
        config = _strip_async_executor_tags(config)

        # Spawn workers for any stdlib ``ProcessPoolExecutor`` in the main process (as children
        # of main, not grandchildren via the pipeline subprocess), then replace the executor
        # with a queue-backed proxy that the subprocess submits to. The remaining stdlib
        # executors (Thread/Interpreter pools, whose workers live inside the subprocess and die
        # with it) are made picklable via lazy-reconstruction proxies.
        config, hoisted_pools = _hoist_process_pools(config, kwargs.get("mp_context"))
        pools.extend(hoisted_pools)

        config = _make_config_executors_picklable(config)

        initializer = _get_initializer(kwargs, shareable_rng_only=False)
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
        # Any eager-spawn pass above (region fusion, identity fusion, hoisting) or the iterable
        # creation failed; the iterable is never returned to the caller, so reap the region and
        # hoisted pools here to avoid leaking their worker processes and pipe fds.
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

    .. note::

       **Random number generators.** Only the stdlib :py:mod:`random` global
       state is copied from the main interpreter into the subinterpreter; unlike
       :py:func:`run_pipeline_in_subprocess`, the **NumPy** and **PyTorch** global
       RNG state is **not**. Two interpreter limitations force this: the restore
       initializers cross the boundary as call arguments and so must be shareable
       across interpreters, which the captured NumPy / PyTorch state objects are
       not; and NumPy and PyTorch cannot be imported in a subinterpreter at all,
       so a pipeline running here cannot use them regardless. For any NumPy /
       PyTorch randomness, seed it from within ``config`` — e.g. via an SPDL
       sampler that stores an explicit seed
       (:py:class:`~spdl.source.DistributedRandomSampler`) — rather than depending
       on the main interpreter's global RNG state.

       .. versionchanged:: 0.6.0
          The subinterpreter now inherits the main interpreter's stdlib
          :py:mod:`random` global state.

    .. seealso::

       - :py:func:`iterate_in_subinterpreter` implements the logic for manipulating an iterable
         in a subinterpreter.
       - :ref:`parallelism-performance` for the context in which this function was created.
    """
    initializer = _get_initializer(kwargs, shareable_rng_only=True)
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
