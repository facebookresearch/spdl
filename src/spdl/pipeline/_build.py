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
]

import logging
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Generic, TypeVar

from spdl.pipeline._components import (
    _build_pipeline_coro,
    _get_global_id,
    _set_global_id,
    AsyncQueue,
    TaskHook,
)
from spdl.pipeline._iter_utils import iterate_in_subinterpreter, iterate_in_subprocess
from spdl.pipeline.defs import PipelineConfig

from ._pipeline import Pipeline

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")

_LG: logging.Logger = logging.getLogger(__name__)


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
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    stage_id: int = 0,
) -> Pipeline[U]:
    desc = repr(pipeline_cfg)

    _LG.debug("%s", desc)

    coro, queue = _build_pipeline_coro(
        pipeline_cfg,
        max_failures=max_failures,
        report_stats_interval=report_stats_interval,
        queue_class=queue_class,
        task_hook_factory=task_hook_factory,
        stage_id=stage_id,
    )

    executor = ThreadPoolExecutor(
        max_workers=num_threads,
        thread_name_prefix="spdl_worker_thread_",
    )
    return Pipeline(coro, queue, executor, desc=desc)


def build_pipeline(
    pipeline_cfg: PipelineConfig[U],
    /,
    *,
    num_threads: int,
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    stage_id: int = 0,
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

        max_failures: The maximum number of failures each pipe stage can have before
            the pipeline is halted. Setting ``-1`` (default) disables it.

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
    )


################################################################################
# run in subprocess
################################################################################


class _Wrapper(Generic[U]):
    def __init__(
        self,
        config: PipelineConfig[U],
        num_threads: int,
        max_failures: int,
        report_stats_interval: float,
        queue_class: type[AsyncQueue] | None,
        task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    ) -> None:
        self.config = config
        self.num_threads = num_threads
        self.max_failures = max_failures
        self.report_stats_interval = report_stats_interval
        self.queue_class = queue_class
        self.task_hook_factory = task_hook_factory

    def __iter__(self) -> Iterator[U]:
        pipeline = build_pipeline(
            self.config,
            num_threads=self.num_threads,
            max_failures=self.max_failures,
            report_stats_interval=self.report_stats_interval,
            queue_class=self.queue_class,
            task_hook_factory=self.task_hook_factory,
        )
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
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    **kwargs: Any,
) -> Iterable[T]:
    """Run the given Pipeline in a subprocess, and iterate on the result.

    Args:
        config_or_builder: The definition of :py:class:`Pipeline`. Can be either a
            :py:class:`PipelineConfig` or :py:class:`PipelineBuilder`.

            .. warning::

               The support for :py:class:`PipelineBuilder` is deprecated, and will be removed in
               the future. Please call `get_config()` method and pass the config object.

        num_threads,max_failures,report_stats_interval,queue_class,task_hook_factory:
            Passed to :py:func:`build_pipeline`.
        kwargs: Passed to :py:func:`iterate_in_subprocess`.

    Yields:
        The results yielded from the pipeline.

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

    initializer = _get_initializer(kwargs)
    return iterate_in_subprocess(
        fn=partial(
            _Wrapper,
            config=config,
            num_threads=num_threads,
            max_failures=max_failures,
            report_stats_interval=report_stats_interval,
            queue_class=queue_class,
            task_hook_factory=task_hook_factory,
        ),
        initializer=initializer,
        **kwargs,
    )


################################################################################
# run in subinterpreter
################################################################################


def run_pipeline_in_subinterpreter(
    config: PipelineConfig[T],
    /,
    *,
    num_threads: int,
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    **kwargs: Any,
) -> Iterable[T]:
    """**[Experimental]** Run the given Pipeline in a subinterpreter, and iterate on the result.

    Args:
        config: The definition of :py:class:`Pipeline`.
        num_threads,max_failures,report_stats_interval,queue_class,task_hook_factory:
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
        ),
        initializer=initializer,
        **kwargs,
    )
