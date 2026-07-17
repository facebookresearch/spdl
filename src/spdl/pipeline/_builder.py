# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import sys
from collections.abc import AsyncIterable, Callable, Iterable, Sequence
from concurrent.futures import Executor
from fractions import Fraction
from typing import Generic, TypeVar

from spdl._internal import log_api_usage_once
from spdl.pipeline._components import AsyncQueue, StageInfo, TaskHook
from spdl.pipeline.defs import (
    _MainProcess,
    _PipeType,
    _TPipeInputs,
    Aggregate,
    AggregateConfig,
    Aggregator,
    Disaggregate,
    DisaggregateConfig,
    InterpreterPoolExecutorConfig,
    PathVariants,
    PathVariantsConfig,
    Pipe,
    PipeConfig,
    PipelineConfig,
    PlacementConfig,
    ProcessPoolExecutorConfig,
    SinkConfig,
    SourceConfig,
)

from ._build import build_pipeline
from ._pipeline import Pipeline

__all__ = [
    "PipelineBuilder",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
T_ = TypeVar("T_")
U_ = TypeVar("U_")


def _has_ordered_pipe(cfg: object) -> bool:
    """Whether ``cfg`` is (or, for path-variants, contains) an ``output_order="input"`` pipe.

    Recurses into :py:class:`~spdl.pipeline.defs.PathVariantsConfig` branches: those stages run
    inside the region worker too, so an input-ordered pipe nested in a branch is just as invalid
    as a top-level one (global input order cannot be preserved across the pool's workers).
    """
    if isinstance(cfg, PipeConfig):
        return cfg._type is _PipeType.OrderedPipe
    if isinstance(cfg, PathVariantsConfig):
        return any(_has_ordered_pipe(s) for path in cfg.paths for s in path)
    return False


def _validate_executor_regions(
    pipes: Sequence[
        PipeConfig
        | AggregateConfig
        | DisaggregateConfig
        | PathVariantsConfig
        | PlacementConfig
    ],
) -> None:
    """Validate the :py:meth:`PipelineBuilder.to` region markers in ``pipes``.

    Stateless scan (a pipeline starts on the main process). Rejects: a subinterpreter region on
    Python < 3.14; a stage using ``output_order="input"`` inside a region (including one nested
    in a ``path_variants`` branch, since order cannot be preserved across independent workers); an
    empty region (a ``.to(...)`` marker with no stages before the next marker/sink); and a region
    left open at the sink.
    """
    in_region = False
    region_has_stage = False
    for p in pipes:
        if isinstance(p, PlacementConfig):
            # A non-main region that opened but never received a stage does nothing; reject it
            # rather than silently dropping it (usually a stray or duplicated ``.to(...)``).
            # Adjacent *non-empty* regions with different targets remain valid.
            if in_region and not region_has_stage:
                raise ValueError(
                    "An empty `to(...)` region has no stages. Add stages before the next "
                    "`.to(...)`, or remove the redundant marker."
                )
            target = p.target
            in_region = not isinstance(target, _MainProcess)
            region_has_stage = False
            if isinstance(
                target, InterpreterPoolExecutorConfig
            ) and sys.version_info < (3, 14):
                raise RuntimeError(
                    "A subinterpreter region (`to(InterpreterPoolExecutorConfig(...))`) requires "
                    "Python 3.14 or later. Current version: "
                    f"{sys.version_info.major}.{sys.version_info.minor}"
                )
        elif in_region:
            region_has_stage = True
            if _has_ordered_pipe(p):
                raise ValueError(
                    "A stage with `output_order='input'` cannot run inside a `to()` region: "
                    "input order cannot be preserved across the region's independent workers."
                )
    if in_region and not region_has_stage:
        raise ValueError(
            "An empty `to(...)` region has no stages. Add stages before closing it, or remove "
            "the redundant marker."
        )
    if in_region:
        raise ValueError(
            "A `to()` execution region must be closed with `to(MAIN_PROCESS)` before "
            "`add_sink()`. The sink always runs in the main process."
        )


################################################################################
# Builder
################################################################################


class PipelineBuilder(Generic[T, U]):
    """Build :py:class:`Pipeline` object.

    .. note::

       ``PipelineBuilder`` supports only chain of operations.
       If you need to build a pipeline composed of multiple sub-pipelines,
       use :py:class:`~spdl.pipeline.defs.PipelineConfig`.

    .. seealso::

       :ref:`intro`
          Explains the basic usage of ``PipelineBuilder`` and ``Pipeline``.

       :ref:`pipeline-caveats`
          Lists known anti-patterns that can cause a deadlock.

       :ref:`pipeline-parallelism`
          Covers how to switch (or combine) multi-threading and
          multi-processing in detail.

       :ref:`Example: Pipeline definitions <example-pipeline-definitions>`
          Illustrates how to build a complex pipeline that
          ``PipelineBuilder`` does not support.
    """

    def __init__(self) -> None:
        log_api_usage_once("spdl.pipeline.PipelineBuilder")

        self._src: SourceConfig[T] | None = None
        self._process_args: list[
            PipeConfig
            | AggregateConfig
            | DisaggregateConfig
            | PathVariantsConfig
            | PlacementConfig
        ] = []
        self._sink: SinkConfig[U] | None = None

    def add_source(
        self,
        source: Iterable[T] | AsyncIterable[T],
        *,
        continuous: bool = False,
    ) -> "PipelineBuilder[T, U]":
        """Attach an iterator to the source buffer.

        Args:
            source: A lightweight iterator that generates data.

                .. warning::

                   The source iterator must be lightweight as it is executed in async
                   event loop. If the iterator performs a blocking operation,
                   the entire pipeline will be blocked.

            continuous: If ``True``, the source continuously re-iterates,
                injecting a sentinel object representing an epoch boundary
                between iterations. This enables multi-epoch pipeline reuse
                without rebuilding. Use :py:func:`is_epoch_end` to detect
                epoch boundaries in custom merge operations if needed;
                regular pipe stage functions do not need to handle it.
        """
        if self._src is not None:
            raise ValueError("Source already set.")

        self._src = SourceConfig(source, continuous=continuous)
        return self

    def pipe(
        self,
        op: _TPipeInputs[T_, U_],
        /,
        *,
        concurrency: int = 1,
        executor: Executor | None = None,
        name: str | None = None,
        output_order: str = "completion",
        max_failures: int | Fraction | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Apply an operation to items in the pipeline.

        Args:
            op: A function, callable or container with ``__getitem__`` method
                (such as dict, list and tuple).
                If it's function or callable, it is inovked with the input from the input queue.
                If it's container type, the input is passed to ``__getitem__`` method.

                The function or callable must take exactly one argument, which is the output
                from the upstream. If passing around multiple objects, take
                them as a tuple or use :py:class:`~dataclasses.dataclass` and
                define a custom protocol.

                If the result of applying ``op`` to an input item is ``None``,
                the pipeline skips absorb the result and it won't be propagated to
                the downstream stages.

                Optionally, the op can be a generator function, async function or
                async generator function.

                If ``op`` is (async) generator, the items yielded are put in the output
                queue separately.

                .. warning::

                   If ``op`` is synchronous geneartor, and ``executor`` is an instance of
                   :py:class:`concurrent.futures.ProcessPoolExecutor`, the output items
                   are not put in the output queue until the generator is exhausted.

                   Async generator, or synchronous generator without ``ProcessPoolExecutor``
                   does not have this issue, and the yielded items are put in the output
                   queue immediately.

                .. tip::

                   When passing an async op, make sure that the op does not call sync
                   function inside.
                   If calling a sync function, use :py:meth:`asyncio.loop.run_in_executor`
                   or :py:func:`asyncio.to_thread` to delegate the execution to the thread pool.

            concurrency: The maximum number of async tasks executed concurrently.
            executor: A custom executor object to be used to convert the synchronous operation
                into asynchronous one. If ``None``, the default executor is used.

                It is invalid to provide this argument when the given op is already async.
            name: The name (prefix) to give to the task.
            output_order: If ``"completion"`` (default), the items are put to output queue
                in the order their process is completed.
                If ``"input"``, then the items are put to output queue in the order given
                in the input queue.

            max_failures: The maximnum number (int) or rate (Fraction) of failures allowed
                before the pipe operation is considered failure and the whole Pipeline is
                shutdown.
                When an int is provided, it specifies the maximum count of failures.
                When a Fraction is provided (e.g., Fraction(1, 10) for 10%), it specifies
                the maximum failure rate (failures / invocations).
                This overrides the value provided to the :py:meth:`~PipelineBuilder.build`
                method.
        """
        self._process_args.append(
            Pipe(
                op,
                concurrency=concurrency,
                executor=executor,
                name=name,
                output_order=output_order,
                max_failures=max_failures,
            )
        )
        return self

    def aggregate(
        self,
        input: int | Aggregator,
        /,
        *,
        drop_last: bool = False,
    ) -> "PipelineBuilder[T, U]":
        """Buffer the items in the pipeline.

        Args:
            input: Either an integer specifying the number of items to buffer, or an
                :py:class:`~spdl.pipeline.defs.Aggregator` instance for custom aggregation
                logic.

                - If ``int``: Buffers that many items before emitting.
                  It uses :py:class:`~spdl.pipeline.defs.Collate` aggregator class.
                - If :py:class:`~spdl.pipeline.defs.Aggregator`: Custom aggregation using
                  the :py:meth:`~Aggregator.accumulate` and :py:meth:`~Aggregator.flush`
                  methods.

            drop_last: Drop the last aggregation if incomplete.
                - When ``drop_last=False`` (default): Calls :py:meth:`~Aggregator.flush`
                  at EOF
                - When ``drop_last=True``: Does NOT call :py:meth:`~Aggregator.flush`,
                  dropping incomplete batches
        """
        self._process_args.append(Aggregate(input, drop_last=drop_last))
        return self

    def disaggregate(self) -> "PipelineBuilder[T, U]":
        """Disaggregate the items in the pipeline."""
        self._process_args.append(Disaggregate())
        return self

    def to(
        self,
        target: "ProcessPoolExecutorConfig | InterpreterPoolExecutorConfig | _MainProcess",
        /,
    ) -> "PipelineBuilder[T, U]":
        """**[Experimental]** Designate where the subsequent stages execute.

        .. versionadded:: 0.6.0

        Opens (or closes) an *execution region*: every stage added after this call runs on
        ``target`` until the next :py:meth:`to`. A pipeline starts on the main process, so a
        region is opened by ``to(ProcessPoolExecutorConfig(...))`` or ``to(InterpreterPoolExecutorConfig(...))``
        and closed by ``to(MAIN_PROCESS)``. The stages inside a region are fused into one nested
        pipeline that runs together in a worker process (or subinterpreter), so the value handed
        from one stage to the next stays in the worker — it is **not** copied back to the main
        process between stages and need not be picklable. Only the region's inputs and outputs
        cross the boundary.

        Unlike passing ``executor=`` to individual :py:meth:`pipe` calls, a region also carries
        :py:meth:`aggregate`, :py:meth:`disaggregate`, and :py:meth:`path_variants` stages into
        the worker, and gives the worker-pool configuration a single home.

        Each stage's ``concurrency`` applies *within each worker*, so a stage's
        effective concurrency across the pool is ``concurrency × max_workers``. For
        example, ``pipe(op, concurrency=2)`` in a pool with ``max_workers=4`` runs
        up to 8 invocations of ``op`` at once. Size each stage's ``concurrency``
        together with ``max_workers`` to stay within your CPU budget.

        Args:
            target: Where the following stages run.

                - :py:class:`~spdl.pipeline.defs.ProcessPoolExecutorConfig` — a pool of worker processes.
                - :py:class:`~spdl.pipeline.defs.InterpreterPoolExecutorConfig` — a pool of subinterpreters
                  (Python 3.14+; the region's ops must avoid NumPy/PyTorch, which cannot be
                  imported in a subinterpreter).
                - :py:data:`~spdl.pipeline.defs.MAIN_PROCESS` — close the current region; the
                  following stages run in the main process.

                A live :py:class:`~concurrent.futures.Executor` is **not** accepted — pass a spec
                so the pipeline stays expressible as static config. To run a single stage on a
                custom executor, use ``pipe(executor=...)`` instead.

        .. note::

           The region must be closed with ``to(MAIN_PROCESS)`` before :py:meth:`add_sink`, a
           stage inside a region may not use ``output_order="input"`` (order cannot be preserved
           across independent workers), and a subinterpreter region requires Python 3.14+. These
           are checked when the pipeline is built.
        """
        if isinstance(target, Executor):
            raise TypeError(
                "`to()` takes a serializable execution target (ProcessPoolExecutorConfig, "
                "InterpreterPoolExecutorConfig, or MAIN_PROCESS), not a live Executor. To run a single "
                "stage on a custom executor, pass it to `pipe(executor=...)` instead."
            )
        if not isinstance(
            target,
            (ProcessPoolExecutorConfig, InterpreterPoolExecutorConfig, _MainProcess),
        ):
            raise TypeError(
                "`to()` target must be a ProcessPoolExecutorConfig, InterpreterPoolExecutorConfig, or "
                f"MAIN_PROCESS. Got: {type(target).__name__}."
            )
        self._process_args.append(PlacementConfig(target=target))
        return self

    def path_variants(
        self,
        router: Callable,
        paths: Sequence,
        name: str | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Route items to different processing paths based on a router function.

        Args:
            router: A callable that takes an item and returns an int index
                selecting which path the item should be routed to.
            paths: A sequence of paths, where each path is a sequence of
                pipe configs.
            name: Optional name for the stage.
        """
        self._process_args.append(PathVariants(router, paths, name=name))
        return self

    def add_sink(
        self,
        buffer_size: int = 3,
    ) -> "PipelineBuilder[T, U]":
        """Attach a buffer to the end of the pipeline.

        Args:
            buffer_size: The size of the buffer. Pass ``0`` for unlimited buffering.
        """
        if self._sink is not None:
            raise ValueError("Sink is already set.")

        self._sink = SinkConfig(buffer_size)
        return self

    def get_config(self) -> PipelineConfig[U]:
        """Get the pipeline configuration.

        Returns:
            A PipelineConfig object representing the current pipeline configuration.

        Raises:
            RuntimeError: If source or sink is not set, or a subinterpreter region is used on
                Python < 3.14.
            ValueError: If an execution region opened by :py:meth:`to` is not closed with
                ``to(MAIN_PROCESS)`` before the sink, or a stage inside a region uses
                ``output_order="input"``.
        """
        if (src := self._src) is None:
            raise RuntimeError("Source is not set. Did you call `add_source`?")

        if (sink := self._sink) is None:
            raise RuntimeError("Sink is not set. Did you call `add_sink`?")

        _validate_executor_regions(self._process_args)

        return PipelineConfig(src, self._process_args, sink)

    def build(
        self,
        *,
        num_threads: int,
        max_failures: int | Fraction = -1,
        report_stats_interval: float = -1,
        queue_class: type[AsyncQueue] | None = None,
        task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None = None,
        stage_id: int = 0,
        use_thread_output_queue: bool = False,
    ) -> Pipeline[U]:
        """Build the pipeline.

        Args:
            num_threads: The number of threads in the thread pool attached to
                async event loop.

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

            use_thread_output_queue: If ``True``, replace the sink's output queue with a
                ``queue.Queue``-backed queue for lower-latency batch handoff.
                Default: ``False``.

        .. seealso::

           :py:meth:`~spdl.pipeline.PipelineBuilder.to`
              Run a region of stages in a subprocess (or subinterpreter) worker pool.
        """
        return build_pipeline(
            self.get_config(),
            num_threads=num_threads,
            max_failures=max_failures,
            queue_class=queue_class,
            report_stats_interval=report_stats_interval,
            task_hook_factory=task_hook_factory,
            stage_id=stage_id,
            use_thread_output_queue=use_thread_output_queue,
        )
