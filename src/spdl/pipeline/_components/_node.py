# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import sys
from asyncio import ALL_COMPLETED, FIRST_COMPLETED, Task
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from typing import Any, TypeAlias, TypeVar

from spdl.pipeline._bg_task import BackgroundTaskFactory
from spdl.pipeline._common._misc import create_task
from spdl.pipeline.defs import (
    _PipeArgs,
    _PipeType,
    _SubprocessPipelineConfig,
    AggregateConfig,
    DisaggregateConfig,
    MergeConfig,
    PathVariantsConfig,
    PipeConfig,
    PipelineConfig,
    PlacementConfig,
    SinkConfig,
    SourceConfig,
)

from ._aggregate import _aggregate
from ._common import StageInfo
from ._hook import get_default_hook_class, TaskHook
from ._pipe import (
    _disaggregate,
    _FailCounter,
    _get_fail_counter,
    _merge,
    _ordered_pipe,
    _pipe,
)
from ._queue import _ThreadBasedAsyncQueue, AsyncQueue, get_default_queue_class
from ._sink import _sink
from ._source import _source, _source_continuous
from ._subprocess_pipe import _subprocess_pipeline
from ._variants import _batched_path_variants_merge, _path_variants_router

T = TypeVar("T")
S = TypeVar("S")


_LG: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "_FanInNode",
    "_FanOutNode",
    "_Node",
    "_NodeMixin",
    "_build_pipeline_coro",
    "_get_global_id",
    "_set_global_id",
    "PipelineFailure",
]

# pyre-strict


@dataclass
class _PathVariantsMergeConfig:
    """Internal config for the fan-in merge node of PathVariants.

    This is not user-facing. It is used as the ``cfg`` of the merge ``_Node``
    so that ``_build_node`` can dispatch to the correct coroutine builder.
    """

    batched: bool = False
    """Whether the parent PathVariants stage routes batches (see
    :py:func:`~spdl.pipeline._components._variants._batched_path_variants_merge`)."""


# Used to express the upstream relation ship of coroutines,
# which is necessary for graceful shutdown.
@dataclass
class _NodeMixin:
    """Represents a node in the pipeline graph.

    Each node corresponds to a pipeline stage and contains references to its upstream
    nodes, forming a directed acyclic graph (DAG). Nodes are used internally during
    the pipeline build process to create and manage coroutines for each stage.
    """

    info: StageInfo
    """Stage identity for the node, used for logging, task naming, and telemetry."""

    _coro: Coroutine[None, None, None] | None = field(default=None, init=False)
    _task: Task | None = field(default=None, init=False)

    @property
    def coro(self) -> Coroutine[None, None, None]:
        if self._coro is None:
            raise RuntimeError(
                "[Internal Error] Attempted to retrieve a coroutine object, "
                f"but it is not created: {self}"
            )
        return self._coro

    @property
    def task(self) -> Task:
        if self._task is None:
            raise AssertionError(
                "[INTERNAL ERROR] Attempted to retrieve a task object, "
                f"but it is not created: {self}"
            )
        return self._task

    def create_task(self) -> Task:
        if self._task is not None:
            raise AssertionError(
                "[INTERNAL ERROR] Attempted to cteate a task, "
                f"but it is already crated (started): {self}"
            )
        task = create_task(self.coro, name=str(self.info))
        self._task = task
        return task


@dataclass
class _SourceNode(_NodeMixin):
    """Node with only a single output queue."""

    cfg: SourceConfig
    """The configuration object that defines the behavior of this stage."""

    output_queue: AsyncQueue
    """Queue this node writes to."""

    @property
    def upstream(self) -> "Sequence[_TNodes]":
        """Convenience attributes for compatibility with _Node."""
        return []


@dataclass
class _Node(_NodeMixin):
    """Node with a single input queue and a single output queue."""

    cfg: (
        PipeConfig
        | AggregateConfig
        | DisaggregateConfig
        | PathVariantsConfig
        | _SubprocessPipelineConfig
        | SinkConfig
    )
    """The configuration object that defines the behavior of this stage."""

    upstream: "Sequence[_TNodes]"
    """A sequence of upstream nodes that this node depends on."""

    input_queue: AsyncQueue
    """Queue this node reads from. Empty for source."""

    output_queue: AsyncQueue
    """Queue this node writes to."""

    def __post_init__(self) -> None:
        if len(self.upstream) != 1:
            raise AssertionError(
                "[INTERNAL ERROR] _Node must have exactly one upstream node. "
                f"Upstream nodes: {self.upstream}"
            )


@dataclass
class _FanInNode(_NodeMixin):
    """Node with multiple input queues for fan-in stages (e.g., MergeConfig)."""

    cfg: _PathVariantsMergeConfig | MergeConfig
    """The configuration object that defines the behavior of this stage."""

    upstream: "Sequence[_TNodes]"
    """A sequence of upstream nodes that this node depends on."""

    input_queues: Sequence[AsyncQueue]
    """Queues this node reads from."""

    output_queue: AsyncQueue
    """Queues this node writes to."""

    def __post_init__(self) -> None:
        if len(self.upstream) < 1 or len(self.input_queues) < 1:
            raise AssertionError(
                "[INTERNAL ERROR] _FanInNode must have at least one upstream node/input queue."
            )


@dataclass
class _FanOutNode(_NodeMixin):
    """Node with multiple output queues for fan-out stages (e.g., PathVariants router).

    Fan-out nodes are shared: they appear as upstream of multiple downstream nodes,
    so graph traversals visit them multiple times.
    """

    cfg: PathVariantsConfig
    """The configuration object that defines the behavior of this stage."""

    upstream: "Sequence[_TNodes]"
    """A sequence of upstream nodes that this node depends on."""

    input_queue: AsyncQueue
    """Queues this node reads from."""

    output_queues: Sequence[AsyncQueue]
    """Queues this node writes to."""

    def __post_init__(self) -> None:
        if len(self.upstream) != 1:
            raise AssertionError(
                "[INTERNAL ERROR] _FanOutNode must have exactly one upstream node. "
                f"Upstream nodes: {self.upstream}"
            )

        if len(self.output_queues) < 1:
            raise AssertionError("[INTERNAL ERROR] Output queues cannot be empty.")

    def create_task(self) -> Task:
        if self._task is not None:
            return self._task  # Shared: return existing task on revisit.
        task = create_task(self.coro, name=str(self.info))
        self._task = task
        return task


_TNodes: TypeAlias = _SourceNode | _Node | _FanInNode | _FanOutNode
_TOutputNodes: TypeAlias = _SourceNode | _Node | _FanInNode


def _get_output_queue(n: _TNodes, path_idx: int) -> AsyncQueue:
    match n:
        case _SourceNode() | _Node() | _FanInNode():
            assert path_idx == -1
            return n.output_queue
        case _FanOutNode():
            assert path_idx >= 0
            return n.output_queues[path_idx]
        case _:
            raise NotImplementedError(f"`{type(n)}` is not supported.")


################################################################################
# Build logic
################################################################################


class _MutableInt:
    def __init__(self, v: int) -> None:
        self.v = v

    def __iadd__(self, v: int) -> "_MutableInt":
        self.v += v
        return self

    def __repr__(self) -> str:
        return str(self.v)


def _get_stage_info(cfg: object, pipeline_id: int, stage_id: _MutableInt) -> StageInfo:
    concurrency: int | None = None
    match cfg:
        case SourceConfig():
            base = "src"
        case SinkConfig():
            base = "sink"
        case PipeConfig():
            base = cfg.name
            concurrency = cfg._args.concurrency if cfg._args.concurrency > 1 else None
        case AggregateConfig():
            base = cfg.name or repr(cfg.op)
        case DisaggregateConfig():
            base = "disaggregate"
        case _SubprocessPipelineConfig():
            base = cfg.name
        case MergeConfig():
            base = "merge"
        case PathVariantsConfig():
            base = cfg.name or "path_variants_router"
        case _PathVariantsMergeConfig():
            base = "path_variants_merge"
        case _:
            raise NotImplementedError(f"`{type(cfg)}` is not supported.")
    return StageInfo(
        pipeline_id=pipeline_id,
        stage_id=str(stage_id),
        stage_name=base,
        concurrency=concurrency,
    )


# For queues other than Sink, we use buffer_size=2.
# This makes it possible for queues to always have an item
# as long as upstream is fast enough.
# This make it possible for data readiness (occupancy rate)
# to reach 100%, instead of 99.999999%
_BUFFER_SIZE: int = 2


def _convert_pipes(
    pipes: Sequence[
        PipeConfig
        | AggregateConfig
        | DisaggregateConfig
        | PathVariantsConfig
        | _SubprocessPipelineConfig
        | PlacementConfig
    ],
    n: _TNodes,
    q_class: type[AsyncQueue],
    pipeline_id: int,
    stage_id: _MutableInt,
    path_idx: int = -1,
) -> _TOutputNodes:
    """Convert a sequence of pipe configs into a chain of _Nodes.

    Each config becomes a new _Node linked to the previous one via upstream references.

    Args:
        pipes: The sequence of pipe configurations to convert.
        n: The upstream node to start chaining from.
        q_class: The queue class to use for creating buffers between stages.
        pipeline_id: A unique identifier for the pipeline, used in stage naming.
        stage_id: A mutable counter for assigning unique stage IDs.
        path_idx: Used if the upstream node is _FanOutNode.

    Returns:
        The last node in the chain.
    """
    ret: _TOutputNodes
    for i, cfg in enumerate(pipes):
        idx = path_idx if i == 0 else -1
        match cfg:
            case PathVariantsConfig():
                n = _convert_path_variants(cfg, n, q_class, pipeline_id, stage_id, idx)
            case PlacementConfig():
                # Region markers are build-time directives resolved before the pipeline is
                # built; they must never reach node construction.
                raise ValueError(
                    "PlacementConfig region markers must be resolved before building nodes."
                )
            case _:
                in_q = _get_output_queue(n, idx)
                info = _get_stage_info(cfg, pipeline_id, stage_id)
                stage_id += 1
                out_q = q_class(info, buffer_size=_BUFFER_SIZE)
                n = _Node(info, cfg, [n], input_queue=in_q, output_queue=out_q)
        ret = n
    # pyrefly: ignore [unbound-name]
    return ret


def _convert_path_variants(
    cfg: PathVariantsConfig,
    n: _TNodes,
    q_class: type[AsyncQueue],
    pipeline_id: int,
    stage_id: _MutableInt,
    path_idx: int = -1,
) -> _FanInNode:
    """Convert a PathVariantsConfig into a sub-graph of _Nodes.

    Creates the following graph structure::

        prev_node → router(output_queues=[q0,q1,...])
                      path0_first(input_queues=[q0]) → ... → path0_end ─┐
                      path1_first(input_queues=[q1]) → ... → path1_end ─┤
                      ...                                                 ├→ merge → downstream
                      pathN_first(input_queues=[qN]) → ... → pathN_end ─┘

    Args:
        cfg: The PathVariantsConfig to convert.
        n: The upstream node to connect the router to.
        q_class: Queue class for creating inter-stage buffers.
        pipeline_id: Pipeline identifier for naming.
        stage_id: Mutable stage counter.
        path_idx: Used if the upstream node is _FanOutNode.

    Returns:
        The merge node (last node of the sub-graph).
    """
    if len(cfg.paths) == 0:
        raise AssertionError(
            "[Internal Error] No variant path. "
            "Empty variant should be rejected by PathVariantsConfig.__post_init__."
        )
    if any(len(p) == 0 for p in cfg.paths):
        raise AssertionError(
            "[Internal Error] Empty variant path. "
            "Empty variant should be rejected by PathVariantsConfig.__post_init__."
        )

    # Create per-path queues
    path_queues = []
    for i in range(len(cfg.paths)):
        path_info = StageInfo(
            pipeline_id=pipeline_id,
            stage_id=str(stage_id),
            stage_name=f"path_variants_path{i}",
        )
        path_queues.append(q_class(path_info, buffer_size=_BUFFER_SIZE))

    # Create router node (fan-out: 1 input, N output queues)
    router_info = _get_stage_info(cfg, pipeline_id, stage_id)
    stage_id += 1
    start_node = _FanOutNode(
        router_info,
        cfg,
        [n],
        input_queue=_get_output_queue(n, path_idx),
        output_queues=path_queues,
    )
    # Create each path
    end_nodes = [
        _convert_pipes(path_configs, start_node, q_class, pipeline_id, stage_id, i)
        for i, path_configs in enumerate(cfg.paths)
    ]
    # Create merge node (fan-in: N input queues, 1 output)
    merge_info = StageInfo(
        pipeline_id=pipeline_id,
        stage_id=str(stage_id),
        stage_name="path_variants_merge",
    )
    stage_id += 1
    merge_out_q = q_class(merge_info, buffer_size=_BUFFER_SIZE)
    merge_node = _FanInNode(
        merge_info,
        _PathVariantsMergeConfig(batched=cfg.batched),
        end_nodes,
        input_queues=[n.output_queue for n in end_nodes],
        output_queue=merge_out_q,
    )

    return merge_node


def _convert_config(
    plc: PipelineConfig[T],
    q_class: type[AsyncQueue],
    pipeline_id: int,
    stage_id: _MutableInt,
    disable_sink: bool = False,
    use_thread_output_queue: bool = False,
) -> _TOutputNodes:
    """Convert a :py:class:`~spdl.pipeline.defs.PipelineConfig` into a linked list of
    :py:class:`~spdl.pipeline._components._node._Node` objects.

    This function recursively transforms a declarative pipeline configuration into
    a node graph where each node represents a pipeline stage. The nodes are linked
    via upstream references, forming a directed acyclic graph (DAG) without branching.

    Args:
        plc: The pipeline configuration to convert.
        q_class: The queue class to use for creating buffers between stages.
        pipeline_id: A unique identifier for the pipeline, used in stage naming.
        stage_id: A mutable counter for assigning unique stage IDs.
        disable_sink: If True, skip creating the sink node (used for merge branches).

    Returns:
        The sink node (or the last processing node if disable_sink is True) with
        all upstream nodes linked.
    """
    info = _get_stage_info(plc.src, pipeline_id, stage_id)
    stage_id += 1
    q = q_class(info, buffer_size=_BUFFER_SIZE)
    match cfg := plc.src:
        case SourceConfig():
            n = _SourceNode(info, cfg, output_queue=q)
        case MergeConfig():
            upstream = [
                _convert_config(c, q_class, pipeline_id, stage_id, disable_sink=True)
                for c in cfg.pipeline_configs
            ]
            in_qs = [n.output_queue for n in upstream]
            n = _FanInNode(info, cfg, upstream, input_queues=in_qs, output_queue=q)
        case _:
            raise NotImplementedError(f"`{type(cfg)}` is not supported.")

    if plc.pipes:
        n = _convert_pipes(plc.pipes, n, q_class, pipeline_id, stage_id)

    if not disable_sink:
        info = _get_stage_info(plc.sink, pipeline_id, stage_id)
        stage_id += 1
        if use_thread_output_queue:
            sink_out_q = _ThreadBasedAsyncQueue(info, buffer_size=plc.sink.buffer_size)
        else:
            sink_out_q = q_class(info, buffer_size=plc.sink.buffer_size)
        n = _Node(
            info, plc.sink, [n], input_queue=n.output_queue, output_queue=sink_out_q
        )
    return n


def _build_node(
    node: _TNodes,
    fc_class: type[_FailCounter],
    task_hook_factory: Callable[[StageInfo], list[TaskHook]],
    max_failures: int | Fraction,
) -> None:
    """Build a coroutine for a single node based on its configuration type.

    This function creates the appropriate coroutine for a given node by pattern
    matching on the node's configuration type. Each configuration type corresponds
    to a specific pipeline stage behavior:

    - :py:class:`~spdl.pipeline.defs.SourceConfig`: Creates a source coroutine that
      generates data from an iterator using
      :py:func:`~spdl.pipeline._components._source._source`
    - :py:class:`~spdl.pipeline.defs.MergeConfig`: Creates a merge coroutine that
      combines multiple input streams using
      :py:func:`~spdl.pipeline._components._pipe._merge`
    - :py:class:`~spdl.pipeline.defs.SinkConfig`: Creates a sink coroutine that
      buffers output data using
      :py:func:`~spdl.pipeline._components._sink._sink`
    - :py:class:`~spdl.pipeline.defs.PipeConfig`: Creates a processing coroutine
       using
      :py:func:`~spdl.pipeline._components._pipe._pipe` or
      :py:func:`~spdl.pipeline._components._pipe._ordered_pipe`
    - :py:class:`~spdl.pipeline.defs.AggregateConfig`: Creates a pipe coroutine
      with aggregation logic using
      :py:func:`~spdl.pipeline._components._pipe._pipe`.
    - :py:class:`~spdl.pipeline.defs.DisaggregateConfig`: Creates a pipe coroutine
      with disaggregation logic using
      :py:func:`~spdl.pipeline._components._pipe._pipe`

    The created coroutine is stored in the node's ``_coro`` attribute.

    Args:
        node: The node to build a coroutine for. The node must not already have
            a coroutine.
        fc_class: The failure counter class for tracking task failures.
        task_hook_factory: A factory function for creating task hooks for
            monitoring.
        max_failures: The maximum number of failures allowed before halting.

    Raises:
        ValueError: If an unsupported configuration type is encountered.
        AssertionError: If the node's upstream structure doesn't match the
            configuration type's requirements (e.g., SourceConfig must have no
            upstream nodes).

    Note:
        This function does not handle recursion. Use
        :py:func:`_build_node_recursive` to build coroutines for a node and all
        its upstream dependencies.
    """
    if node._coro is not None:
        if isinstance(node, _FanOutNode):
            return
        raise RuntimeError(f"[Internal Error] coroutine cannot be built twice. {node}")

    match node:
        case _SourceNode():
            fn = _source_continuous if node.cfg.continuous else _source
            node._coro = fn(node.cfg.source, node.output_queue)
        case _FanOutNode():
            hooks = task_hook_factory(node.info)
            node._coro = _path_variants_router(
                node.input_queue,
                node.output_queues,
                node.cfg.router,
                hooks,
                batched=node.cfg.batched,
            )
        case _FanInNode():
            hooks = task_hook_factory(node.info)
            fc = fc_class(max_failures)
            match cfg := node.cfg:
                case MergeConfig():
                    node._coro = _merge(
                        node.info,
                        node.input_queues,
                        node.output_queue,
                        fc,
                        hooks,
                        cfg.op,
                    )
                case _PathVariantsMergeConfig():
                    # A batched stage recombines each input batch's per-path
                    # sub-batches into one batch; the per-item stage passes items
                    # through in arrival order (default merge).
                    node._coro = _merge(
                        node.info,
                        node.input_queues,
                        node.output_queue,
                        fc,
                        hooks,
                        _batched_path_variants_merge if cfg.batched else None,
                    )
                case _:  # pragma: no cover
                    raise ValueError(
                        f"[Internal Error] Unexpected config type in _FanInNode: {cfg}"
                    )
        case _Node():
            match cfg := node.cfg:
                case SinkConfig():
                    node._coro = _sink(node.input_queue, node.output_queue)
                case PipeConfig():
                    in_q, out_q = node.input_queue, node.output_queue
                    hooks = task_hook_factory(node.info)
                    fc = fc_class(max_failures, cfg._max_failures)
                    match cfg._type:
                        case _PipeType.Pipe:
                            node._coro = _pipe(
                                node.info, in_q, out_q, cfg._args, fc, hooks, False
                            )
                        case _PipeType.OrderedPipe:
                            node._coro = _ordered_pipe(
                                node.info, in_q, out_q, cfg._args, fc, hooks
                            )
                        case _:  # pragma: no cover
                            raise ValueError(
                                f"[Internal Error] Unexpected process type: {cfg._type}"
                            )

                case AggregateConfig():
                    in_q, out_q = node.input_queue, node.output_queue
                    hooks = task_hook_factory(node.info)
                    fc = fc_class(max_failures)
                    # Use specialized aggregate pipe that drains the input queue
                    # in bulk to reduce context switching overhead.
                    # When drop_last=False, pass EOF to op so it can emit remaining items
                    # When drop_last=True, don't pass EOF to op, effectively dropping last batch
                    node._coro = _aggregate(
                        node.info,
                        in_q,
                        out_q,
                        cfg.op,
                        fc,
                        hooks,
                        op_requires_eof=not cfg.drop_last,
                    )

                case DisaggregateConfig():
                    in_q, out_q = node.input_queue, node.output_queue
                    hooks = task_hook_factory(node.info)
                    fc = fc_class(max_failures)
                    args = _PipeArgs(op=_disaggregate)  # pyre-ignore[6]
                    node._coro = _pipe(node.info, in_q, out_q, args, fc, hooks, False)

                case _SubprocessPipelineConfig():
                    # The fused run executes as a nested pipeline inside the worker pool; the
                    # per-stage hooks/stats fire there, so this bridge stage needs none here.
                    node._coro = _subprocess_pipeline(
                        node.input_queue, node.output_queue, cfg.handle
                    )

                case _:  # pragma: no cover
                    raise ValueError(
                        f"[Internal Error] Unexpected config type in _Node: {cfg}"
                    )


def _build_node_recursive(
    node: _TNodes,
    fc_class: type[_FailCounter],
    task_hook_factory: Callable[[StageInfo], list[TaskHook]],
    max_failures: int | Fraction,
) -> None:
    """Recursively build coroutines for a node and all its upstream nodes.

    This function traverses the node graph starting from the given node, following
    upstream references recursively. For each node, it creates a coroutine based on
    the node's configuration type (Source, Pipe, Merge, Sink, etc.).

    Args:
        node: The node to build coroutines for.
        fc_class: The failure counter class for tracking task failures.
        task_hook_factory: A factory function for creating task hooks for monitoring.
        max_failures: The maximum number of failures allowed before halting.

    Raises:
        RuntimeError: If attempting to build a coroutine for a node that already has one.
    """
    if isinstance(node, _FanOutNode) and node._coro is not None:
        return

    for n in node.upstream:
        _build_node_recursive(n, fc_class, task_hook_factory, max_failures)

    _build_node(node, fc_class, task_hook_factory, max_failures)


# Used to append stage name with pipeline
_PIPELINE_ID: int = -1


def _get_global_id() -> int:
    return _PIPELINE_ID


def _set_global_id(val: int) -> None:
    global _PIPELINE_ID
    _PIPELINE_ID = val


def _default_q(interval: float) -> type[AsyncQueue]:
    queue_class = get_default_queue_class()
    return partial(queue_class, interval=interval)  # pyre-ignore[7]


def _default_hook_factory(
    report_stats_interval: float,
) -> Callable[[StageInfo], list[TaskHook]]:
    if (hook_class := get_default_hook_class()) is not None:

        def _hook_factory(stage_info: StageInfo) -> list[TaskHook]:
            # pyrefly: ignore [bad-argument-count, unexpected-keyword]
            return [hook_class(stage_info, interval=report_stats_interval)]

    else:

        def _hook_factory(_: StageInfo) -> list[TaskHook]:
            return []

    return _hook_factory


def _build_pipeline_node(
    plc: PipelineConfig[T],
    /,
    *,
    max_failures: int | Fraction,
    report_stats_interval: float,
    queue_class: type[AsyncQueue] | None,
    task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None,
    stage_id: int,
    use_thread_output_queue: bool = False,
) -> _TOutputNodes:
    global _PIPELINE_ID
    _PIPELINE_ID += 1

    q_class = _default_q(report_stats_interval) if queue_class is None else queue_class
    hook_factory = (
        _default_hook_factory(report_stats_interval)
        if task_hook_factory is None
        else task_hook_factory
    )

    fc_class = _get_fail_counter()
    node = _convert_config(
        plc,
        q_class,
        _PIPELINE_ID,
        _MutableInt(stage_id),
        use_thread_output_queue=use_thread_output_queue,
    )
    _validate_continuous_mode(node)
    _build_node_recursive(node, fc_class, hook_factory, max_failures)
    return node


def _collect_source_nodes(node: _TNodes) -> list[_SourceNode]:
    """Collect all source nodes in the pipeline graph."""
    if isinstance(node, _SourceNode):
        return [node]
    result: list[_SourceNode] = []
    for n in node.upstream:
        result.extend(_collect_source_nodes(n))
    return result


def _validate_continuous_mode(node: _TNodes) -> None:
    """Validate that all source nodes agree on continuous mode."""
    sources = _collect_source_nodes(node)
    if not sources:
        return
    continuous_values = {s.cfg.continuous for s in sources}
    if len(continuous_values) > 1:
        raise ValueError(
            "Mixed continuous mode is not supported. All sources in a pipeline "
            "(including merged sub-pipelines) must have the same continuous setting."
        )


################################################################################
# Coroutine execution logics
################################################################################
def _start_tasks(node: _TNodes) -> set[Task]:
    if isinstance(node, _FanOutNode) and node._task is not None:
        return {node.task}
    node.create_task()
    ret = {node.task}
    for n in node.upstream:
        ret |= _start_tasks(n)
    return ret


def _cancel_recursive(node: _TNodes) -> None:
    node.task.cancel()
    for n in node.upstream:
        _cancel_recursive(n)


def _cancel_orphaned(node: _TNodes) -> None:
    """
    Cancel upstream tasks when this node's task completes to avoid leaving producer tasks orphaned.

    This function traverses the pipeline graph upstream from the given node.
    If the current node's asyncio Task is done (completed, errored, or cancelled),
    all upstream tasks are recursively cancelled to avoid deadlocks where upstream
    producers keep waiting to push data into queues that will no longer be consumed.

    The function then continues traversing upstream regardless of the current node's
    state to ensure any upstream nodes that have completed also trigger their own
    upstream cancellations.

    Args:
        node: The pipeline node whose task and upstream relationship are inspected
              for potential cancellation.
    """
    task = node.task
    if task.done():  # done includes success, error or cancelled.
        for n in node.upstream:
            _cancel_recursive(n)

    for n in node.upstream:
        _cancel_orphaned(n)


def _gather_error(
    node: _TNodes, _visited: set[int] | None = None
) -> list[tuple[str, Exception]]:
    if _visited is None:
        _visited = set()
    if id(node) in _visited:
        return []
    _visited.add(id(node))

    task = node.task
    errs = []
    if not task.cancelled() and (err := task.exception()) is not None:
        errs.append((task.get_name(), err))

    for n in node.upstream:
        errs.extend(_gather_error(n, _visited))
    errs.sort(key=lambda i: i[0])
    # pyrefly: ignore [bad-return]
    return errs


if sys.version_info >= (3, 11):

    class PipelineFailure(ExceptionGroup[Exception]):
        """PipelineFailure()
        Thrown by :py:class:`spdl.pipeline.Pipeline` when pipeline encounters an error.

        On Python 3.11+, this is an :py:class:`ExceptionGroup` subclass, so
        ``except*`` and :py:meth:`~BaseExceptionGroup.subgroup` /
        :py:meth:`~BaseExceptionGroup.split` work as expected.
        Individual exceptions are accessible via the
        :py:attr:`~BaseExceptionGroup.exceptions` attribute.
        """

        def __new__(
            cls,
            errs: list[tuple[str, Exception]],
        ) -> "PipelineFailure":
            for name, exc in errs:
                exc.add_note(f"Pipeline stage: {name}")
            return super().__new__(
                cls,
                "pipeline stages failed",
                [e for _, e in errs],
            )

        def __init__(self, errs: list[tuple[str, Exception]]) -> None:
            pass

        def derive(  # pyre-ignore[14]
            self, excs: Sequence[Exception]
        ) -> "PipelineFailure":
            return super().__new__(type(self), self.message, excs)

else:

    class PipelineFailure(RuntimeError):  # type: ignore[no-redef]
        """PipelineFailure()
        Thrown by :py:class:`spdl.pipeline.Pipeline` when pipeline encounters an error.
        """

        def __init__(self, errs: list[tuple[str, Exception]]) -> None:
            msg = []
            for k, v in errs:
                e = str(v)
                msg.append(f"{k}:{e if e else type(v).__name__}")
            msg.sort()
            super().__init__(", ".join(msg))
            self.exceptions: tuple[Exception, ...] = tuple(e for _, e in errs)


async def _run_pipeline_coroutines(
    node: _TNodes,
    background_tasks: Sequence[BackgroundTaskFactory] | None = None,
) -> None:
    """Orchestrate the execution of all pipeline stage coroutines.

    This is the main coroutine that manages the lifecycle of all pipeline stage tasks.
    It creates asyncio Tasks from each node's coroutine, monitors their execution,
    handles failures, and ensures proper cleanup.

    The execution flow:

    1. Creates asyncio Tasks for all nodes (starting from sink, traversing upstream)
    2. Starts background tasks that run alongside the pipeline stages
    3. Waits for tasks to complete using asyncio.wait with FIRST_COMPLETED
    4. When a task completes, cancels orphaned upstream tasks to prevent deadlocks
    5. Handles cancellation requests from the foreground thread
    6. Cancels background tasks when pipeline stages complete
    7. Gathers errors from failed tasks and raises PipelineFailure if any

    Args:
        node: The sink node of the pipeline (all upstream nodes are accessed via
            the upstream references).
        background_tasks: Optional list of BackgroundTaskFactory callables. Each
            factory is called to create a BackgroundTask instance whose ``run()``
            coroutine runs alongside the pipeline. Background tasks are cancelled
            when the pipeline completes and their errors are logged but do not
            cause the pipeline to fail.

    Raises:
        asyncio.CancelledError: If the pipeline is cancelled by the foreground thread.
        PipelineFailure: If any pipeline stage encounters an error.
    """
    pending = _start_tasks(node)

    # Start background tasks
    bg_tasks: set[Task] = set()
    for factory in background_tasks or []:
        try:
            task_obj = factory()
            bg_tasks.add(create_task(task_obj.run(), name="background_task"))
        except Exception:
            _LG.exception("Failed to start a background task.")

    while pending:
        # Note:
        # `asyncio.wait` does not automatically propagate the cancellation to its children.
        # For graceful shutdown, we manually cancel the child tasks.
        #
        # Also, it seems asyncio loop throws Cancellation on most outer task.
        # I am not sure where this behavior is documented, but here is an example script to
        # demonstrate the behavior.
        # https://gist.github.com/mthrok/3a1c11c2d8012e29f4835679ac0baaee
        try:
            _, pending = await asyncio.wait(pending, return_when=FIRST_COMPLETED)
        except asyncio.CancelledError:
            for task in pending:
                task.cancel()
            for task in bg_tasks:
                task.cancel()
            if tasks := (pending | bg_tasks):
                await asyncio.wait(tasks, return_when=ALL_COMPLETED)
            raise
        else:
            # If a task is done, we cancel all of its upstream stages, because
            # otherwise they get stuck.
            # This does not happen usually because we implement pipes in a way that
            # upstream stages complete first.
            # But `Merge` with custom op can exit before all the upstream stages complete.
            # https://github.com/facebookresearch/spdl/issues/1204
            #
            # Note: Usually, one must await the cancelled task, but since we await all
            # tasks in this `while pending` loop, here we cancel and not await.
            _cancel_orphaned(node)

    # Pipeline stages are done — cancel background tasks
    for task in bg_tasks:
        task.cancel()
    if bg_tasks:
        await asyncio.wait(bg_tasks, return_when=ALL_COMPLETED)
        for task in bg_tasks:
            if not task.cancelled() and (exc := task.exception()) is not None:
                _LG.error("Background task failed: %s", exc, exc_info=exc)

    if errs := _gather_error(node):
        raise PipelineFailure(errs)


def _build_pipeline_coro(
    plc: PipelineConfig[Any],
    /,
    *,
    max_failures: int | Fraction = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue] | None = None,
    task_hook_factory: Callable[[StageInfo], list[TaskHook]] | None = None,
    stage_id: int = 0,
    background_tasks: Sequence[BackgroundTaskFactory] | None = None,
    use_thread_output_queue: bool = False,
) -> tuple[Coroutine[None, None, None], asyncio.Queue]:
    try:
        node = _build_pipeline_node(
            plc,
            max_failures=max_failures,
            report_stats_interval=report_stats_interval,
            queue_class=queue_class,
            task_hook_factory=task_hook_factory,
            stage_id=stage_id,
            use_thread_output_queue=use_thread_output_queue,
        )
        coro = _run_pipeline_coroutines(node, background_tasks=background_tasks)

        return coro, node.output_queue
    except Exception as e:
        if sys.version_info[1] >= 11:
            e.add_note(f"PipelineConfig: {plc}")
        raise
