# SPDL Pipeline

A generic async pipeline engine for multi-stage data processing. Stages run concurrently in a background thread's asyncio event loop and expose results through a queue consumed from the foreground thread.

## Directory Structure

```
pipeline/
├── __init__.py          # Public API surface
├── _build.py            # build_pipeline() — converts config into a running Pipeline
├── _builder.py          # PipelineBuilder — fluent API for building PipelineConfig
├── _pipeline.py         # Pipeline class — manages event loop thread and item consumption
├── _bg_task.py          # BackgroundTask ABC and default registration
├── _profile.py          # profile_pipeline() and diagnostic mode
├── _pgrp_stats.py       # Process-group resource monitoring (CPU, RSS, IO, network)
├── config.py            # Misc config helpers
│
├── defs/                # Declarative pipeline configuration
│   └── _defs.py         # SourceConfig, PipeConfig, AggregateConfig, DisaggregateConfig,
│                        # PathVariantsConfig, MergeConfig, SinkConfig, PipelineConfig
│                        # + factory functions: Pipe, Aggregate, Disaggregate, PathVariants, Merge
│
├── _components/         # Pipeline stage implementation
│   ├── _node.py         # Config → node graph → asyncio coroutines (_build_pipeline_coro)
│   ├── _source.py       # Source stage coroutine
│   ├── _pipe.py         # Pipe, merge, disaggregate stage coroutines
│   ├── _aggregate.py    # Aggregate stage coroutine
│   ├── _variants.py     # PathVariants router coroutine (fan-out + fan-in)
│   ├── _sink.py         # Sink stage coroutine
│   ├── _queue.py        # AsyncQueue, StatsQueue (with periodic stats reporting)
│   ├── _hook.py         # TaskHook, TaskStatsHook (per-task timing + P90/P99)
│   └── _common.py       # EOF/SKIP sentinels, P-square percentile, periodic dispatch
│
├── _common/             # Shared utilities
│   ├── _convert.py      # Sync-to-async iterable conversion
│   ├── _misc.py         # create_task helper
│   ├── _source_locator.py  # Locate source file/line of user-provided ops
│   └── _types.py        # Type aliases
│
├── _iter_utils/         # Subprocess/subinterpreter pipeline execution
│   ├── _subprocess.py   # iterate_in_subprocess
│   ├── _subinterpreter.py  # iterate_in_subinterpreter
│   ├── _cache_iterator.py  # cache_iterator utility
│   └── _common.py       # Shared helpers
│
└── fb/                  # Meta-internal extensions (see fb/README.md)
```

## Architecture

### Pipeline Lifecycle

```
1. CONFIGURE           2. BUILD                    3. RUN                     4. CONSUME

PipelineBuilder ──►  PipelineConfig ──►  Pipeline ──►  foreground thread
   (fluent API)      (declarative)       .start()      .get_item() / __iter__()
                         │                  │
                         ▼                  ▼
                build_pipeline()      _EventLoop (background Thread)
                         │               asyncio.run(main_coro)
                         ▼                  │
              _build_pipeline_coro()     stages connected by AsyncQueues
                         │               EOF sentinel propagates shutdown
                         ▼               BackgroundTasks run alongside
                   (coroutine, output_queue)
```

### Pipeline Graph Construction (`_components/_node.py`)

`_build_pipeline_coro()` converts a `PipelineConfig` into a graph of `_Node` objects, then builds asyncio coroutines for each:

1. **`_convert_config()`** walks the config tree and creates node objects linked by queues:
   - `_SourceNode` — single output queue, no upstream
   - `_Node` — single input + output queue (Pipe, Aggregate, Disaggregate, Sink)
   - `_FanOutNode` — single input, multiple output queues (PathVariants router)
   - `_FanInNode` — multiple input queues, single output (Merge, PathVariants merge)

2. **`_build_node_recursive()`** traverses the graph and assigns each node a coroutine (`_source`, `_pipe`, `_aggregate`, `_sink`, `_merge`, `_path_variants_router`).

3. **`_run_pipeline_coroutines()`** starts all node coroutines as asyncio Tasks, monitors them via `asyncio.wait(FIRST_COMPLETED)`, cancels orphaned upstream tasks, and manages background tasks.

### Stage Types and Their Properties

| Config | Node Type | Has Task Hook | Has Queue Stats | Coroutine |
|--------|-----------|---------------|-----------------|-----------|
| SourceConfig | `_SourceNode` | No | Yes (output) | `_source` |
| PipeConfig | `_Node` | Yes | Yes (output) | `_pipe` / `_ordered_pipe` |
| AggregateConfig | `_Node` | Yes | Yes (output) | `_aggregate` |
| DisaggregateConfig | `_Node` | Yes | Yes (output) | `_pipe` (with `_disaggregate` op) |
| MergeConfig | `_FanInNode` | Yes | Yes (output) | `_merge` |
| PathVariantsConfig | `_FanOutNode` | Yes | Yes (outputs) | `_path_variants_router` |
| SinkConfig | `_Node` | No | Yes (output) | `_sink` |

### Runtime Stats

Stats collection is built on two extension points that can be customized via `set_default_hook_class` and `set_default_queue_class`:

- **`TaskStatsHook`** (`_hook.py`): Wraps each task execution, tracks count, failures, average/P90/P99 times using P-square streaming percentile. Reports periodically via `interval_stats_callback`.
- **`StatsQueue`** (`_queue.py`): Extends `AsyncQueue` with put/get timing, occupancy rate tracking, and periodic reporting via `interval_stats_callback`. Each stage calls `stage_hook()` on its output queue.

**Stage naming convention:** `{pipeline_id}:{stage_id}:{base_name}` for task hooks, `{pipeline_id}:{stage_id}:{base_name}_queue` for queues.

**Note:** Source and Sink stages have no task hooks — they only emit queue stats.

### Background Tasks

Background tasks (`BackgroundTask` subclasses) run in the pipeline's event loop alongside stage coroutines. They are cancelled when the pipeline completes. Errors are logged but don't fail the pipeline. Register defaults via `set_default_background_tasks`.

### Profiling (`_profile.py`)

`profile_pipeline()` benchmarks each Pipe stage independently at concurrency levels [32, 16, 8, 4, 1], measuring QPS and queue occupancy. Activated automatically via `SPDL_PIPELINE_DIAGNOSTIC_MODE=1` env var.

### Meta-Internal Extensions (`fb/`)

The `fb/` subpackage is automatically imported at `pipeline/__init__.py` load time (silently skipped in OSS). It configures default hook, queue, background task, and profiling classes to emit telemetry to internal backends. See `fb/README.md` for details.
