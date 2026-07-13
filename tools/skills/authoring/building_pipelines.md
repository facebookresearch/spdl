# Building an Efficient SPDL Pipeline From Scratch

Instructions for authoring a **new** SPDL data pipeline (greenfield) that keeps the GPU fed without starving it or triggering the noisy-neighbour effect. If you are converting an *existing* PyTorch `DataLoader`/`Dataset`, start from `migrating_to_spdl_pipeline.md` instead — it covers the same construction patterns plus how to decompose existing `__getitem__`/iterator code. This guide is for when there is no prior loader to migrate.

## Mental Model

An SPDL pipeline is a series of stages connected by queues. Each stage runs concurrently and applies one transformation to items flowing through it:

```
source → pipe(fetch) → pipe(decode) → aggregate(batch) → pipe(collate) → pipe(transfer) → sink
```

The core idea for efficiency: **classify every operation by its nature and give each its own stage with independent concurrency.** Never bundle operations of different natures (e.g. network fetch + CPU decode) into one function — they need different concurrency and would block each other.

| Nature | Examples | Stage design |
|---|---|---|
| Network I/O | HTTP fetch, blob store read, DB query | `.pipe(fn, concurrency=16-32, output_order="completion")` |
| Disk I/O | local file read, mmap | `.pipe(fn, concurrency=8-16)` |
| CPU, GIL-free | media decode (`spdl.io`), tiktoken, NumPy/Torch ops | `.pipe(fn, concurrency=4-8)` |
| CPU, GIL-holding | pure-Python transforms | `.pipe(fn, concurrency=1)` — minimize or replace |
| Batching | collation / stacking | `.aggregate(batch_size)` then `.pipe(collate_fn)` |
| GPU transfer | move batch to device | `spdl.io.transfer_tensor` in a dedicated 1-worker executor |

## Minimal End-to-End Example

```python
from spdl.pipeline import PipelineBuilder
from spdl.source import DistributedRandomSampler

source = DistributedRandomSampler(num_samples, rank=rank, world_size=world_size)

pipeline = (
    PipelineBuilder()
    .add_source(source, continuous=True)
    .pipe(fetch, concurrency=24)                 # network I/O
    .pipe(decode, concurrency=8)                 # CPU, GIL-free (use spdl.io)
    .aggregate(batch_size, drop_last=True)
    .pipe(collate)
    .add_sink(buffer_size=3)
    .build(num_threads=16)
)

# Build the pipeline once, then re-iterate it every epoch — a Pipeline is
# re-iterable, so call get_iterator() afresh each epoch (no rebuild needed).
for epoch in range(num_epochs):
    for batch in pipeline.get_iterator(timeout=900):
        train_step(batch)
```

## Efficiency Principles

1. **One nature per stage.** Split I/O, CPU, and GPU work into separate `.pipe()` calls with independent `concurrency`. This is the single biggest lever.
2. **Respect the CPU budget.** Keep total CPU utilization ≤ 40% — above that, the OS can't schedule GPU kernel launches promptly (the noisy-neighbour effect) and the GPU idles even with data ready. Do not set concurrency higher than `num_CPU_cores / 8`. Using all CPUs for data loading is an anti-pattern.
3. **Use `spdl.io` for media.** Replace Pillow/TorchVision/torchaudio with `spdl.io` for image/video/audio — it releases the GIL, keeps data in native format until batch creation, and converts zero-copy into one contiguous batch tensor.
4. **Async stage functions: pass them as-is.** `.pipe()` accepts both sync (`def`) and async (`async def`) callables. Pass a coroutine function **directly** — SPDL awaits it on its own event loop and drives up to `concurrency` of them at once on a single thread, which is ideal for I/O-bound work.

   ```python
   async def fetch(key):
       async with session.get(url(key)) as resp:
           return await resp.read()

   .pipe(fetch, concurrency=32)   # CORRECT — coroutine function passed as-is
   ```

   **Do not** wrap it as `asyncio.run(fetch(x))`: that spins up and tears down a brand-new event loop on every single item — pure per-item overhead — and forces the work onto pipeline worker threads instead of running as cheap cooperative coroutines on SPDL's shared event loop.
5. **`transfer_tensor` for GPU.** Use `spdl.io.transfer_tensor` in a dedicated `ThreadPoolExecutor(max_workers=1)` so it gets its own CUDA stream and overlaps transfer with compute.
6. **Batching is two steps.** `.aggregate(batch_size, drop_last=True)` then a `.pipe(collate)` stage. `drop_last=True` avoids partial-batch shape mismatches under DDP.
7. **Iterate with a timeout.** Prefer `pipeline.get_iterator(timeout=<seconds>)` over `for batch in pipeline` / manual `next()` — it prevents jobs from hanging forever on a stall.

## Going to Production: Multi-Threading in Subprocess (MTP)

For production training, isolate the CPU-heavy stages in a subprocess and keep only GPU transfer in the main process — this removes GIL contention between data loading and the training loop. Build the CPU stages with `PipelineBuilder`, obtain a `PipelineConfig` via `.get_config()`, hand it to `run_pipeline_in_subprocess()`, then build a small frontend pipeline that only does `transfer_tensor`. Stage functions must then be **picklable** (module-level functions with `functools.partial`, or callable classes — never lambdas or nested functions).

See `migrating_to_spdl_pipeline.md` for the full MTP construction pattern, pickling/thread-safety constraints, and media (`spdl.io`) recipes — they apply identically to a from-scratch build. See `optimization_strategies.md` for deep tuning: concurrency search, subprocess IPC / shared-memory arena, GPU (NVDEC) video decode, decoder-thread tuning, GC-stall mitigation, and headspace analysis.

## Pipeline Lifecycle and Cleanup

A `Pipeline` cleans itself up, so holding a reference to one is safe. It drains its background thread and any worker processes/subinterpreters on two occasions: when it is garbage collected (via `weakref.finalize`), and — if a reference is still held when the program ends — at the very start of interpreter finalization, via a hook `Pipeline.start()` registers with `threading._register_atexit`. You do **not** have to call `Pipeline.stop()` for cleanup to happen, keeping the `Pipeline` alive as an attribute is fine, and it is in fact **required** for a `continuous=True` source re-iterated across epochs, since the same object must survive to be iterated again. (See `Pipeline.start`'s docstring for why the exit hook uses `threading._register_atexit` rather than `atexit`.)

Releasing the reference (or calling `Pipeline.stop()`) as soon as you are done is still good hygiene — it frees the worker processes and memory promptly instead of at GC/exit — but it is no longer needed to avoid a hang.

This matters when a framework stashes the loader on a long-lived object. TorchTNT, for example, holds the dataloader on its `State` (`PhaseState._dataloader`) until the process exits. This no longer hangs the run (the exit hook cleans the pipeline up), but if you want its resources released as soon as training ends (e.g. between fit and eval) rather than at exit, detach it and force a collection with a callback:

```python
import gc

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTestUnit, TTrainUnit


class DetachDataloaderCallback(Callback):
    """Optional: drop TNT's dataloader refs at train end so the SPDL Pipeline
    (and its workers) are released promptly, instead of lingering on State until
    the pipeline is cleaned up at exit."""

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._detach(state)

    def on_exception(
        self,
        state: State,
        unit: TTrainUnit | TEvalUnit | TPredictUnit | TTestUnit,
        exc: BaseException,
    ) -> None:
        # on_train_end does not fire on failure/preemption, so reap here too.
        self._detach(state)

    def _detach(self, state: State) -> None:
        for phase_state in (state.train_state, state.eval_state):
            if phase_state is not None:
                phase_state._dataloader = None  # pyre-ignore[8]
        gc.collect()
```

## Structuring the Construction Code

When code builds a `Pipeline` (or `PipelineConfig`) in more than one shape — e.g. a script that supports several execution modes, or a loader with optional stages — keep **each pipeline's construction in one place**. The maintainability win is being able to read a single fluent chain top-to-bottom and see the whole pipeline; it is lost when the construction is scattered across helpers that each tack on a few stages.

**Route at the top, then let each builder own its whole chain.** Dispatch on the mode/variant first, and have each branch build its pipeline as a single uninterrupted `PipelineBuilder().add_source(...)...add_sink(...).build()` chain:

```python
def build_pipeline_for_mode(mode, ...):
    match mode:
        case "mt":
            return _build_mt_pipeline(...)
        case "mp":
            return _build_mp_pipeline(...)
        case "mtp":
            return _build_mtp_pipeline(...)
        case _:
            raise ValueError(f"unknown mode: {mode!r}")

def _build_mt_pipeline(...):
    return (
        PipelineBuilder()
        .add_source(...)
        .pipe(...)
        .aggregate(...)
        .add_sink(...)
        .build(num_threads=...)
    )
```

**Extract a shared builder only when the variants differ purely by argument.** When two modes really do produce the identical chain except for one value (say a region target or a buffer size), factor out *the entire chain* into a helper parameterized by that value — never a helper that builds only part of it:

```python
def _build_subprocess_pipeline(args):
    return _build_local_pipeline(args, ProcessPoolExecutorConfig(max_workers=args.n))
def _build_subinterp_pipeline(args):
    return _build_local_pipeline(args, InterpreterPoolExecutorConfig(max_workers=args.n))

def _build_local_pipeline(args, target: ProcessPoolExecutorConfig | InterpreterPoolExecutorConfig):
    return (
        PipelineBuilder()
        .add_source(...)
        .aggregate(...)
        .to(target)
        .pipe(decode)
        .to(MAIN_PROCESS)
        .add_sink(...)
        .build(num_threads=args.n)
    )
```

Apply that test honestly — it holds only when the chains are otherwise identical. A variant that changes the *stage graph* does **not** qualify, even if it superficially sounds like "the same thing with a different executor." For example, a PyTorch-`DataLoader`-style mode that aggregates the source into batches *first* and hands each whole batch to a process worker (download + decode + transform fused into one stage) is not "the thread mode with a process executor" — it has a different topology and gets its own complete builder.

**Anti-pattern — partial-construction helpers.** Do **not** write helpers that take a half-built `PipelineBuilder`, append a few stages, and return it for the caller to continue chaining. This splits one pipeline's definition across several functions and forces the reader to jump between them to reconstruct the shape:

```python
# DON'T: the pipeline's stages are scattered across _add_cache_variants and the caller
builder = (
    _add_cache_variants(builder, cache_dir=cd)   # appends some stages, returns builder
    .aggregate(batch_size, drop_last=True)        # caller appends more
    .pipe(decode, executor=executor)
)
```

When variants genuinely differ in topology (not just an argument), inline each one as its own complete chain in a branch, even at the cost of repeating the common `add_source`/`add_sink`/`build` boilerplate — one readable chain per variant beats a deduplicated tangle.

**A `PipelineConfig` is a complete artifact, not a partial build.** Returning `builder...add_sink(...).get_config()` from a helper is fine: a config is a finished, picklable pipeline spec. It is the right tool when the *same* topology must be materialized more than one way — e.g. `build_pipeline(config, ...)` to run in-process versus `run_pipeline_in_subprocess(config, ...)` for an MTP-style subprocess stage — letting one topology definition serve every mode without splitting its construction. Note that a config sent to `run_pipeline_in_subprocess` is pickled: every stage op must be picklable (module-level functions / `functools.partial`, not closures), and a live `ProcessPoolExecutor` can't ride along — use `PriorityProcessPoolExecutor(...).get_executor()` for a process-pool stage that must survive into the subprocess.

## Checklist

- [ ] Every operation classified by nature and given its own stage
- [ ] I/O stages high concurrency (16-32); CPU stages moderate (4-8); GPU transfer isolated (1 worker)
- [ ] Media decoding uses `spdl.io` (not Pillow/TorchVision) where applicable
- [ ] Async functions passed to `.pipe()` as-is (no `asyncio.run()` wrapping)
- [ ] Batching uses `.aggregate()` + an explicit collate stage
- [ ] Total concurrency respects the CPU budget (≤ 40% utilization)
- [ ] Production build uses MTP with picklable stage functions
- [ ] Iterated via `get_iterator(timeout=...)`
- [ ] Each pipeline shape is a single readable builder chain (no partial-construction helpers)
- [ ] Long-lived holders (e.g. TorchTNT `State`) release the `Pipeline` reference promptly to free resources (optional hygiene; cleanup at exit is automatic)
