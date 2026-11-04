# Optional SPDL Construction Patterns

Use only when implementing a required execution adapter, a greenfield pipeline or a separately measured post-parity candidate. The baseline procedure is in [migrating_to_spdl_pipeline.md](migrating_to_spdl_pipeline.md); these patterns do not require decomposition or component replacement during migration.

## Preserve existing operations

- Reuse the original sampler/iterable and callable components. Make subprocess callables picklable with module-level adapters/`functools.partial` or explicit callable-state serialization; create unpicklable resources in their worker owner.
- Preserve the original batch/partial-batch policy and output ordering. Thread concurrency requires actual library/callable thread safety and safe state ownership; use TLS where the workload requires it rather than assuming every decoder/tokenizer releases the GIL.
- Pass existing `async def` operations directly to `.pipe()`. Do not introduce `asyncio.run` or another sync bridge per item. Place blocking work in an appropriate executor/process.
- Keep GPU operations outside CPU worker regions/subprocesses. Validate device/stream handoff and tensor/shared-buffer lifetime; background execution alone does not make handoff safe.

## Explicit pipeline and shared priority executor

Illustrative topology using the original operations; resource values and the split are candidates, not universal defaults:

```python
from spdl.pipeline import PipelineBuilder, PriorityThreadPoolExecutor

pool = PriorityThreadPoolExecutor(max_workers=num_threads)
pipeline = (
    PipelineBuilder()
    .add_source(source, continuous=True)
    .pipe(fetch_data, executor=pool.get_executor(), concurrency=fetch_concurrency)
    .pipe(decode, executor=pool.get_executor(), concurrency=decode_concurrency)
    .aggregate(batch_size, drop_last=baseline_drop_last)
    .pipe(collate, executor=pool.get_executor())
    .add_sink(buffer_size=sink_depth)
    .build(num_threads=1)
)
```

Later `get_executor()` calls prioritize downstream work. Test the complete path; keep fused processing if independent scheduling costs more than it saves. Do not add a second large default executor when the shared pool already owns the work.

## Background H2D

Keep the original transfer boundary for initial parity. Afterward, prefer verified non-regressing background-transfer hygiene; retain the original path if this candidate regresses.

```python
from concurrent.futures import ThreadPoolExecutor
from spdl.io import transfer_tensor_h2d

transfer_executor = ThreadPoolExecutor(max_workers=1)
frontend = (
    PipelineBuilder()
    .add_source(cpu_batches, continuous=True)
    .pipe(transfer_tensor_h2d, executor=transfer_executor)
    .add_sink(buffer_size=sink_depth)
)
```

The helper pins CPU tensors, copies on a thread-local CUDA stream and synchronizes that stream before returning. Overlap comes from calling it in the background relative to training. Preserve pin/shared-output ownership and cleanup of the transfer executor.

## MTP candidate

`run_pipeline_in_subprocess` takes a complete `PipelineConfig`, not a built pipeline or iterator:

```python
from spdl.pipeline import run_pipeline_in_subprocess

backend = (
    PipelineBuilder()
    .add_source(source, continuous=True)
    .pipe(fetch_data, executor=pool.get_executor(), concurrency=fetch_concurrency)
    .pipe(decode, executor=pool.get_executor(), concurrency=decode_concurrency)
    .aggregate(batch_size, drop_last=baseline_drop_last)
    .pipe(collate, executor=pool.get_executor())
    .add_sink(buffer_size=sink_depth)
)
cpu_batches = run_pipeline_in_subprocess(
    backend.get_config(), num_threads=num_threads
)
```

Use the frontend above only when it matches the intended transfer contract. MTP is optional; it removes backend work from the trainer's GIL domain, not all frontend serialization/pinning interference.

## Multiprocessing regions

- For an original multiprocessing loader, regions can be the closest shape: `.to(ProcessPoolExecutorConfig(...))` opens in-worker processing and `.to(MAIN_PROCESS)` closes it before the sink.
- Keep adjacent lookup/decode/transform/collate work and large intermediates inside the worker; per-stage process executors add repeated IPC. Boundary payloads and callables must be picklable; retain original worker/order/RNG semantics.
- An outer MTP wrapper is a separate measured design, not required around every region. Count worker, per-stage and library thread demand across all ranks; actual resource and throughput evidence decides the budget.

## Decoder or stage-boundary changes

Decoder replacements and stage splitting are post-parity hypotheses. Compare values, layouts/dtypes, colorspace/EXIF/rounding, temporal selection, masks and RNG behavior against the original, then measure end-to-end impact. Do not replace working media operations solely to satisfy a recipe.

Build once where the source/lifecycle requires persistence, then obtain a per-epoch iterator with `get_iterator(timeout=...)`. Preserve the trainer's real loader interface, epoch boundaries and cleanup rather than imposing a flat `next()` loop.
