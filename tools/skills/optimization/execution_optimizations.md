# Remedies for Measured Execution and Transport Costs

Read [analysis.md](analysis.md) first. Use a recipe only when pipeline statistics plus function/holder/transport measurements identify an exposed bottleneck. Empirical architecture comparisons use [architectures.md](architectures.md); they do not exempt unrelated cache/I/O/processing changes from evidence.

## Network calls hidden behind synchronous bridges

Perform the shared `async_boundary_audit.md` startup audit before choosing a remedy. Inspect recurring fetch helpers through their sync facades, retry/decorator layers and actual network client to the standard-library/native boundary. Attribute per-call loop/client setup and shutdown separately from request latency; a cached-model assay does not measure this input path. Preserve the untouched baseline and record any necessary migration repair as its own scoped deviation.

- Pass native `async def` operations directly to `.pipe()` and await the verified network client operation. Do not wrap a coroutine per item with `asyncio.run` or another async-to-sync bridge.
- Move synchronous blocking I/O/CPU work off the asyncio loop into the appropriate executor/process. A native call can release the GIL yet still block the loop when called synchronously; diagnose loop blocking separately from GIL contention.
- Avoid hot-path handler dispatch, file locks, temporary-path materialization and second reads if actual timings show their overhead and a supported native byte-returning path preserves semantics. Setup/manifest access is not the same per-sample path.

```python
async def fetch(key):
    return await native_read(key)

builder.pipe(fetch, concurrency=fetch_concurrency)
```

`native_read` is the workload's verified API, not an invented SPDL storage method. Preserve cache identity, retry/failure policy and data values when changing access. Coalescing requests can reduce fixed I/O cost, but may increase buffering/latency and change worker assignment; measure the complete path.

## Excess scheduling or fused-stage overhead

- Classify operation nature to understand its cost; classification does not mandate one nature per stage. Compare a split only when measured bottlenecks justify independent scheduling.
- Extra stages add allocations, copies, coordination, IPC and queueing. Retain an optimized fused collate if splitting reduces local function latency but loses training throughput.
- Profile default/shared/dedicated executor contention before changing scheduling. Concurrency is in-flight capacity and may queue behind threads; raw stage counts are not CPU resource use. Broad sweeps wait for architecture finalization.

## Recursive metadata/request serialization

- Inspect costs on both producer and consumer sides. If main-process recursive request pickling is the actual GIL holder, send compact indices/references and materialize rows in workers where the original processing contract permits.
- Preserve sample identity/order, duplicate/null/nested metadata, lookup/cache behavior and worker state. One-time process bootstrap differs from recurring queue-feeder serialization; fixing one does not prove the other is gone.
- Torch tensor reducers may already share storage. Containers, metadata, reducers, restoration and pinning still cost CPU; shared tensors do not imply zero GIL or copying overhead.

## Large binary IPC/copy cost

For a measured large-binary transport bottleneck, a supported subprocess arena can replace serialized/copy-heavy binary transport. Verify the version's API and supported payloads:

```python
from spdl.pipeline import SharedMemorySegmentPool, run_pipeline_in_subprocess

arena = SharedMemorySegmentPool(segment_size=max_unit_bytes, count=inflight_units)
cpu_batches = run_pipeline_in_subprocess(
    backend.get_config(), num_threads=worker_threads, arena=arena
)
```

- Segment sizing covers all offloaded binaries plus alignment in one transported unit, not only its largest field. Restored aliases retain the segment; producer backpressure/lifetime and consumer-held views matter.
- Pool and ring backends have different restore/copy semantics. In the inspected version both can block until `acquire_timeout`; `acquire_timeout=0` requests immediate failure. Check source rather than assuming rings are always nonblocking.
- Validate view lifetime, writability, serialization boundaries, slow consumer, early termination, re-iteration and cleanup. The owning subprocess iterable tears down the arena; external caches may retain views.
- Representation choice is Phase 2; capacity/count tuning follows only once retained. Measure CPU and end-to-end benefit, not freed CPU alone. Do not sum shared RSS as unique memory.

## Allocation and result-buffer churn

- Separate collate allocation/copy, private-to-shared transport and pinning costs. Direct shared writes can remove a copy only if lifetime and consumer release are correct.
- Reuse generation/ownership-tracked buffers only after the prior consumer is finished. Holding tensors/views across batches, pin-copy boundaries, CUDA streams or headroom caches can prevent safe reuse.
- Validate slow/retaining consumers, cancellation/error paths and shutdown; memory reduction is not sufficient if it changes values or introduces races. Record buffer sizing separately from the representation mechanism.

## Cache or format overhead

- Profile cache hits/misses, local reads and decode/serde separately. A cache hit still pays file/parse/copy costs; remote transport is not necessarily the next bottleneck.
- Native/mmap/outer-wrapper changes can avoid copying but must preserve inner payload, dtype/format support, bounds, writability and lifetime. Do not assume a universal zero-copy drop-in.
- Match cold/warm cache state and whole-path bytes/copy volume; RPC may add network and serialization to an already efficient local cache. Change cache boundary/identity only as its own recorded hypothesis with data/quality checks.

After any remedy, re-profile the implicated operation and repeat matched completed-work throughput/resources. If it moves rather than removes the bottleneck, use the new analysis for the next approach; do not continue applying the old recipe list.
