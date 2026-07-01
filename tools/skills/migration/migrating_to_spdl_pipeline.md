# Migrating Data Loaders to SPDL Pipeline

Instructions for converting existing data loading code (PyTorch DataLoader, StatefulDataLoader, or any Dataset/iterable-based loader) into an SPDL Pipeline.

## Goal

Produce the final-form SPDL pipeline directly. Do not preserve intermediate compatibility layers or Dataset abstractions.

## Analysis: Decompose the Existing Data Loading Logic

Before writing any SPDL code, analyze the existing `__getitem__`, `__iter__`, or collate logic and classify each operation by its nature:

| Nature | Examples | SPDL Stage Design |
|---|---|---|
| **Network I/O** | HTTP fetch, blob store read, database query, file download | `.pipe(fn, concurrency=16-32, output_order="completion")` |
| **Disk I/O** | Local file read, mmap access | `.pipe(fn, concurrency=8-16)` |
| **CPU-bound (GIL-free)** | Image/video/audio decode, tokenization (tiktoken), tensor ops, NumPy | `.pipe(fn, concurrency=4-8)` |
| **CPU-bound (GIL-holding)** | Pure Python transforms, non-GIL-releasing libraries | `.pipe(fn, concurrency=1)` — minimize or replace |
| **Batching** | Collation, stacking tensors | `.aggregate(batch_size)` then `.pipe(collate_fn)` |
| **GPU transfer** | `.to(device)`, pinned memory copy | `spdl.io.transfer_tensor` in a dedicated 1-worker ThreadPoolExecutor (see below) |

### Key Questions to Ask

1. **What does each operation in `__getitem__` / the iterator actually do?** Split monolithic functions into separate stages by nature.
2. **Does each operation release the GIL?** If not, can it be replaced with a GIL-releasing alternative? (e.g., Pillow → `spdl.io`, custom Python parsing → C extension)
3. **Is each operation thread-safe?** If not, use thread-local storage (TLS). HuggingFace tokenizers require TLS; tiktoken is safe.
4. **What is the data flow?** Identify where items multiply (disaggregate) or combine (aggregate).

## Extracting Stage Functions from Existing Code

When decomposing `__getitem__` or iterator logic into pipeline stages, prefer **Path 1** (leave original intact). If numerical issues arise, fall back to **Path 2** (refactor original).

### Path 1: Reimplement Separately (Default)

Write new standalone functions for each stage based on reading and understanding the original implementation. Leave the original `__getitem__`/Dataset code untouched. This allows the user to run both pipelines side-by-side and compare outputs for correctness.

### Path 2: Refactor Original (Fallback)

If the user reports numerical discrepancies between the original and new pipeline, refactor the original `__getitem__` implementation to call the same extracted functions used by the SPDL pipeline. This ensures both paths share identical logic, making it possible to isolate whether differences come from the extraction or from pipeline execution order/threading.

Inform the user: "If you observe any numerical differences between the original loader and the new SPDL pipeline, I can refactor the original implementation to use the same extracted functions, so you can verify the extraction itself is not the source of divergence."

## Constructing the Pipeline

### Source

Replace sampler/index logic:

```python
from spdl.dataloader import PipelineBuilder
from spdl.source import DistributedRandomSampler

source = DistributedRandomSampler(num_samples, rank=rank, world_size=world_size)
```

Or any iterable that yields keys/paths/metadata for downstream stages.

### Stages

Map each classified operation to a `.pipe()` stage with appropriate concurrency. Separate operations of different natures into distinct stages — never bundle network I/O and CPU decode in one function.

`.pipe()` accepts both sync (`def`) and async (`async def`) callables. If the original loader used `async` I/O (e.g. `aiohttp`), **pass the coroutine function to `.pipe()` as-is** — SPDL awaits it on its own event loop. Do **not** wrap it as `asyncio.run(coro(x))`: that spins up and tears down a brand-new event loop on every item — pure per-item overhead — and forces the work onto worker threads instead of running as cheap cooperative coroutines on SPDL's shared event loop.

Use `PriorityThreadPoolExecutor` from `spdl.pipeline` as the shared executor. It prioritizes downstream stages over upstream ones — when the thread pool is contended, items closer to the pipeline output are processed first, reducing end-to-end latency and ensuring data flows through the pipeline faster.

```python
from spdl.pipeline import PipelineBuilder, PriorityThreadPoolExecutor

pool = PriorityThreadPoolExecutor(max_workers=num_threads)

pipeline = (
    PipelineBuilder()
    .add_source(source, continuous=True)
    .pipe(fetch_data, executor=pool.get_executor(), concurrency=24)   # network I/O
    .pipe(decode, executor=pool.get_executor(), concurrency=8)        # CPU, GIL-free
    .aggregate(batch_size, drop_last=True)
    .pipe(collate, executor=pool.get_executor())                      # batching
    .add_sink(buffer_size=3)
    .build(num_threads=1)
)
```

Each `pool.get_executor()` call auto-assigns increasing priority to later (downstream) stages. The pipeline's own thread pool can be set to 1 since execution is delegated to the priority executor.

### Media Workloads: Use spdl.io

For image/video/audio, replace Pillow/TorchVision/torchaudio with `spdl.io` to eliminate redundant memory allocation:

```python
import spdl.io

filter_desc = spdl.io.get_video_filter_desc(scale_width=224, scale_height=224)

def decode_image(path: str):
    packets = spdl.io.demux_image(path)
    return spdl.io.decode_packets(packets, filter_desc=filter_desc)

def collate_frames(items):
    frames, labels = list(zip(*items))
    buffer = spdl.io.convert_frames(frames)
    tensor = spdl.io.to_torch(buffer).permute(0, 3, 1, 2)
    return tensor, labels
```

`spdl.io` keeps data in native format (e.g., YUV420) until batch creation, then converts directly into one contiguous batch tensor with zero-copy to PyTorch.

### GPU Transfer: `spdl.io.transfer_tensor`

Always use `spdl.io.transfer_tensor` for moving batches to GPU. It pins memory and uses a dedicated CUDA stream internally, enabling overlap of data transfer and compute.

Run it in a **dedicated ThreadPoolExecutor with 1 worker** to isolate the transfer thread from pipeline worker threads. This ensures the CUDA stream is not shared and transfer does not contend with CPU stages:

```python
from concurrent.futures import ThreadPoolExecutor
from spdl.io import transfer_tensor

_gpu_executor = ThreadPoolExecutor(max_workers=1)

pipeline = (
    PipelineBuilder()
    ...
    .pipe(transfer_tensor, executor=_gpu_executor)
    .add_sink(buffer_size=3)
    .build(num_threads=1)
)
```

### Production: Multi-Threading in Subprocess (MTP)

For production training, isolate CPU work in a subprocess to avoid GIL contention with GPU kernels:

```python
import spdl.pipeline
from concurrent.futures import ThreadPoolExecutor
from spdl.io import transfer_tensor
from spdl.pipeline import PipelineBuilder, PriorityThreadPoolExecutor

_gpu_executor = ThreadPoolExecutor(max_workers=1)

# All CPU work — runs in subprocess with priority scheduling
pool = PriorityThreadPoolExecutor(max_workers=num_threads)

backend = (
    PipelineBuilder()
    .add_source(source, continuous=True)
    .pipe(fetch_data, executor=pool.get_executor(), concurrency=24)
    .pipe(decode, executor=pool.get_executor(), concurrency=8)
    .aggregate(batch_size, drop_last=True)
    .pipe(collate, executor=pool.get_executor())
    .add_sink(buffer_size=3)
)

source2 = spdl.pipeline.run_pipeline_in_subprocess(
    backend.get_config(),
    num_threads=num_threads,
    mp_context="forkserver",
)

# GPU transfer only — runs in main process, dedicated thread + CUDA stream
frontend = (
    PipelineBuilder()
    .add_source(source2, continuous=True)
    .pipe(transfer_tensor, executor=_gpu_executor)
    .add_sink(buffer_size=3)
)
pipeline = frontend.build(num_threads=1)
```

## Constraints

### Pickling (required for MTP subprocess mode)

Stage functions must be picklable. Use module-level functions with `functools.partial`, or callable classes with `__getstate__`/`__setstate__`. Never use lambdas or nested functions.

### Thread Safety

- HuggingFace tokenizers: NOT thread-safe — wrap in TLS (`threading.local`)
- tiktoken: thread-safe, no TLS needed
- `spdl.io`: thread-safe
- PyTorch/NumPy ops: generally thread-safe and GIL-releasing

### Concurrency Budget

Keep total CPU utilization ≤ 40% to avoid the noisy-neighbour effect on GPU kernel scheduling. Do not set concurrency higher than `num_CPU_cores / 8`.

## Training Loop

```python
for epoch in range(num_epochs):
    for batch in pipeline.get_iterator(timeout=900):
        train_step(batch)
```

Prefer `for batch in pipeline.get_iterator():` over manually calling `next(iterator)` — the manual approach requires extra `StopIteration` handling that complicates the code for no benefit.

## Checklist

- [ ] Decomposed monolithic `__getitem__`/iterator into stages by operation nature
- [ ] Each stage function releases GIL (or has concurrency=1 if unavoidable)
- [ ] Thread-unsafe libraries wrapped in TLS
- [ ] Media decoding uses `spdl.io` (not Pillow/TorchVision) where applicable
- [ ] Batching uses `.aggregate()` + explicit collate stage
- [ ] Production deployment uses MTP (subprocess for CPU, main process for GPU transfer)
- [ ] Stage functions are picklable (module-level or callable classes)
- [ ] Total concurrency respects CPU budget (≤ 40% utilization)
