# SPDL Pipeline Optimization Strategies

## How SPDL Pipelines Work

SPDL pipelines are async, multi-stage data processing pipelines for AI training. A pipeline is built with `PipelineBuilder`:

```python
pipeline = (
    PipelineBuilder()
    .add_source(sampler)
    .pipe(download, concurrency=16)    # I/O-bound: high concurrency
    .pipe(preprocess, concurrency=8)   # CPU-bound: moderate concurrency
    .aggregate(batch_size)
    .pipe(collate)
    .pipe(gpu_transfer)                # hardware-bound: no concurrency
    .add_sink(buffer_size=3)
    .build(num_threads=16)
)
```

### Stage Types
- **Source**: produces items (samplers, index iterators)
- **Pipe**: transforms items; `concurrency` controls how many run in parallel
- **Aggregate**: batches N items into one
- **Disaggregate**: unbatches one item into N
- **Sink**: output buffer for the consumer

### Key Parameters
| Parameter | Location | Guidance |
|---|---|---|
| `concurrency` | `.pipe(fn, concurrency=N)` | I/O stages: 16-32. CPU stages: 4-8. GPU transfer: 1 |
| `num_threads` | `.build(num_threads=N)` | Must be >= max concurrency of any sync stage |
| `buffer_size` | `.add_sink(N)` | 2-10; affects memory, NOT throughput |
| `output_order` | `.pipe(fn, output_order=...)` | "completion" for I/O (avoids head-of-line blocking), "input" for determinism |

### Async Stage Functions: Pass Them As-Is

`.pipe()` accepts both regular (`def`) and asynchronous (`async def`) callables. **Pass an async function directly — do not wrap it in `asyncio.run()`:**

```python
async def fetch(key):
    async with session.get(url(key)) as resp:
        return await resp.read()

pipeline = (
    PipelineBuilder()
    .add_source(source)
    .pipe(fetch, concurrency=32)   # CORRECT: coroutine function passed as-is
    ...
)
```

SPDL runs stages on its own event loop, so it awaits async stage functions natively and drives up to `concurrency` of them at once as cheap cooperative coroutines — ideal for I/O-bound work.

**Anti-pattern:** wrapping the coroutine so the stage becomes synchronous.

```python
# WRONG — asyncio.run() spins up and tears down a brand-new event loop on
# every single item. That per-item loop init/finalize is pure overhead, and
# it forces the work onto pipeline worker threads instead of running as cheap
# cooperative coroutines on SPDL's shared event loop.
.pipe(lambda key: asyncio.run(fetch(key)), concurrency=32)
```

If a stage is already synchronous, just pass it as a normal function — no event loop is involved either way.

## Recommended Architecture: Multi-Threading in Subprocess (MTP)

The production pattern runs the CPU-heavy pipeline in a **subprocess** and only does GPU transfer in the main process.

### MTP Structure

```python
import spdl.pipeline
import spdl.source.utils
from spdl.io import transfer_tensor
from spdl.pipeline import PipelineBuilder
from spdl.source import DistributedRandomSampler


def build_spdl_pipeline(
    samples, tokenizer, max_seq_len, batch_size,
    rank, world_size, num_threads, mp_context="forkserver",
):
    # embed_shuffle ensures correct sampling across iterations
    source = spdl.source.utils.embed_shuffle(
        DistributedRandomSampler(len(samples), rank=rank, world_size=world_size)
    )

    # Inner pipeline (subprocess) — all CPU work
    # Stage functions must be picklable — see "Pickling Constraints" section
    backend = (
        PipelineBuilder()
        .add_source(source, continuous=True)
        .pipe(lookup_fn)          # e.g., _Lookup(samples) or partial(_lookup, samples=samples)
        .pipe(tokenize_fn, concurrency=num_threads)  # must use TLS for HF tokenizers
        .aggregate(batch_size, drop_last=True)
        .pipe(collate_fn)
        .add_sink(buffer_size=3)
    )

    # Move to subprocess
    source2 = spdl.pipeline.run_pipeline_in_subprocess(
        backend.get_config(),
        num_threads=num_threads,
        mp_context=mp_context,
    )

    # Outer pipeline (main process) — GPU transfer only
    frontend = (
        PipelineBuilder()
        .add_source(source2, continuous=True)
        .pipe(transfer_tensor)
        .add_sink(buffer_size=3)
    )
    return frontend.build(num_threads=1)
    # NOTE: num_threads=1 is sufficient here because transfer_tensor is the
    # only stage and runs without concurrency.  When the frontend has
    # concurrent stages (e.g. NVDEC decode at concurrency=7), set
    # num_threads >= max stage concurrency (see the Key Parameters table).
```

### Training Loop with MTP

Build the dataloader **once** before the epoch loop. The Pipeline object supports re-iteration — each `for batch in dl:` triggers a fresh run inside the subprocess:

```python
dl = build_spdl_pipeline(
    samples=samples, tokenizer=tokenizer, max_seq_len=max_seq_len,
    batch_size=batch_size, rank=rank, world_size=world_size,
    num_threads=num_workers, mp_context=mp_context,
)

for epoch in range(num_epochs):
    for batch in dl.get_iterator(timeout=900):
        train_step(batch)
```

**Do NOT** restructure into a flat `next(data_iter)` loop — MTP's re-iteration pattern requires per-epoch `for ... in dl:`.

### MTP Best Practices

1. **`continuous=True`** on `.add_source()` — always use it. Eliminates pipeline teardown/rebuild between epochs. Harmless even when not strictly needed.
2. **`spdl.source.utils.embed_shuffle()`** — wrap `DistributedRandomSampler` with this for correct sampling behavior across iterations.
3. **`spdl.io.transfer_tensor`** — use for GPU transfer in the frontend pipeline. Use a dedicated thread (ThreadPoolExecutor with 1 worker) so that it uses own CUDA stream which allows overlapping data transfer and compute.
4. **`aggregate(batch_size, drop_last=True)`** — `drop_last=True` avoids partial batches that cause shape mismatches in DDP.
5. **`mp_context="forkserver"`** (the default). Try `"spawn"` and `"fork"` only if multiprocessing with "forkserver" fails.
6. **Do NOT use `output_order="completion"`** — only needed when output ordering matters (e.g., evaluation). For performance optimization pipelines, do not use it.

Benefits: isolates CPU work from GPU kernel launches, avoids the noisy-neighbour effect. Dedicated thread and CUDA stream for transferring the batch to GPU.

### Cut IPC CPU with a Shared-Memory Arena

MTP ships every batch across the subprocess boundary. By default the payload is pickled and copied through a multiprocessing queue, which costs CPU on **both** the producer and consumer sides, scaling with payload size. For large per-item payloads (NumPy arrays, Torch tensors, `bytes`, `spdl.io` `VideoPackets` — anything ≳ 1 MiB) this IPC CPU is itself a noisy-neighbour contributor.

Pass a shared-memory **arena** to `run_pipeline_in_subprocess` to move large binaries through shared memory by reference instead of pickling + copying them:

```python
from spdl.pipeline import SharedMemorySegmentPool, run_pipeline_in_subprocess

# segment_size: the largest single item; count: a few in-flight units (>= buffer_size + 2)
arena = SharedMemorySegmentPool(segment_size=64 << 20, count=8)
source2 = run_pipeline_in_subprocess(
    backend.get_config(), num_threads=num_threads, arena=arena
)
```

Two backends (both importable from `spdl.pipeline`):
- **`SharedMemorySegmentPool`** — the reader restores a zero-copy view directly over shared memory. Lowest CPU, and it **blocks** the producer when full (safe under a slow consumer). **Prefer this.**
- **`SharedMemoryRingBuffer`** — the reader copies each payload out of the ring. Still far cheaper than pickle, but it is **non-blocking**: it raises `shared-memory arena full` if its `capacity` is too small for the in-flight high-water, so size it generously (roughly `(buffer_size + 2) × max_item_bytes`).

Measured effect (recv-only, 32 MiB payloads): plain IPC spent several CPU-seconds per config pickling + copying, while the segment pool dropped that to ~0 (zero-copy) — simultaneously raising throughput and lowering peak memory. The win is largest for `bytes` / NumPy / `VideoPackets`; **Torch tensors already transfer via shared memory** (their multiprocessing reducer), so the arena helps them least.

**Why it matters for the CPU budget:** lower IPC CPU means more host-CPU headroom for timely GPU kernel launches — directly easing the noisy-neighbour effect (keep total CPU ≤ 40%). Reach for the arena whenever MTP moves large payloads; for tiny payloads the benefit is negligible, so skip it. See `examples/benchmark_arena_transport.py` for the benchmark this guidance is based on.

### When MTP Adds Overhead Instead of Helping

MTP introduces subprocess serialization and IPC overhead on every batch. This cost is fixed regardless of how much work the pipeline does per item. When per-item pipeline work is heavy (large decode, expensive preprocessing), MTP's isolation benefit far outweighs this overhead. But when per-item work becomes cheap — due to optimizations like subclipping (reducing decode work 10×), low-resolution inputs, lightweight preprocessing, or aggressive caching — the IPC overhead can become a significant fraction of the total per-batch time, and **running the pipeline in the main process with low concurrency can outperform MTP**.

**Rule of thumb:** if the pipeline's per-batch production time (source-to-sink) drops below ~50ms after optimization, benchmark with and without MTP. The noisy-neighbour effect that MTP prevents is only relevant when CPU utilization is high; if optimized stages use little CPU, MTP's isolation is unnecessary and its overhead hurts.

**Recommendation:** always test MTP as a best practice, but also test the best configuration *without* MTP. Report both results and let the data decide. Do not assume MTP is universally beneficial.

## The Noisy-Neighbour Effect

When CPU utilization exceeds ~40%, the OS cannot schedule GPU kernel launches promptly, causing GPU idle time even when data is available. **Keep total CPU utilization at or below 40%.** Using all CPUs for data loading is an anti-pattern.

## Python Garbage Collection Stalls

Python's GC can cause periodic spikes in step time by pausing threads. When data loading uses multi-threading in the main process, GC runs more frequently due to higher object allocation rates.

**Mitigation:**

Disable automatic GC and run it at a fixed step interval. This ensures all ranks pause at roughly the same time during distributed training, so only one step is affected instead of many:

```python
import gc
gc.disable()

for step, batch in enumerate(dataloader):
    train(batch)
    if step % gc_interval == 0:
        gc.collect()
```

If using TorchTNT, use `torchtnt.framework.callbacks.GarbageCollector(step_interval=N)` instead of manual GC management.

This technique is most impactful at large scale (many GPUs/nodes), where random GC pauses across ranks cause repeated synchronization delays. However, it can also help at smaller scale by improving cross-rank data readiness uniformity — random GC pauses can create a bimodal starvation pattern where some ranks are paused while others are ready, causing DDP synchronization jitter.

**Diagnosing GC stalls:** When step time spikes appear at irregular intervals (not correlated with epochs), GC is a likely cause. Another telltale sign is a **large discrepancy between p50 and p90/p99 latencies** in pipeline stage metrics — GC pauses affect only the unlucky iterations that coincide with a collection, inflating the tail latencies while leaving the median relatively unaffected. If you observe p90 ≫ p50 for a stage that should have stable per-item cost, suspect GC before investigating other causes.

## Why `next(dataloader)` Time Is Misleading

A common mistake is measuring `next(dataloader)` time alone to evaluate data loading performance. This can be deeply misleading due to GIL contention.

When data loading runs in multi-threaded mode in the main process (without MTP), GIL contention between data loading threads and the training loop can slow down model training itself (forward/backward/parameter update). The training step takes longer, which gives the data loader more time to catch up — so `next(dataloader)` superficially drops to near zero. The pipeline looks fast, but overall training is slower.

**What to check instead:**
1. **Training step time** (forward + backward + parameter update): compare against the baseline. If step time increased, GIL contention is the likely cause — even if `next(dataloader)` looks fine.
2. **Pipeline stats**: queue occupancy and throughput at each stage give an accurate picture independent of GIL effects.
3. **SM Utilization**: if it dropped despite `next(dataloader)` being fast, the GPU is starved due to delayed kernel launches from GIL contention.

This is a key reason to use MTP (subprocess mode) — it eliminates GIL contention between data loading and training entirely.

## System Metrics

- **SM Utilization (%)**: How active GPU streaming multiprocessors are. Target: ≥40%.
- **GPU Device Utilization (%)**: Overall GPU busy percentage.
- **CPU Utilization (%)**: Host CPU usage. Should stay ≤40%.

## Fixing Bottlenecks

1. **Adjust concurrency**: Usually 4-8 is enough. Never exceed `num_CPU_cores / 8`. Over-concurrency triggers noisy-neighbour.
2. **Split monolithic stages**: Separate I/O-bound, CPU-bound, and memory-bound operations into distinct `.pipe()` stages with independent concurrency.
3. **Move to subprocess**: Use `run_pipeline_in_subprocess()` to isolate CPU work. When it ships large payloads, also pass a shared-memory `arena` to cut the pickle/copy IPC CPU (see "Cut IPC CPU with a Shared-Memory Arena").
4. **Profile the function**: Look for GIL contention (use `spdl.io.load_npz` instead of numpy NPZ), unnecessary `asyncio.run()` wrappers (pass async functions directly), or expensive serialization.
5. **Use coalesced data access**: Batching I/O requests (reading multiple samples per call) can be significantly faster than single-sample access.
6. **Combine MTP with GPU decode**: `VideoPackets` are now picklable, so you can demux in a subprocess and decode with NVDEC in the main process. This combines the CPU-isolation benefits of MTP with the decode throughput of GPU hardware decoders. See the "GPU Video Decoding" section for details.

## Concurrency Tuning: Less Can Be More

For stages where the per-item operation is very fast (e.g., demuxing, which is mostly data copying), **reducing concurrency below the default can improve throughput**. When the operation itself is cheap, the overhead of thread scheduling and contention can dominate the actual work, and fewer concurrent tasks means less contention per item.

### Why lower concurrency can be faster

When multiple concurrent tasks share a thread pool (as in MTP subprocess pipelines), each task competes for thread scheduling, locks, and cache lines. For fast operations, this contention overhead is large relative to the work itself. Effective throughput (concurrency / per-item-latency) can decrease as concurrency rises:

```
Concurrency 8:  per-item = 0.13s  →  throughput = 8 / 0.13 = 62 items/s
Concurrency 6:  per-item = 0.065s →  throughput = 6 / 0.065 = 92 items/s  ← 48% faster
```

This effect is most pronounced for operations that are inherently fast (data copying, demuxing, disaggregation), where the operation itself is cheap but contention from concurrent scheduling inflates the observed per-item latency. For slow operations (network I/O, heavy CPU compute), higher concurrency is still beneficial because the parallelism gains outweigh the contention cost.

### Demux contention: FFmpeg's `codec_mutex`

Demuxing contention is not just generic thread scheduling overhead — there is a specific root cause inside FFmpeg. Every `Demuxer` construction calls `avformat_find_stream_info`, which probes each stream by opening a decoder via `avcodec_open2`. For codecs backed by external libraries (e.g. libvpx, libaom, libopus) whose thread-safety guarantees are unknown to FFmpeg, this initialization is guarded by a process-wide `codec_mutex` (`FF_CODEC_CAP_NOT_INIT_THREADSAFE` flag). At even modest concurrency (4+ threads), demux threads serialize on this lock and per-item latency rises sharply without improving throughput.

**Recommendation:** always split demuxing and decoding into separate pipeline stages with independent concurrency, and assign **at most 3 concurrency** to the demux stage. This allows the decode stage to use higher concurrency independently (or leverage NVDEC hardware decoders) without being bottlenecked by the demux lock:

```python
# Split demux and decode with independent concurrency
pipeline = (
    PipelineBuilder()
    .add_source(source)
    .pipe(fetch, concurrency=16)
    .pipe(demux, concurrency=3)          # at most 3 — codec_mutex contention
    .pipe(decode, concurrency=8)         # decode can scale independently
    .aggregate(batch_size)
    .pipe(collate)
    .add_sink(buffer_size=3)
)
```

### Search strategy

When tuning a stage's concurrency — especially for fast operations:

1. Start with the default (e.g., c=8)
2. Test higher values: c=10, c=12, c=16
3. **Also test lower values**: c=6, c=4 — particularly for stages whose underlying operation is fast (e.g., demuxing, data copying)
4. Monitor per-item latency (from pipeline stats) — if it increases faster than concurrency, you're in the contention regime
5. The sweet spot is where `concurrency / per_item_latency` is maximized

## Video Pipeline Optimization

### Subclipping: Truncate Before Decode

For video workloads where only a fraction of frames are needed (e.g., sampling 16 frames from a 10-second clip), **truncating the video to a short subclip before decoding** can dramatically reduce decode work.

```python
# Limit demuxed video to first 0.5 seconds before decoding
packets = spdl.io.demux_video(data, timestamp=(0, 0.5))
```

**Why this matters:**
- If a video is 10s at 30fps, that's ~300 frames, but temporal sampling may only need 16
- Decoding all frames then discarding most of them wastes decode compute proportional to the discard ratio
- This optimization is orthogonal to decoder choice (NvDec vs CPU) and concurrency tuning

**Tuning the subclip duration:**
- Set `subclip_duration` so that `fps × subclip_duration ≥ num_frames_needed`
- Going too short forces frame padding/repetition, which adds CPU overhead and may reduce quality
- The sweet spot depends on the specific fps and frame count requirements — aim for the shortest duration that still provides enough frames without excessive padding

**Multiple subclips from one video:** When videos are significantly longer than the subclip duration, you can extract multiple non-overlapping subclips from a single video to make better use of the downloaded data. Use `Demuxer.streaming_demux(duration=...)` to stream packets in chunks without re-instantiating the demuxer for each subclip. A generator pipeline stage will emit each chunk as a separate item downstream, amortizing the I/O cost of fetching the video across multiple training samples:

```python
def extract_subclips(video_data, subclip_duration=0.5):
    """Generator that yields subclip packets from one video in streaming fashion."""
    demuxer = spdl.io.Demuxer(video_data)
    for packets in demuxer.streaming_demux(
        demuxer.video_stream_index, duration=subclip_duration
    ):
        yield packets
```

### GPU Video Decoding with NvDec

When using NvDec hardware decoders, be aware that GPUs have a **fixed number of NvDec engines** (typically 7 on A100/H100). Setting decode concurrency above this number means tasks queue for hardware access:

```python
# Match hardware: 7 NvDec engines on A100/H100
# Use a dedicated thread pool for CUDA context management
exec = ThreadPoolExecutor(max_workers=7)

.pipe(nvdec_decode, executor=exec)
```

Higher concurrency values (e.g., c=16) still work — excess tasks wait for an engine — but waste thread resources on waiting. Setting concurrency to match the hardware count (c=7) is marginally more efficient.

**Always split demux and decode into separate stages** (see "Demux contention" above). Demuxing should be limited to 2-3 concurrency due to FFmpeg's `codec_mutex`, while decode — whether CPU or NvDec — can scale independently. Combining them in a single stage forces decode concurrency to match the demux bottleneck. With NvDec, the MTP pattern works particularly well because demuxed packets are picklable and can cross the subprocess boundary:

```python
# MTP backend (subprocess) — demux only, low concurrency
backend = (
    PipelineBuilder()
    .add_source(source, continuous=True)
    .pipe(fetch, concurrency=num_fetch_threads)
    .pipe(demux, concurrency=3)          # codec_mutex limits scaling
    .add_sink(buffer_size=3)
)
source2 = spdl.pipeline.run_pipeline_in_subprocess(backend.get_config(), ...)

# Frontend (main process) — NvDec decode at hardware concurrency
decode_exec = ThreadPoolExecutor(max_workers=7)
frontend = (
    PipelineBuilder()
    .add_source(source2, continuous=True)
    .pipe(nvdec_decode, concurrency=7, executor=decode_exec)
    .aggregate(batch_size)
    .pipe(collate)
    .add_sink(buffer_size=3)
)
```

### Video Decoder Thread Tuning

When the pipeline decodes video with `spdl.io.load_video` / `spdl.io.decode_packets`, the underlying FFmpeg decoder uses **a single thread by default**. This is a deliberate SPDL design choice — concurrency is expected to come from running many decoders in parallel at the pipeline level — but for some workloads (high-resolution video, low pipeline concurrency, or CPU headroom available) raising the per-decoder thread count is faster overall.

#### How to set it

Pass `decoder_options={"threads": "X"}` via `spdl.io.decode_config`:

```python
decode_config = spdl.io.decode_config(decoder_options={"threads": "2"})
frames = spdl.io.load_video(video_data, decode_config=decode_config)
```

Valid values for `X` (passed as a string):
- `"1"` — default. One decoder thread per call. Lowest CPU per video, best when pipeline concurrency is already saturating CPU.
- `"2"`, `"4"`, `"8"`, ... — fixed thread count per decoder. Higher = faster single-video decode at the cost of more CPU.
- `"0"` — let FFmpeg choose automatically. Often gives the **fastest** single-video decode but uses **the most CPU** (may grab many cores per decode).

#### The Trade-Off

Decoder threads multiply with pipeline concurrency for total CPU pressure:

```
effective_CPU_load ≈ pipe_concurrency × decoder_threads
```

- More decoder threads → lower per-video latency, higher per-video CPU.
- `"0"` is tempting because it benchmarks fastest in isolation, but in a real pipeline it can blow past the 40% CPU budget, trigger the noisy-neighbour effect, and slow training overall.
- The right value depends on resolution (4K benefits more from threads than SD), codec, pipeline `concurrency`, and total CPU budget.

#### Search Space for Autoresearch

When the pipeline contains a video decode stage, treat decoder threads as a tunable parameter alongside `concurrency`:

1. **Always try** `decoder_options={"threads": "1"}` (default) and `{"threads": "2"}` — frequently 2 threads is the sweet spot for H.264 at HD/4K.
2. **Explore** `{"threads": "4"}` and `{"threads": "0"}` (auto) when there is CPU headroom (CPU utilization well below 40%) or when pipeline concurrency is low.
3. **Co-tune** with pipe `concurrency`: when raising decoder threads, lower pipe `concurrency` proportionally to keep `concurrency × decoder_threads` within the CPU budget.
4. **Reject** configurations that push CPU utilization above 40% even if isolated throughput looks better — the noisy-neighbour effect will cancel the gains.
5. **Optimize for end-to-end metric** (step time / SM utilization / total throughput), not for single-decode latency. `"0"` often wins microbenchmarks and loses real pipelines.

See `examples/benchmark_video.py` (`load_video_with_config`) for a reference benchmark harness that sweeps decoder threads × worker counts × resolutions.

### NvDec vs CPU Decode

NvDec hardware decode is almost always faster than CPU (FFmpeg) decode for video pipelines. Even at small resolutions (112×112), NvDec decoded at ~0.6s/item while CPU FFmpeg took ~0.8s/item with 16 concurrent threads — and the CPU decode pushed CPU utilization to 60%, collapsing SM utilization from ~15% to 3.9% via the noisy-neighbour effect.

**When CPU decode might be considered:** only when NvDec engines are fully saturated AND the video is already subclipped to very short durations (where per-frame overhead dominates over decode throughput). Even then, empirical validation is essential.

## torch.compile

`torch.compile` can improve GPU compute efficiency by fusing kernels and reducing launch overhead. However, for short training runs (< 1000 steps), the compilation overhead may outweigh the gains — the model spends significant time in the compile phase before reaching steady-state throughput.

**Guidance for autoresearch:**
- `torch.compile` benefits are most visible when data loading is no longer the bottleneck (i.e., after pipeline optimization has reduced headspace significantly)
- Prefer `torch.compile(model)` without `fullgraph=True` — fullgraph mode is more fragile and frequently crashes with complex models
- When the job has short duration (e.g., 500 steps for benchmarking), the compile warmup cost can dominate; in this case, compare steady-state step times (excluding the first ~100 steps) rather than overall throughput

## Pickling Constraints (Subprocess Mode)

When using `run_pipeline_in_subprocess()`, all pipeline stage functions must be picklable because they are serialized and sent to the subprocess.

### What NOT to use

- **Lambda functions** — cannot be pickled.
- **Locally defined (inner/nested) functions** — not picklable.
- **Bound methods** (e.g., `dataset.__getitem__`, `self.process`) — while Python 3.5+ can pickle bound methods, prefer module-level functions with `functools.partial` instead (see Tier 1 pattern below). Module-level functions make the serialization boundary explicit, produce clearer error messages when pickling fails, and can be extended with lazy initialization or thread-local storage without modifying the original class.

### Two-Tier Approach for MTP Stage Functions

#### Tier 1 (try first): Module-level functions with `functools.partial`

The simplest approach. Define functions at module level, use `functools.partial` to bind picklable arguments. Re-create objects lazily in the subprocess when possible:

```python
from functools import partial

_tokenizer_tls = threading.local()

def _lookup(idx: int, samples: list[dict[str, str]]) -> dict[str, str]:
    return samples[idx]

def _tokenize(sample: dict, model_path: str, max_seq_len: int) -> dict:
    tok = getattr(_tokenizer_tls, 'tokenizer', None)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_path)
        _tokenizer_tls.tokenizer = tok
    # ... tokenize ...

pipeline.pipe(partial(_lookup, samples=samples))
pipeline.pipe(partial(_tokenize, model_path=model_path, max_seq_len=512), concurrency=8)
```

This works when the subprocess can successfully import and re-create the objects.

#### Tier 2 (fallback): Picklable callable classes

If Tier 1 fails (subprocess crashes on startup, silent 0-batch output), switch to **callable classes** that pickle objects directly from the main process, bypassing subprocess imports entirely:

```python
class _Lookup:
    def __init__(self, samples):
        self.samples = samples  # pickled directly

    def __call__(self, idx):
        return self.samples[idx]

class _Tokenize:
    def __init__(self, tokenizer, max_seq_len):
        self._tokenizer = tokenizer  # pickled directly — no re-import needed
        self.max_seq_len = max_seq_len
        self._tlt = _ThreadLocalTokenizer(tokenizer)

    def __getstate__(self):
        return {"tokenizer": self._tokenizer, "max_seq_len": self.max_seq_len}

    def __setstate__(self, state):
        self._tokenizer = state["tokenizer"]
        self.max_seq_len = state["max_seq_len"]
        self._tlt = _ThreadLocalTokenizer(self._tokenizer)

    def __call__(self, sample):
        return tokenize(sample, self._tlt.tokenizer, self.max_seq_len)

pipeline.pipe(_Lookup(samples))
pipeline.pipe(_Tokenize(tokenizer, 512), concurrency=8)
```

**When to use Tier 2**: Some libraries (notably HuggingFace `transformers`) can cause unusual errors when imported or when objects are re-created in a subprocess. If a subprocess pipeline crashes silently (0 batches, no error message), the most likely cause is a failed import or object re-creation. Switching to Tier 2 — where objects are pickled from the main process and unpickled in the subprocess — avoids the problematic import entirely.

**Always try Tier 1 first. If the job fails with subprocess-related issues, retry with Tier 2.**

### Thread Safety

**HuggingFace tokenizers are NOT thread-safe.** When using concurrent pipeline stages with HF tokenizers, you MUST use thread-local storage (TLS) to give each thread its own copy:

```python
class _ThreadLocalTokenizer(threading.local):
    def __init__(self, source):
        self._source = source
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import copy
            self._tokenizer = copy.deepcopy(self._source)
        return self._tokenizer
```

Note: OpenAI's `tiktoken` tokenizers ARE thread-safe and do not need TLS.

### What NOT to do

```python
# WRONG — all of these fail or cause subtle bugs:
pipeline.pipe(dataset.__getitem__, concurrency=8)            # bound method — works but opaque;
                                                             #   prefer module-level fn (see below)
pipeline.pipe(lambda x: tokenize(x, max_length=512))        # lambda — not picklable
pipeline.pipe(my_inner_function)                             # inner function — not picklable
pipeline.pipe(partial(fn, tls=threading.local()))            # non-picklable arg
pipeline.pipe(partial(tokenize, tokenizer=hf_tokenizer),     # shared HF tokenizer —
              concurrency=8)                                 #   not thread-safe, will corrupt
```

```python
# PREFERRED — module-level function + partial:
def _fetch(index: int, *, dataset: MyDataset) -> dict:
    return dataset[index]

pipeline.pipe(partial(_fetch, dataset=dataset), concurrency=8)
```

## Headspace Analysis

Before optimizing, estimate the maximum possible improvement using `CacheDataLoader`. It caches a small number of batches and returns them repeatedly, eliminating actual data loading overhead. The training accuracy is not meaningful, but the **step time** reflects the best-case performance with zero data loading cost.

```python
from spdl.dataloader import CacheDataLoader
dataloader = CacheDataLoader(dataloader, num_caches=10, return_caches_after=100)
```

**Key properties of CacheDataLoader runs:**
- The pipeline configuration (concurrency, MTP, batch size, etc.) **does not matter** — after the initial cache fill, all batches come from cache. The result is pure model compute time.
- The job duration and step time from a CacheDataLoader run represent the **lower bound** of achievable training time — you cannot beat it by optimizing data loading alone.

Compare cached vs uncached `step_time`:
- **`headspace = (baseline_step_time - cached_step_time) / baseline_step_time`**
- If headspace is small (<5%), data loading is NOT the bottleneck — focus on model optimization (torch.compile, CUDA graphs, etc.).
- If headspace is large (>10%), data loading optimization can yield significant gains.
- The headspace number is the **upper bound** of improvement from data loading optimization alone.

## Distributed Training Considerations

- All ranks synchronize at `nccl:all_reduce`. The slowest rank determines overall speed.
- Compare data readiness across ranks to find stragglers.
- Straggler rank can shift between epochs — indicates global data loading insufficiency.
- Rank 0 is more susceptible (extra work like logging, checkpointing).

## Parameter Tuning Strategy

When tuning parameters like concurrency, num_threads, batch_size:
1. **Start with exploration**: test a spread of values (e.g., concurrency in {4, 8, 16, 32})
2. **Search below the default too**: lower concurrency can outperform higher when thread contention dominates (see "Concurrency Tuning: Less Can Be More")
3. **Model the response**: from results so far, estimate which region of the parameter space yields best metrics
4. **Exploit the best region**: test values near the current best, with small perturbations
5. **Balance explore/exploit**: occasionally test values far from the best to avoid local optima
6. **Consider interactions**: parameters can interact (e.g., high concurrency + high num_threads may exceed CPU budget)
7. **Respect constraints**: total CPU usage ≤ 40%, concurrency ≤ num_CPU_cores / 8

### Batch Size Helps Even When Data-Loading-Bound

Increasing batch size can improve throughput even when data loading — not GPU compute — is the bottleneck. This is counter-intuitive: if the GPU is idle waiting for data, processing more samples per step shouldn't help. But larger batches amortize **per-step fixed costs** across more samples:

- **DDP synchronization**: each `all_reduce` has fixed latency regardless of batch size. Doubling batch size halves the number of syncs per sample.
- **Pipeline drain/refill cycles**: each step boundary causes a brief pipeline stall as the training loop consumes the batch and requests the next one. Fewer steps per epoch means fewer stalls.
- **CUDA kernel launch overhead**: fewer, larger kernels are more efficient than many small ones.

**Guidance:** when headspace is large (>50%), try doubling the batch size as an early experiment. It is a zero-code-change parameter that often yields 5-25% improvement. Test 2-3 batch sizes (e.g., 2×, 3×, 4× the default) and stop when throughput plateaus.
