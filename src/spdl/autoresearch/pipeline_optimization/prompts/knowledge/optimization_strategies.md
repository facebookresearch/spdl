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

## The Noisy-Neighbour Effect

When CPU utilization exceeds ~40%, the OS cannot schedule GPU kernel launches promptly, causing GPU idle time even when data is available. **Keep total CPU utilization at or below 40%.** Using all CPUs for data loading is an anti-pattern.

## Python Garbage Collection Stalls

Python's GC can cause periodic spikes in step time by pausing threads. When data loading uses multi-threading in the main process, GC runs more frequently due to higher object allocation rates.

**Mitigation by scale:**
- **Medium scale**: Using MTP (subprocess architecture) naturally reduces GC impact in the main process, since most object-heavy work runs in the subprocess.
- **Large scale** (many GPUs/nodes): In distributed training, all ranks synchronize at `nccl:all_reduce` — the slowest rank determines the step time. GC is inevitable and must happen occasionally, but if ranks trigger GC at random times, multiple training steps are slowed down (each time a different rank pauses). By aligning GC to a fixed step interval, all ranks pause at roughly the same time, so only one step is affected instead of many. This effect is most visible in long-running jobs with many distributed nodes — for short training runs or small-scale setups the impact is negligible.

```python
import gc
gc.disable()

for step, batch in enumerate(dataloader):
    train(batch)
    if step % gc_interval == 0:
        gc.collect()
```

If using TorchTNT, use `torchtnt.framework.callbacks.GarbageCollector(step_interval=N)` instead of manual GC management.

When diagnosing step time spikes that appear at irregular intervals (not correlated with epochs), GC is a likely cause.

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
3. **Move to subprocess**: Use `run_pipeline_in_subprocess()` to isolate CPU work.
4. **Profile the function**: Look for GIL contention (use `spdl.io.load_npz` instead of numpy NPZ), unnecessary `asyncio.run()` wrappers (pass async functions directly), or expensive serialization.
5. **Use coalesced data access**: Batching I/O requests (reading multiple samples per call) can be significantly faster than single-sample access.
6. **Combine MTP with GPU decode**: `VideoPackets` are now picklable, so you can demux in a subprocess and decode with NVDEC in the main process. This combines the CPU-isolation benefits of MTP with the decode throughput of GPU hardware decoders. See the "GPU Video Decoding" section for details.

## Video Decoder Thread Tuning

When the pipeline decodes video with `spdl.io.load_video` / `spdl.io.decode_packets`, the underlying FFmpeg decoder uses **a single thread by default**. This is a deliberate SPDL design choice — concurrency is expected to come from running many decoders in parallel at the pipeline level — but for some workloads (high-resolution video, low pipeline concurrency, or CPU headroom available) raising the per-decoder thread count is faster overall.

### How to set it

Pass `decoder_options={"threads": "X"}` via `spdl.io.decode_config`:

```python
decode_config = spdl.io.decode_config(decoder_options={"threads": "2"})
frames = spdl.io.load_video(video_data, decode_config=decode_config)
```

Valid values for `X` (passed as a string):
- `"1"` — default. One decoder thread per call. Lowest CPU per video, best when pipeline concurrency is already saturating CPU.
- `"2"`, `"4"`, `"8"`, ... — fixed thread count per decoder. Higher = faster single-video decode at the cost of more CPU.
- `"0"` — let FFmpeg choose automatically. Often gives the **fastest** single-video decode but uses **the most CPU** (may grab many cores per decode).

### The Trade-Off

Decoder threads multiply with pipeline concurrency for total CPU pressure:

```
effective_CPU_load ≈ pipe_concurrency × decoder_threads
```

- More decoder threads → lower per-video latency, higher per-video CPU.
- `"0"` is tempting because it benchmarks fastest in isolation, but in a real pipeline it can blow past the 40% CPU budget, trigger the noisy-neighbour effect, and slow training overall.
- The right value depends on resolution (4K benefits more from threads than SD), codec, pipeline `concurrency`, and total CPU budget.

### Search Space for Autoresearch

When the pipeline contains a video decode stage, treat decoder threads as a tunable parameter alongside `concurrency`:

1. **Always try** `decoder_options={"threads": "1"}` (default) and `{"threads": "2"}` — frequently 2 threads is the sweet spot for H.264 at HD/4K.
2. **Explore** `{"threads": "4"}` and `{"threads": "0"}` (auto) when there is CPU headroom (CPU utilization well below 40%) or when pipeline concurrency is low.
3. **Co-tune** with pipe `concurrency`: when raising decoder threads, lower pipe `concurrency` proportionally to keep `concurrency × decoder_threads` within the CPU budget.
4. **Reject** configurations that push CPU utilization above 40% even if isolated throughput looks better — the noisy-neighbour effect will cancel the gains.
5. **Optimize for end-to-end metric** (step time / SM utilization / total throughput), not for single-decode latency. `"0"` often wins microbenchmarks and loses real pipelines.

See `examples/benchmark_video.py` (`load_video_with_config`) for a reference benchmark harness that sweeps decoder threads × worker counts × resolutions.

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
2. **Model the response**: from results so far, estimate which region of the parameter space yields best metrics
3. **Exploit the best region**: test values near the current best, with small perturbations
4. **Balance explore/exploit**: occasionally test values far from the best to avoid local optima
5. **Consider interactions**: parameters can interact (e.g., high concurrency + high num_threads may exceed CPU budget)
6. **Respect constraints**: total CPU usage ≤ 40%, concurrency ≤ num_CPU_cores / 8
