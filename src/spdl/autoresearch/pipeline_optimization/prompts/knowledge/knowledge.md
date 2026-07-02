# SPDL Pipeline Optimization Knowledge

This knowledge is assembled from shared skill files. The canonical sources are in `spdl/tools/skills/pipeline_perf/`.

## Autoresearch-Specific Notes

### MTP — PipelineBuilder vs Pipeline vs PipelineConfig

MTP's main performance goal is to move CPU data loading work into a separate
process so background data loading threads do not interfere with CUDA kernel
launch scheduling in the training process. Avoiding GIL/GC interference is
another benefit. Even in free-threaded Python, which does not have GIL contention,
isolating CPU contention is still benefitial.

The SPDL pipeline API has three distinct types:

- **`PipelineBuilder`** — accumulates pipeline configuration in a monad pattern. Chain `.add_source()`, `.pipe()`, `.aggregate()`, `.add_sink()` on it. It does NOT run anything — it only builds up the config.
- **`PipelineConfig`** — a frozen, picklable dataclass describing the pipeline. Obtained via `builder.get_config()`. This is what `run_pipeline_in_subprocess()` needs.
- **`Pipeline`** — a facade for the running pipeline. Obtained via `builder.build(num_threads=N)` or `build_pipeline(config, num_threads=N)`. `Pipeline` is an **iterable** (not an iterator) — always use `pipeline.get_iterator(timeout=<seconds>)` to get an iterator with a timeout, rather than calling `iter(pipeline)` or iterating directly with `for batch in pipeline`.

**`run_pipeline_in_subprocess()` accepts ONLY `PipelineConfig`** (from `.get_config()`). It does NOT accept `Pipeline` objects or iterators. If existing code calls `.build()` and returns a `Pipeline` or iterator, it must be refactored to return a `PipelineConfig` instead for MTP to work.

**GPU stages (e.g. `transfer_tensor`) must NOT be in the subprocess pipeline.** They require a CUDA context which is unavailable in the subprocess. Split the pipeline: backend (subprocess) has CPU stages only; frontend (main process) applies GPU transfer.

### MTP Tier 1/Tier 2 Retry

When implementing MTP, always try Tier 1 (module-level functions with `functools.partial`) first. If the job fails with subprocess-related issues (crashes on startup, silent 0-batch output), retry with Tier 2 (picklable callable classes). The autoresearch loop should attempt both automatically.

### GPU Video Decoding (NVDEC)

When the pipeline bottleneck is CPU video decoding (FFmpeg), GPU video decoding via NVDEC can eliminate the bottleneck entirely by offloading decode to dedicated GPU hardware decoders.

**CPU decode pipeline:** `demux_video → decode_packets → convert_frames → transfer_buffer → to_torch`
**GPU decode pipeline:** `demux_video → decode_packets_nvdec → to_torch` (much simpler, already on GPU)

#### Key API: `spdl.io.decode_packets_nvdec`

```python
buffer = spdl.io.decode_packets_nvdec(
    packets,                    # VideoPackets from demux_video()
    device_config=spdl.io.cuda_config(
        device_index=device_index,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
    ),
    scale_width=width,          # hardware-accelerated resize (must be even)
    scale_height=height,        # hardware-accelerated resize (must be even)
    pix_fmt="rgb",              # auto-converts NV12 → RGB
)
tensor = spdl.io.to_torch(buffer)  # shape: [N, 3, H, W], already on GPU
```

#### What it replaces

When refactoring from CPU to GPU decode, replace this pattern:
```python
# CPU decode (remove all of this)
packets = spdl.io.demux_video(src, ...)
frames = spdl.io.decode_packets(packets, ...)     # CPU FFmpeg decode
buffer = spdl.io.convert_frames(frames)            # CPU pixel format conversion
buffer = spdl.io.transfer_buffer(buffer, ...)      # CPU → GPU transfer
tensor = spdl.io.to_torch(buffer)
```

With this:
```python
# GPU decode (replace with this)
packets = spdl.io.demux_video(src, ...)
buffer = spdl.io.decode_packets_nvdec(packets, device_config=cuda_cfg, pix_fmt="rgb")
tensor = spdl.io.to_torch(buffer)  # already on GPU, no transfer needed
```

Note: `decode_packets_nvdec` automatically applies bitstream filtering (h264_mp4toannexb / hevc_mp4toannexb), so remove any manual `apply_bsf()` calls if present.

#### Constraints

- **GPU decoder slots are hardware-dependent** — H100 and B100 have 7 NVDEC instances per GPU, so decode concurrency can be set around 7. If the decoding pattern is **sparse** (sampling a few frames from each file rather than decoding many sequential frames), concurrency can exceed 7 since each decode job finishes quickly and doesn't saturate a decoder for long. Older GPUs (A100, V100) have fewer slots (3–5). Do NOT set concurrency equal to CPU thread count.
- **Even dimensions required** — `scale_width` and `scale_height` must be even numbers.
- **CUDAConfig must use PyTorch allocator** — always pass `allocator=(torch.cuda.caching_allocator_alloc, torch.cuda.caching_allocator_delete)` to `cuda_config()` so tensors share PyTorch's memory pool.
- **No FFmpeg filters** — GPU decode does not support FFmpeg filter graphs. Resize/crop is done via NVDEC hardware parameters instead.
- **Decoder creation is expensive** (~300ms) — SPDL uses thread-local caching to amortize this. Avoid passing crop parameters unless needed (they bypass the cache).
- **SM utilization may appear lower** — GPU video decode uses dedicated NVDEC hardware engines, not SM (shader) cores. A pipeline using NVDEC may show lower SM utilization than a CPU-decode pipeline even though sample throughput is higher. Do NOT interpret lower SM utilization as a regression when using GPU decode — compare sample throughput (step time) instead.

#### Pipeline integration

GPU video decode supports two integration modes:

##### Option A: MTP + GPU decode (recommended for video pipelines)

`VideoPackets` are now picklable, so **MTP and GPU decode can be combined**. Demux in a subprocess and decode with NVDEC in the main process:

```python
from concurrent.futures import ThreadPoolExecutor

# Backend (subprocess) — fetch + demux only
backend = (
    PipelineBuilder()
    .add_source(source, continuous=True)
    .pipe(fetch_fn, concurrency=16)
    .pipe(demux_fn, concurrency=8)
    .add_sink(buffer_size=3)
)

source2 = spdl.pipeline.run_pipeline_in_subprocess(
    backend.get_config(),
    num_threads=8,
    mp_context="forkserver",
)

# Frontend (main process) — GPU decode + to_torch
# Use a dedicated executor for NVDEC decode (separate from pipeline thread pool)
decode_executor = ThreadPoolExecutor(max_workers=7)
frontend = (
    PipelineBuilder()
    .add_source(source2, continuous=True)
    .pipe(decode_nvdec_fn, concurrency=7, executor=decode_executor)
    .pipe(spdl.io.to_torch)
    .add_sink(buffer_size=3)
)
pipeline = frontend.build(num_threads=8)
```

This combines the CPU-isolation benefits of MTP (no noisy-neighbour effect from demux threads) with the decode throughput of GPU hardware decoders.

##### Option B: Pure multithreading (no MTP)

Build the entire pipeline (demux → decode_packets_nvdec → to_torch) in the main process using `PipelineBuilder`:
- Use `.build(num_threads=N)` with appropriate thread count
- GPU decode releases the GIL during hardware decode, so multithreading works well
- Set decode stage concurrency to match GPU hardware decoder slots (7 for H100/B100; higher for sparse decoding patterns)
- **Use a dedicated `ThreadPoolExecutor` for the decode stage** — do NOT share the pipeline's default thread pool. Create `ThreadPoolExecutor(max_workers=C)` and pass it via `executor=` to the decode `.pipe()` call. This avoids contention between NVDEC decode threads (which need their own CUDA contexts) and CPU fetch/processing threads


### CPU FFmpeg Video Decoding (reverse of GPU decode)

When the pipeline uses GPU NVDEC decode and you want to switch to CPU FFmpeg decode (e.g., to bypass the NVDEC hardware decoder slot limit), replace the NVDEC decode chain with CPU FFmpeg decode + GPU transfer.

**GPU decode pipeline (current):** `demux_video → decode_packets_nvdec → to_torch` (output on GPU)
**CPU decode pipeline (replacement):** `demux_video → decode_packets → convert_frames → to_torch → transfer_tensor` (output on CPU, then transferred to GPU)

#### Key APIs and their exact signatures

**`spdl.io.decode_packets`** — CPU FFmpeg decode:
```python
spdl.io.decode_packets(
    packets,                           # VideoPackets from demux_video()
    filter_desc: str | None = <default>,  # FFmpeg filter graph string
    decode_config: DecodeConfig | None = None,
    *,
    num_frames: int = -1,              # keyword-only
) -> VideoFrames
```

The `filter_desc` parameter controls pixel format conversion and scaling. By default it converts to `rgb24`. Use `spdl.io.get_video_filter_desc()` to construct filter strings with scaling:

```python
filter_desc = spdl.io.get_video_filter_desc(
    scale_width=112,
    scale_height=112,
    pix_fmt="rgb24",
)
frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
```

**`spdl.io.convert_frames`** — convert decoded frames to buffer:
```python
spdl.io.convert_frames(
    frames,                    # VideoFrames from decode_packets()
    storage: CPUStorage | None = None,
) -> CPUBuffer
```

**IMPORTANT**: `convert_frames` does NOT accept `filter_desc`. Filtering (scaling, pixel format) is done at `decode_packets` time via the `filter_desc` parameter. Passing `filter_desc` to `convert_frames` will raise a `TypeError`.

**Output tensor shape depends on pixel format:**
- **`pix_fmt="rgb24"`** (interleaved/packed) → `convert_frames` produces **`[N, H, W, C]`** (channel-last, C=3)
- **`pix_fmt="rgb"` via NVDEC** (planar) → `to_torch` produces **`[N, C, H, W]`** (channel-first, C=3)

This is a critical difference. When replacing NVDEC (`[N, C, H, W]`) with CPU FFmpeg decode using `rgb24` (`[N, H, W, C]`), you MUST add an extra permute step: `tensor = tensor.permute(0, 3, 1, 2)` to convert from channel-last to channel-first before applying the existing frame sampling and normalization logic.

**`spdl.io.to_torch`** — convert buffer to torch tensor:
```python
tensor = spdl.io.to_torch(buffer)  # CPU tensor, shape matches buffer layout
```

**`spdl.io.transfer_tensor`** — transfer CPU tensors to GPU:
```python
spdl.io.transfer_tensor(batch, /, *, num_caches: int = 4)
```

Handles nested structures (dict, list, tuple, dataclass). Uses `LOCAL_RANK` env var to determine target GPU device. Creates a dedicated CUDA stream per thread for overlapping transfer with compute.

#### Replacing NVDEC with CPU decode (in-place pattern)

The simplest approach is to modify the existing decode callable's `__call__` method. Replace the NVDEC decode body:

```python
# BEFORE (NVDEC):
buffer = spdl.io.decode_packets_nvdec(
    packets, device_config=self.cuda_cfg,
    scale_width=self.width, scale_height=self.height, pix_fmt="rgb",
)
tensor = spdl.io.to_torch(buffer)  # [T, C, H, W] on GPU

# AFTER (CPU FFmpeg):
filter_desc = spdl.io.get_video_filter_desc(
    scale_width=self.width, scale_height=self.height, pix_fmt="rgb24",
)
frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
buffer = spdl.io.convert_frames(frames)
tensor = spdl.io.to_torch(buffer)  # [T, H, W, C] on CPU (rgb24 is channel-last!)
tensor = tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W] to match NVDEC layout
```

After the decode callable, add GPU transfer. Since CPU decode produces CPU tensors, add `transfer_tensor` as a pipeline stage **after collate but before the sink**:

```python
.pipe(collate)
.pipe(spdl.io.transfer_tensor, executor=ThreadPoolExecutor(max_workers=1))
.add_sink(buffer_size=3)
```

Remove `cuda_config` setup code that was only used for NVDEC. Keep the `device` variable if the label tensor needs to be on GPU.

#### Common mistakes to avoid

1. **Do NOT pass `filter_desc` to `convert_frames()`** — it only accepts `(frames, storage=None)`.
2. **Do NOT forget GPU transfer** — CPU decode produces CPU tensors; add `transfer_tensor` stage.
3. **Do NOT forget the channel-last → channel-first permute** — `rgb24` produces `[N, H, W, C]` but the rest of the pipeline expects `[N, C, H, W]`. Add `tensor = tensor.permute(0, 3, 1, 2)` immediately after `to_torch`. Without this, the downstream frame sampling and permute logic operates on the wrong dimensions, raising shape errors.
4. **Do NOT use bare `except` to swallow errors** — `except RuntimeError: return None` or `except Exception: return None` hides bugs like shape mismatches, making debugging impossible. Always log the exception: `except RuntimeError as e: logging.getLogger(__name__).warning("Decode failed: %s", e); return None`. This way, if every sample fails, the pipeline stats show 100% failure rate and the analysis agent can diagnose the root cause from logs.
5. **Do NOT modify the wrong file** — if the decode callable is in a utility module (e.g., `utils/pipeline.py`) and the engine modifies the training script, define a new decode callable in the training script and override the pipeline construction call.
6. **Do NOT remove BUCK dependencies** — the training script needs its model/training deps (torchvision, DDP, etc.) even when changing the pipeline.

### Headspace Analysis in the Loop

The autoresearch loop runs CacheDataLoader analysis automatically as part of the first iteration, alongside other experiments like MTP. CacheDataLoader results must NOT be treated as a real training run — they are excluded from the running best and plateau detection.

### TorchTNT Integration Pattern

Many training scripts use TorchTNT (via `torchtnt.framework`) to manage the training loop. In TorchTNT-based code, the training loop is NOT a simple `for batch in pipeline:` — it is managed by `torchtnt.framework.fit()` or `torchtnt.framework.train()`, which call `iter(dataloader)` per epoch and `next(data_iter)` per step internally.

**How SPDL Pipeline integrates with TorchTNT:**

The recommended pattern is a wrapper class that builds the pipeline in its constructor and exposes `__iter__` and `__len__`:

1. Create a function that builds the pipeline (returns a `Pipeline` object).
2. In the wrapper class constructor, call the function and assign the pipeline to an instance attribute.
3. Add `__iter__` which calls `self.pipeline.get_iterator(timeout=...)`.
4. Add `__len__` which refers to the source object used when building the pipeline.

```python
class SPDLDataLoader:
    def __init__(self, source, batch_size, ...):
        self.source = source
        self.pipeline = build_pipeline(source, batch_size, ...)

    def __iter__(self):
        return self.pipeline.get_iterator(timeout=30)

    def __len__(self):
        return len(self.source) // self.batch_size
```

This wrapper is passed to TorchTNT's `fit()` or `train()` as the `train_dataloader` argument.

**Key facts about Pipeline:**
- `Pipeline` is both iterable and iterator. It can technically be directly iterated, but the wrapper pattern above is recommended because it provides `get_iterator(timeout=...)` for timeout handling and `__len__` for epoch length.
- **Preferred iteration**: Always use `pipeline.get_iterator(timeout=<seconds>)` to obtain an iterator with a timeout, so that jobs do not get stuck. Directly iterating with `for batch in pipeline:` or `iter(pipeline)` is discouraged because it lacks timeout handling.
- The pipeline is built once and iterated many times. `auto_stop()` is obsolete — do not call it, and do not rebuild per epoch.
- The `Pipeline` abstracts away whether it uses MTP (subprocess) or pure multithreading internally, so switching between them does not affect how the Pipeline is consumed.
- TorchTNT calls `iter(dataloader)` at the start of each epoch, then `next(data_iter)` per step until `StopIteration`.
- The pipeline's source controls how many items are produced per epoch. When the source is exhausted, the iterator raises `StopIteration`, ending the epoch in TorchTNT.

**Detecting TorchTNT usage in code:**
- Imports from `torchtnt.framework` (`fit`, `train`, `AutoUnit`, `TrainUnit`)
- A unit class extending `AutoUnit` or implementing `TrainUnit` with a `compute_loss()` or `train_step()` method
- A wrapper class with `__iter__` that calls `pipeline.get_iterator(timeout=...)`
- The `fit()` or `train()` call with `train_dataloader=` argument

**Where code changes target in TorchTNT scripts:**
- **Pipeline construction**: Inside the function that builds the `PipelineBuilder` and calls `.build()`. Same as non-TorchTNT scripts. The `Pipeline` abstracts MTP vs pure multithreading, so the wrapper class and TorchTNT internals do not change when switching backends.
- **Training step**: Inside the unit's `compute_loss()` or `train_step()` — equivalent to the training step in a simple loop.
- **Instrumentation**: Add TTFB timing inside the wrapper's `__iter__` using `enumerate` to detect the first batch and log init-to-steady-state transition, and/or use TorchTNT callbacks for step/fetch timing.
