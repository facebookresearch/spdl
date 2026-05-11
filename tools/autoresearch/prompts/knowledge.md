# SPDL Pipeline Optimization Knowledge

This knowledge is assembled from shared skill files. The canonical sources are in `spdl/tools/skills/pipeline_perf/`.

## Autoresearch-Specific Notes

### MTP — PipelineBuilder vs Pipeline vs PipelineConfig

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

#### Pipeline integration — multithreading only, no MTP

**GPU video decode is NOT compatible with MTP (subprocess mode)** because `VideoPackets` (the output of `demux_video`) are not picklable and cannot be passed across process boundaries.

Use pure multithreading in the main process instead:
- Build the entire pipeline (demux → decode_packets_nvdec → to_torch) in the main process using `PipelineBuilder`
- Use `.build(num_threads=N)` with appropriate thread count
- GPU decode releases the GIL during hardware decode, so multithreading works well
- Set decode stage concurrency to match GPU hardware decoder slots (7 for H100/B100; higher for sparse decoding patterns)
- **Use a dedicated `ThreadPoolExecutor` for the decode stage** — do NOT share the pipeline's default thread pool. Create `ThreadPoolExecutor(max_workers=C)` and pass it via `executor=` to the decode `.pipe()` call. This avoids contention between NVDEC decode threads (which need their own CUDA contexts) and CPU fetch/processing threads

`gpu_video_decode` and `subprocess_mtp` are **independent, mutually exclusive** optimization paths. They cannot be combined. GPU decode experiments must apply changes to the original instrumented pipeline, not on top of MTP changes. Compare their results independently to determine which approach yields better throughput.

### Headspace Analysis in the Loop

The autoresearch loop runs CacheDataLoader analysis automatically as part of the first iteration, alongside other experiments like MTP. CacheDataLoader results must NOT be treated as a real training run — they are excluded from the running best and plateau detection.
