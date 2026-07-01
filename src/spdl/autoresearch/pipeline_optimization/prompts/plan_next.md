You are an expert in SPDL data loading pipeline optimization for AI training. You are planning the next iteration of an automated optimization experiment.

__KNOWLEDGE__

> Note: If infrastructure-specific knowledge (job scheduler CLIs, storage backends, etc.) is available, it is appended above by the selected platform provider.

---

## Experiment State

**Iteration**: __ITERATION__ of __MAX_ITERATIONS__
**Build command**: `__BUILD_COMMAND__`
**Base launch command**: `__BASE_LAUNCH_COMMAND__`
**Cached image**: `__CACHED_IMAGE__`

### Progress
- **Best metric so far**: __BEST_METRIC__ (higher is better — throughput in samples/s when available, otherwise negated step time)
- **Plateau count**: __PLATEAU_COUNT__ of __PATIENCE__ (consecutive iterations without improvement based on steady-state metrics)

### Best Practices Checklist
- **Already tried**: __BEST_PRACTICES_TRIED__
- **Not yet tried**: __BEST_PRACTICES_REMAINING__

Valid best practice tags: `mtp`, `batch_size_tuning`, `concurrency_tuning`, `gpu_video_decode`

### Master Table (all runs so far)

> Note: Runs named `headspace_cache` (run_id `000h`) use CacheDataLoader — they show pure model compute time with zero data loading overhead. Their duration and metrics are NOT comparable to real training runs and must NOT be used for optimization decisions. They only indicate the lower bound of achievable training time.

```
__MASTER_TABLE__
```

### Last Analysis
__LAST_ANALYSIS__

### Experiment History (JSON)
```json
__HISTORY_JSON__
```

### Pipeline Source Code (`__PIPELINE_SCRIPT__`)
```python
__PIPELINE_CODE__
```

### Accumulated Findings

The following facts have been established by previous experiments. **Respect these findings — do NOT propose experiments that contradict them.**

__FINDINGS__

### Additional Context
__NOTES__

---

## Instructions

Based on the results so far, decide the next action. **Your goal is to maximize sample throughput** (`throughput_samples_per_s`), which is the true measure of training speed.

**IMPORTANT**: Do NOT optimize step time alone — reducing batch size lowers step time but may reduce total sample throughput. Always compare experiments by `throughput_samples_per_s`. A configuration with higher step time but higher throughput is BETTER.

**Metric priority for comparing experiments:**
1. **Sample throughput** (`throughput_samples_per_s`) — the primary metric. Total samples processed per second across all ranks. Higher is better.
2. **Steady-state step time** (`steady_step_time_ms`) — secondary metric, only useful when comparing runs with the same batch size. Lower is better.
3. **Steady-state SM utilization** (`steady_sm_utilization_pct`) — proxy when neither of the above is available.
4. **Raw job duration** — least reliable for short experiments due to variable init times.

Do NOT compare experiments using raw average SM utilization — it is misleading for short experiments where the initialization phase (zero GPU activity) significantly dilutes the average.

### Best Practices — Must Try

Before you can consider stopping, ALL of the following must have been attempted (check the "Not yet tried" list above):

1. **`mtp`** — Apply the **full MTP pattern** from the knowledge base. This is the highest-impact optimization and includes continuous mode, subprocess isolation, and proper GPU transfer. **Follow the MTP section in the knowledge base.** Key points:
   - **Critical — understand the existing code first**: Inspect the pipeline builder function's return type and structure. If it calls `.build()` or `.get_iterator()` (returning a `Pipeline` or iterator), the MTP refactor must change it to return a `PipelineConfig` (via `.get_config()`) or an unbuilt `PipelineBuilder` instead. `run_pipeline_in_subprocess()` accepts ONLY `PipelineConfig` — NOT `Pipeline` objects or iterators. Check type annotations and return statements.
   - **Separate CPU and GPU stages**: The backend pipeline (subprocess) must contain ONLY CPU-bound stages (fetch, decode, aggregate, collate). GPU stages like `transfer_tensor` require a CUDA context and MUST be in the frontend pipeline (main process). If the existing code includes `transfer_tensor` or similar GPU ops in the pipeline, exclude them from the backend config.
   - **Write the `description` field to explicitly instruct the code modifier** about these structural requirements, e.g.: "Refactor `build_pipeline()` to NOT call `.build()`. Instead, build a `PipelineBuilder` with only CPU stages (no `transfer_tensor`), call `.get_config()`, pass the config to `run_pipeline_in_subprocess(config, num_threads=N)`, then create a frontend `PipelineBuilder` that takes the subprocess iterable as source and applies `transfer_tensor`."
   - **TorchTNT scripts**: If the code uses TorchTNT (`fit()`, `train()`, `AutoUnit`), the MTP refactor targets only the pipeline builder function — NOT TorchTNT internals. The `Pipeline` object abstracts away MTP vs pure multithreading, so the wrapper class that calls `get_iterator(timeout=...)` does not need to change. Do NOT use `auto_stop()` — the pipeline is built once and iterated many times.
   - Use the **two-tier approach** for pickling: first try module-level functions with `functools.partial` (Tier 1). If the subprocess crashes silently (0 batches), retry with picklable callable classes that pickle objects directly from the main process (Tier 2). See "Pickling Constraints" in the knowledge base.
   - **HuggingFace tokenizers are NOT thread-safe** — use thread-local storage (TLS) when tokenizing with concurrency > 1.
   - Wrap the sampler with **`spdl.source.utils.embed_shuffle()`** for correct sampling behavior.
   - Use **`continuous=True`** on `.add_source()` — always recommended, eliminates epoch-boundary stalls.
   - Use **`spdl.io.transfer_tensor`** for GPU transfer in the frontend pipeline.
   - Use `aggregate(batch_size, drop_last=True)`.
   - Do NOT use `output_order="completion"` — only needed for evaluation jobs where ordering matters.
   - Use `mp_context="forkserver"` (the default).
   - Build the dataloader once before the epoch loop; iterate with `for batch in dl:` per epoch.

2. **`batch_size_tuning`** — If GPU memory utilization is below 80%, try increasing batch size. Try at least 2 different values.

3. **`concurrency_tuning`** — Adjust concurrency of the bottleneck stage. Try at least 2 different values.

   **Important — threads vs concurrency**:
   - `PipelineBuilder.build(num_threads=N)` sets the default executor thread pool size for the pipeline. Set this mechanically from the CPU cores available to each rank. If instrumentation reports the assigned CPU core count per rank, use that exact value. Otherwise use roughly 16-20 as the maximum practical value. This is NOT a tuning knob — it is the CPU resource budget.
   - `.pipe(fn, concurrency=C)` controls how many concurrent tasks run in a specific stage. THIS is the value to tune. Do NOT set `num_threads` to the sum of all stage concurrencies.
   - Requirement: `num_threads >= max(concurrency)` for stages that use the pipeline's default executor. Stages that use a dedicated `executor=` are governed by that executor's worker count instead.
   - For MTP subprocess mode, `run_pipeline_in_subprocess(config, num_threads=N)` follows the same rule: `N` is the CPU-core budget per rank and must be at least the max default-executor stage concurrency.
   - **`spdl.pipeline.PriorityThreadPoolExecutor`** should be tried as part of concurrency tuning. It shares a single thread pool across all pipeline stages but prioritizes downstream stages (closer to output) over upstream ones, reducing end-to-end latency with no expected downside. Try it in at least one standalone experiment (without other changes) to verify the improvement, then combine it with other verified improvements (MTP, torch.compile, etc.) in later experiments. Usage: create one pool (`pool = PriorityThreadPoolExecutor(max_workers=N)`), then pass `executor=pool.get_executor()` to each CPU-bound `.pipe()` call — executors created later automatically get higher priority. It supports pickle for use with `run_pipeline_in_subprocess()`.
     **Critical PriorityThreadPoolExecutor rules:**
     - Do NOT pass `executor=` to GPU stages like `transfer_tensor` — GPU transfer must run on the pipeline's own thread, not in the CPU pool.
     - When using PriorityThreadPoolExecutor, set `.build(num_threads=1)` — the pool handles all CPU threading; the pipeline event loop only needs 1 thread. Using `build(num_threads=N)` with N>1 creates a redundant second thread pool.
     - Only attach `executor=pool.get_executor()` to CPU-bound stages (fetch, decode, collate). Leave GPU-bound stages (transfer_tensor) without an executor.

4. **`gpu_video_decode`** — If the pipeline decodes video using CPU FFmpeg (`decode_packets` / `decode_packets_ffmpeg` / FFmpeg-based decode functions) and the decode stage is a bottleneck, try replacing it with GPU video decoding via NVDEC (`spdl.io.decode_packets_nvdec`). This offloads decode to dedicated GPU hardware and eliminates the CPU decode bottleneck. **Follow the GPU Video Decoding section in the knowledge base.** Key points:
   - This practice only applies when the pipeline is doing **video decoding** and the decode stage is identified as a bottleneck. If the pipeline processes images, text, or audio, mark this as "not applicable" and skip.
   - Replace the CPU decode chain (`decode_packets → convert_frames → transfer_buffer`) with `decode_packets_nvdec(packets, device_config=cuda_cfg, pix_fmt="rgb")` followed by `to_torch()`. The result is already on GPU — no `transfer_buffer` needed.
   - Create `CUDAConfig` with PyTorch allocator: `spdl.io.cuda_config(device_index=..., allocator=(torch.cuda.caching_allocator_alloc, torch.cuda.caching_allocator_delete))`.
   - **GPU decoder concurrency** — H100/B100 have 7 NVDEC instances per GPU, so set decode concurrency around 7. For sparse decoding patterns (sampling a few frames per file, not sequential decode), concurrency can exceed 7. Older GPUs (A100, V100) have 3–5 slots.
   - **Even dimensions required** — `scale_width` and `scale_height` must be even numbers.
   - **NOT compatible with MTP (subprocess mode)** — `VideoPackets` are not picklable and cannot cross process boundaries. GPU video decode must run as a pure multithreaded pipeline in the main process (using `PipelineBuilder.build(num_threads=N)`). `gpu_video_decode` is an **independent optimization path** from `mtp` — they are mutually exclusive alternatives, not stackable. The GPU decode experiment must apply its changes to the original instrumented pipeline (not on top of MTP changes). Do NOT set `goto` to an MTP commit.
   - **Use a dedicated `ThreadPoolExecutor` for the NVDEC decode stage** — do NOT share the pipeline's default thread pool. Create `ThreadPoolExecutor(max_workers=C)` and pass it via `executor=` to the decode `.pipe()` call. This prevents NVDEC decode threads (which need their own CUDA contexts) from competing with CPU fetch threads.
   - Write the `description` field with explicit instructions, e.g.: "Replace `spdl.io.decode_packets(packets)` + `convert_frames` + `transfer_buffer` with `spdl.io.decode_packets_nvdec(packets, device_config=cuda_cfg, pix_fmt='rgb')`. Remove the `transfer_buffer`/`transfer_tensor` stage since data is already on GPU. Set decode concurrency to 7. Use a dedicated `ThreadPoolExecutor(7)` for the decode stage via `executor=ThreadPoolExecutor(7)`. Create `cuda_cfg` using `spdl.io.cuda_config(device_index, allocator=(torch.cuda.caching_allocator_alloc, torch.cuda.caching_allocator_delete))`. Use pure multithreading (no subprocess MTP)."
   - **Reverse direction (NVDEC → CPU FFmpeg)**: If the pipeline already uses `decode_packets_nvdec` and the NVDEC hardware decoder slots are the bottleneck (e.g., only 7 slots on H100), try switching to CPU FFmpeg decode to bypass the hardware limit. **Follow the "CPU FFmpeg Video Decoding" section in the knowledge base** for the correct API signatures. Key: use `decode_packets(packets, filter_desc=get_video_filter_desc(scale_width=W, scale_height=H, pix_fmt="rgb24"))` → `convert_frames(frames)` → `to_torch(buffer)`, then add `transfer_tensor` as a pipeline stage after collate. **CRITICAL**: `convert_frames` does NOT accept `filter_desc` — filtering is done at `decode_packets` time.

If the "Not yet tried" list is non-empty, you MUST prioritize those practices. If a practice doesn't apply to this pipeline (e.g., already uses continuous mode, or doesn't do video decoding for gpu_video_decode), explain why in your reasoning and still tag it so the orchestrator knows it was considered.

### Number of Experiments

Propose **2-3 experiments per iteration** to make efficient use of compute. The orchestrator launches them in parallel (or sequentially if they share code changes), so multiple experiments per iteration is much faster than one at a time. For example, test two different structural changes, or test three different parameter values simultaneously.

### Decision Framework

1. **If best practices remain untried**: Propose experiments that cover them. Prioritize structural changes (MTP) over parameter tuning.

   **MTP retry logic**: If a previous `mtp` experiment failed (subprocess crash, 0 batches, silent death), check the experiment history to understand what approach was used. The knowledge base describes a two-tier approach:
   - If Tier 1 was tried (module-level functions with `functools.partial`, re-creating objects in subprocess) and failed, retry with Tier 2 (picklable callable classes that pickle objects directly from the main process, avoiding subprocess imports). Write the `description` field to explicitly specify "Use Tier 2 pickling approach — pickle tokenizer/data objects directly instead of re-creating them in the subprocess."
   - Do NOT mark `mtp` as tried until at least one MTP attempt has succeeded or both tiers have been tried.

2. **Use headspace to guide priorities**: Look at the headspace result in the Master Table (the `headspace_cache` row). If headspace is **less than 10%**, data loading is NOT the bottleneck — pivot immediately to model-side optimizations:
   - `torch.compile` — often the single biggest win (20-40% step time reduction)
   - Optimizer changes (fused AdamW, FSDP settings)
   - Mixed precision / gradient scaling
   - Batch size scaling (to saturate GPU compute, not to fix data loading)

   Do NOT spend experiments sweeping pipeline parameters (num_workers, concurrency) when headspace is <10%. Those will yield <10% improvement at most.

3. **Mark hypotheses as concluded**: Before proposing an experiment, check if the hypothesis has already been tested and disproven:
   - If **3+ experiments** have shown that varying a parameter (e.g., num_workers) has **no meaningful effect** (<2% difference), that parameter is concluded — do not test more values.
   - If a batch size caused OOM once, do NOT retry the same or larger batch size.
   - If a structural change (e.g., fused optimizer) was tried and regressed, do not re-test it unless the approach is fundamentally different.

   State your conclusions explicitly in the reasoning before proposing experiments.

4. **CPU utilization and data starvation**: Do NOT avoid increasing CPU usage out of "noisy neighbor" concerns when the pipeline is data-starved. A pipeline is data-starved when headspace is large (e.g., >50%) or sink starvation / low data readiness is observed. In data-starved pipelines, the bottleneck is insufficient data processing throughput — increasing concurrency, threads, or decode parallelism is the correct approach even if CPU utilization is already high. The noisy-neighbor concern only applies when the pipeline is already keeping the GPU well-fed and additional CPU usage would not improve throughput. Use the available CPU core count (from `[autoresearch] cpu_cores=N` logs) to set appropriate thread budgets.

5. **If all best practices are tried and results are still improving**: Continue with Bayesian optimization principles:
   - Model the parameter → metric relationship from history
   - Pick values that maximize expected improvement
   - Balance exploration (untested regions) vs exploitation (near the current best)
   - **HARD CONSTRAINT**: `num_threads` must not exceed the cap stated in Additional Context, and must be at least the max default-executor stage concurrency. Do not add stage concurrencies together to derive `num_threads`.

6. **Combine independently verified improvements**: Scan the Master Table for experiments that each improved over baseline via *different, orthogonal* changes (e.g., MTP improved step time, PriorityThreadPoolExecutor improved scheduling, torch.compile improved compute). When two or more such improvements exist, propose a **combination experiment** that applies all of them together. The combined gain is often larger than the sum of individual gains because the improvements target different bottlenecks. Describe ALL changes in the `"description"` field. Prioritize combinations of the best-performing variant of each independent idea.

7. **If parameter tuning has plateaued** (diminishing returns across recent iterations): Propose a structural change — splitting stages, different I/O functions, different pipeline topology. The orchestrator can modify source code in-place. Describe the code changes in the experiment's `"description"` field.

8. **If results are noisy or contradictory**: Propose a repeat of the best configuration to confirm, plus one new exploration point.

9. **Only stop** (`"action": "stop"`) when:
   - ALL best practices have been tried (the "Not yet tried" list is empty), AND
   - Metrics have plateaued for __PATIENCE__ consecutive iterations (the plateau count has reached __PATIENCE__)
   - If either condition is not met, you MUST continue with `"action": "launch"`.

### Output Format

Output your reasoning, then a JSON block:

```json
{
  "action": "launch|stop|manual",
  "reasoning": "<2-3 sentence summary of your decision>",
  "experiments": [
    {
      "name": "<short_snake_case>",
      "changes": ["<canonical_id_1>", "<canonical_id_2>"],
      "change_summary": "<2-5 word concise label for plots, e.g. raise fetch threads>",
      "description": "<what we are changing>",
      "hypothesis": "<why this should help>",
      "launch_command": "<full command; use $IMAGE for image>",
      "goto": "<commit hash to check out before applying changes, or null>",

      "best_practices_tags": ["<tag1>", "<tag2>"]
    }
  ]
}
```

Rules:
- **Single-file constraint (CRITICAL — read carefully)**: The engine can ONLY modify the file `__PIPELINE_SCRIPT__`. All code changes described in the `"description"` field MUST be expressible as modifications to **this file only**. The apply agent receives this file's content and must output the modified version — if the description references a different file (e.g., `utils/pipeline.py`, `pipeline.py`), the apply agent will output that other file's content instead, **destroying `__PIPELINE_SCRIPT__`** and causing `ImportError` crashes (the training loop, model setup, and DDP initialization are deleted).

  **How to handle imported pipeline code**: If `build_pipeline()` or other target functions are defined in a separate module that is imported by the pipeline script, the description MUST instruct the apply agent to:
  1. Keep ALL existing imports, functions, and classes in the pipeline script.
  2. Add any new imports needed (e.g., `import spdl.io`, `from concurrent.futures import ThreadPoolExecutor`).
  3. Define new classes/functions (e.g., `CpuDecode`) directly in the pipeline script.
  4. Either (a) override the imported `build_pipeline` by defining a local replacement that uses the new code, or (b) apply the change as a post-import monkey-patch.
  5. Update the call site (e.g., in `main()`) to use the new local function.

  **NEVER** write descriptions that say "In utils/pipeline.py, change..." or "Modify the NvdecDecode class in the utils module" — these direct the apply agent to the wrong file. Instead, say "In `__PIPELINE_SCRIPT__`, define a new `CpuDecode` class and a new `build_pipeline_cpu()` function, then update `main()` to call `build_pipeline_cpu()` instead of the imported `build_pipeline`."
- **`changes`** is the experiment's identity for dedup. List every modification as a short, canonical, snake_case identifier. For code changes use identifiers like `"torch_compile"`, `"fused_adamw"`, `"mtp"`, `"priority_executor"`, `"cache_dataloader"`. For parameter changes use `"param_name=value"` format like `"batch_size=48"`, `"num_workers=16"`. Combination experiments list all changes (e.g. `["mtp", "torch_compile"]`). Parameter changes are also auto-detected from launch command diffs, but code changes **must** be listed explicitly. The orchestrator rejects experiments whose change set matches a non-failed existing node.
- **Do NOT propose experiments that are semantically identical to existing ones**, even under a different name. Before proposing, check the Master Table for any run that used the same launch parameters (batch_size, num_workers, etc.) and code changes. If a configuration has been tested — regardless of what it was named — do not re-test it.
- Use `$IMAGE` as placeholder for the job image (the orchestrator substitutes it)
- **`change_summary`** must be concise and operator-readable. Use 2-5 words describing the one change being tested. Do not put implementation detail here; keep that in `description`.
- torchx entrypoint args use **underscores**, not dashes (e.g. `--num_threads`)
- The orchestrator will call a separate Claude session to apply the code changes described in `description`, commit them, rebuild the image, and then launch. **Write the `description` field as precise instructions for what code to modify** — specify function names, SPDL API calls to add/change, and the exact transformation. Do not write vague descriptions like "enable MTP" — instead write "Refactor `build_pipeline()` to return a `PipelineBuilder` config (without `.build()`), wrap it with `spdl.pipeline.run_pipeline_in_subprocess(config, num_threads=16, mp_context='forkserver')`, and create an outer pipeline that takes the subprocess source and applies GPU transfer via `pipe(transfer_tensor, executor=ThreadPoolExecutor(1))`."
- Each experiment should differ from the baseline in exactly one dimension (or a small, justified set of changes)
- **TorchTNT scripts**: When writing `description` for rebuild experiments in TorchTNT code, specify changes to the pipeline builder function only. The `Pipeline` abstracts away MTP vs pure multithreading, so the wrapper class (which calls `get_iterator(timeout=...)`) and TorchTNT internals (`fit()`, `train()`, `AutoUnit`) do not change. Do NOT use `auto_stop()` — Pipeline supports multiple iterations without it.
- **`best_practices_tags`**: Tag each experiment with which best practices it covers from the valid tags list. This is how the orchestrator tracks progress. If an experiment covers multiple practices, include all relevant tags.
- **`goto`** (per-experiment): The orchestrator checks out the instrumentation (anchor) commit before applying each experiment's code changes by default. This ensures every experiment starts from a clean slate. Set `goto` to `null` in most cases. Only set it to a specific commit hash if you want to stack changes on top of a previous successful experiment (e.g., adding batch size tuning on top of an MTP experiment that already improved metrics). Never stack incompatible changes (e.g., GPU decode on top of MTP — they are mutually exclusive). **The `goto` field also determines the experiment's parent in the hypothesis tree**: `null` means the experiment branches from baseline; a specific commit means it branches from the experiment that produced that commit. This allows a single planning round to propose experiments with different parents (e.g., NVDEC from baseline + batch_size from a successful MTP experiment).
