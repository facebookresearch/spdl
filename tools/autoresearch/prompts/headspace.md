You are adding CacheDataLoader instrumentation to a data loading pipeline for headspace analysis. This measures the maximum possible improvement from data loading optimization by caching batches and returning them repeatedly, eliminating actual data loading overhead.

**Important context:** With CacheDataLoader, the pipeline configuration (concurrency, MTP, batch size, etc.) does not affect the result — all batches come from cache after the initial fill. The result shows pure model compute time without any data loading. This is NOT a real training run — it is only used to estimate the upper bound of data loading optimization headroom.

## Pipeline Source Code (`__PIPELINE_SCRIPT__`)

```python
__PIPELINE_CODE__
```

## Instructions

Add `spdl.dataloader.CacheDataLoader` wrapping to the pipeline. The wrapper should be applied to the **final dataloader/pipeline object** right before the training loop consumes it.

The wrapping is:

```python
from spdl.dataloader import CacheDataLoader

# After building the dataloader/pipeline but before the training loop:
dataloader = CacheDataLoader(dataloader, num_caches=10, return_caches_after=100, stop_after=NUM_SAMPLES_PAR_EPOCH)
```

**Rules:**
- Find where the dataloader or pipeline object is built (e.g., `PipelineBuilder().....build()`, or a function that returns the dataloader).
- If the pipeline is built inside a function that returns it, add the wrapping at the **call site** where the return value is used, or wrap it inside the function right before returning. Prefer the call site if it is clear.
- If the pipeline uses `run_pipeline_in_subprocess`, wrap the **final outer pipeline** after `.build()`.
- Add the `from spdl.dataloader import CacheDataLoader` import at the top of the file with the other imports.
- **Add a step limit to prevent the job from running forever.** CacheDataLoader produces an infinite iterator (cached batches repeat indefinitely). The training loop must stop after a fixed number of steps. Add a step counter that breaks out of the training loop after 500 steps (enough for stable measurement, short enough to finish quickly). For example:
  ```python
  headspace_max_steps = 500
  step_count = 0
  for batch in dataloader:
      # ... training step ...
      step_count += 1
      if step_count >= headspace_max_steps:
          break
  ```
  If the code uses nested epoch/batch loops, add the counter and break to the **inner batch loop** and also break out of the outer epoch loop.
- Keep all other code unchanged — only add the CacheDataLoader import, wrapping, and step limit.
- Do NOT remove or modify any existing instrumentation (e.g., TTFB logging).
- **Iterator compatibility**: `CacheDataLoader` is iterable (`for batch in dl:`) but NOT an iterator (`next(dl)` fails). If the training loop uses `next(dataloader)`, you MUST change it to `data_iter = iter(dataloader)` before the loop, then `next(data_iter)` inside the loop. Or simply rewrite to `for batch in dataloader:`.

**CRITICAL**: You MUST output the complete modified file content inside a single fenced code block with the `python` language tag. Do NOT output a diff, do NOT output partial snippets, do NOT describe the changes in prose. The output MUST contain exactly one block in this format:

```python
# full file content here...
```

Do NOT truncate the output. Output the ENTIRE file, even if it is long.
