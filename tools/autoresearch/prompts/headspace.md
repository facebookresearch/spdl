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
dataloader = CacheDataLoader(dataloader, num_caches=10, return_caches_after=100, stop_after=500)
```

**Rules:**
- Find where the dataloader or pipeline object is built (e.g., `PipelineBuilder().....build()`, or a function that returns the dataloader).
- If the pipeline is built inside a function that returns it, add the wrapping at the **call site** where the return value is used, or wrap it inside the function right before returning. Prefer the call site if it is clear.
- If the pipeline uses `run_pipeline_in_subprocess`, wrap the **final outer pipeline** after `.build()`.
- Add the `from spdl.dataloader import CacheDataLoader` import at the top of the file with the other imports.
- **Step limit**: The `stop_after=500` parameter makes `CacheDataLoader` yield at most 500 batches then stop, so the training loop terminates naturally. No manual step counter is needed.
- Keep all other code unchanged — only add the CacheDataLoader import and wrapping.
- Do NOT remove or modify any existing instrumentation (e.g., TTFB logging).
- **Iterator compatibility**: `CacheDataLoader` is iterable (`for batch in dl:`) but NOT an iterator (`next(dl)` fails). If the training loop uses `next(dataloader)`, you MUST change it to `data_iter = iter(dataloader)` before the loop, then `next(data_iter)` inside the loop. Or simply rewrite to `for batch in dataloader:`.
- **TorchTNT scripts**: Always wrap the **outermost iterable** with CacheDataLoader. If the SPDL Pipeline is wrapped in a custom dataloader class, wrap that custom class — not the Pipeline directly. CacheDataLoader must sit at the top of the dataloader chain so it properly blocks all internal work (pipeline stages, subprocess I/O, etc.). Find the function that creates and returns the dataloader, and wrap it there:
  ```python
  def get_dataloader(...):
      dataloader = CustomDataLoader(...)
      return CacheDataLoader(dataloader, num_caches=10, return_caches_after=100, stop_after=500)
  ```

**CRITICAL**: You MUST output the complete modified file content inside a single fenced code block with the `python` language tag. Do NOT output a diff, do NOT output partial snippets, do NOT describe the changes in prose. The output MUST contain exactly one block in this format:

```python
# full file content here...
```

Do NOT truncate the output. Output the ENTIRE file, even if it is long.
