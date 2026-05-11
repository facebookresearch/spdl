You are an expert in SPDL data loading pipeline optimization for AI training. Your task is to add performance instrumentation to a training script so that key metrics are logged during each run.

## Required Instrumentation

Add logging for the following metrics if not already present:

1. **TTFB (Time to First Batch)**: Measure the wall-clock time from the start of the training loop (or dataloader iteration) until the first batch is received. Log it once at the start of training.

2. **Step time**: Measure the wall-clock time of each training step (fetch batch + forward + backward + optimizer step). Log a running average every N steps (e.g. every 10 steps).

3. **Data fetch time**: Measure the time spent fetching each batch separately from the compute time. This helps distinguish data loading bottlenecks from compute bottlenecks — but note that this metric alone can be misleading due to GIL contention (see knowledge base). It is significant only when the data fetch time alone is non-trivial. Fast data fetch time does not necessarily mean fast training.

   **Important**: If the code uses `for batch in dataloader:`, keep that pattern — wrap the timing around it. If you need to time fetch separately, create an explicit iterator with `data_iter = iter(dataloader)` before the loop, then use `batch = next(data_iter)`. Do NOT call `next(dataloader)` directly — not all dataloaders are iterators (some are only iterable).

4. **Available CPU cores**: At startup (before the training loop), log the number of CPU cores available to this process. Use `os.sched_getaffinity(0)` (preferred, reflects cgroup/taskset restrictions) with a fallback to `os.cpu_count()`. Log it once.

Use Python's `time.monotonic()` for timing. Print metrics to stdout in a parseable format like:
```
[autoresearch] cpu_cores=32
[autoresearch] ttfb_s=12.34
[autoresearch] step=100 step_time_ms=150.2 fetch_time_ms=5.1 compute_time_ms=145.1
```

## Source Code to Instrument

The file to modify is: `__PIPELINE_SCRIPT__`

```python
__PIPELINE_CODE__
```

## Instructions

1. Add the timing instrumentation with minimal changes to the existing code structure.
2. Do NOT change any pipeline configuration, model setup, or training logic — only add timing measurements and logging.
3. Preserve all existing functionality exactly.

**CRITICAL: You MUST output the ENTIRE modified file in a single ```python code block below. Do NOT summarize the changes. Do NOT describe what you would change. Do NOT output a diff. Output ONLY the complete file content with instrumentation added.**

```python
