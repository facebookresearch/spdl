You are an expert in SPDL data loading pipeline optimization for AI training. Your task is to add performance instrumentation to a training script so that key metrics are logged during each run.

## Required Instrumentation

Add logging for the following metrics if not already present:

1. **TTFB (Time to First Batch)**: Measure the wall-clock time from the start of the training loop (or dataloader iteration) until the first batch is received. Log it once at the start of training.

2. **Step time**: Measure the wall-clock time of each training step (fetch batch + forward + backward + optimizer step). Log a running average every N steps (e.g. every 10 steps).

3. **Data fetch time**: Measure the time spent in `next(dataloader)` separately from the compute time. This helps distinguish data loading bottlenecks from compute bottlenecks — but note that this metric alone can be misleading due to GIL contention (see knowledge base). It is signigicant only when the data fetch time alone is non-trivial. Fast data fetch time does not necessarily mean fast training.

Use Python's `time.monotonic()` for timing. Print metrics to stdout in a parseable format like:
```
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
