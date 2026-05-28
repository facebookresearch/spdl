You are an expert in SPDL data loading pipeline optimization for AI training. Your task is to add performance instrumentation to a training script so that key metrics are logged during each run.

## Required Instrumentation

Add logging for the following metrics if not already present:

1. **TTFB (Time to First Batch)**: Measure the wall-clock time from the start of the training loop (or dataloader iteration) until the first batch is received. Log it once at the start of training.

2. **Step time**: Measure the wall-clock time of each training step (fetch batch + forward + backward + optimizer step). Log a running average every N steps (e.g. every 10 steps).

3. **Data fetch time**: Measure the time spent fetching each batch separately from the compute time. This helps distinguish data loading bottlenecks from compute bottlenecks — but note that this metric alone can be misleading due to GIL contention (see knowledge base). It is significant only when the data fetch time alone is non-trivial. Fast data fetch time does not necessarily mean fast training.

   **Important**: If the code uses `for batch in dataloader:`, keep that pattern — wrap the timing around it. If you need to time fetch separately, create an explicit iterator with `data_iter = iter(dataloader)` before the loop, then use `batch = next(data_iter)`. Do NOT call `next(dataloader)` directly — not all dataloaders are iterators (some are only iterable).

4. **Available CPU cores**: At startup (before the training loop), log the number of CPU cores available to this process. Use `os.sched_getaffinity(0)` (preferred, reflects cgroup/taskset restrictions) with a fallback to `os.cpu_count()`. Log it once.

5. **Garbage collection alignment**: Disable automatic garbage collection and run it at a fixed step interval. This prevents random GC pauses from causing DDP synchronization jitter across ranks, and produces more stable per-step timing measurements. Add the following at the beginning of the training function (before the training loop):

   ```python
   import gc
   gc.disable()
   ```

   Then inside the training loop, after each step:

   ```python
   if step % 50 == 0:
       gc.collect()
   ```

   If the code already has manual GC management (e.g., `gc.disable()`, `gc.collect()`, or a TorchTNT `GarbageCollector` callback), do NOT add duplicate GC handling — leave the existing logic in place.

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

1. Add the timing instrumentation and GC alignment with minimal changes to the existing code structure.
2. Do NOT change any pipeline configuration, model setup, or training logic — only add timing measurements, logging, and GC alignment.
3. Preserve all existing functionality exactly.
4. **TorchTNT scripts**: If the code uses TorchTNT (`torchtnt.framework.fit`, `torchtnt.framework.train`, `AutoUnit`, etc.), the training loop is managed by TorchTNT — you CANNOT wrap timing around `for batch in dataloader:` directly. The code typically has a wrapper class around the SPDL Pipeline with `__iter__` that calls `pipeline.get_iterator(timeout=...)`. To add instrumentation:
   - **TTFB**: Add timing inside the wrapper's existing `__iter__` method. If there is no wrapper (Pipeline passed directly to TorchTNT), introduce one. Example:
     ```python
     # Add TTFB timing to the existing wrapper's __iter__:
     def __iter__(self):
         t0 = time.monotonic()
         it = self.pipeline.get_iterator(timeout=30)
         for i, item in enumerate(it):
             if i == 0:
                 print(f"[autoresearch] ttfb_s={time.monotonic() - t0:.2f}")
             yield item
     ```
   - **Step time and data fetch time**: Create a **TorchTNT callback** class that implements `on_train_step_start` and `on_train_step_end` to measure step time, and `on_train_get_next_batch_start` / `on_train_get_next_batch_end` to measure data fetch time. Register this callback in the `fit()` or `train()` call. Example:
     ```python
     class AutoresearchTimingCallback(torchtnt.framework.callback.Callback):
         def __init__(self):
             self._step_start = 0.0
             self._fetch_start = 0.0
             self._step_count = 0
             self._ttfb_logged = False

         def on_train_get_next_batch_start(self, state, unit):
             self._fetch_start = time.monotonic()

         def on_train_get_next_batch_end(self, state, unit):
             self._fetch_time = time.monotonic() - self._fetch_start

         def on_train_step_start(self, state, unit):
             self._step_start = time.monotonic()

         def on_train_step_end(self, state, unit):
             elapsed = time.monotonic() - self._step_start
             self._step_count += 1
             if self._step_count % 10 == 0:
                 print(f"[autoresearch] step={self._step_count} step_time_ms={elapsed*1000:.1f} fetch_time_ms={self._fetch_time*1000:.1f}")
     ```
   - Pass `callbacks=[AutoresearchTimingCallback()]` to the `fit()` or `train()` call.
   - **GC alignment for TorchTNT**: If TorchTNT does not already have a `GarbageCollector` callback registered, add one:
     ```python
     from torchtnt.framework.callbacks import GarbageCollector
     ```
     Then add `GarbageCollector(step_interval=50)` to the callbacks list.

**CRITICAL: You MUST output the ENTIRE modified file in a single ```python code block below. Do NOT summarize the changes. Do NOT describe what you would change. Do NOT output a diff. Output ONLY the complete file content with instrumentation added.**

```python
