You are an expert in SPDL data loading pipeline optimization for AI training. You are assessing a baseline training job to identify optimization opportunities.

__KNOWLEDGE__

> Note: If infrastructure-specific knowledge (job scheduler CLIs, storage backends, etc.) is available, it is appended above by the selected platform provider.

---

## Your Task

Analyze the baseline job below. Identify the primary bottleneck in the data loading pipeline, estimate the optimization headroom, and propose concrete experiments to improve performance.

## Baseline Job: `__JOB_ID__`

### System Metrics
```
__SYSTEM_METRICS__
```

### Pipeline Performance Stats
__PIPELINE_STATS__

### Pipeline Source Code
```python
__PIPELINE_CODE__
```

### Launch Command
```
__BASE_LAUNCH_COMMAND__
```

### Additional Context
__NOTES__

---

## Instructions

1. **Summarize baseline performance**: Report SM utilization, CPU utilization, step time, and data readiness at the sink.

2. **Identify the bottleneck**: Walk through the pipeline stages from source to sink. Look at queue occupancy transitions, execution times, and failure rates. Pinpoint which stage is the bottleneck and why.

   **TorchTNT detection**: Check whether the code uses TorchTNT (`torchtnt.framework.fit`, `train`, `AutoUnit`, or a unit class with `compute_loss`). If so, note this in the assessment — it affects where instrumentation and code changes target. In TorchTNT scripts, the SPDL Pipeline is typically wrapped in a dataloader class with `__iter__`, and the training loop is managed by TorchTNT internally. Pipeline optimization changes still target the pipeline builder function and the wrapper, not TorchTNT internals.

3. **Estimate headroom**: Based on the metrics, estimate how much improvement is possible. Even if SM utilization is already high, there are always optimizations to try — reducing variance, improving data readiness, eliminating GIL contention via MTP, etc. Never dismiss optimization opportunities just because one metric looks acceptable.

4. **Propose 2-4 experiments** to run, ordered by expected impact. Always include best-practice structural changes that haven't been applied yet. For each experiment, specify:
   - What parameter or structural change to make
   - Why you expect it to help (hypothesis)
   - The exact launch command to use (use `$IMAGE` as placeholder for the job image)

**Always recommend these best practices if not already applied:**
- **Subprocess mode (MTP)**: Use the full MTP pattern from the knowledge base — `run_pipeline_in_subprocess()` with `embed_shuffle()`, `continuous=True`, and proper pickling. This eliminates GIL contention, isolates GC stalls, and removes epoch-boundary pipeline teardown. For TorchTNT scripts, the MTP refactor changes the pipeline builder and the dataloader wrapper's `__iter__` — TorchTNT internals are not modified.
- **Batch size tuning**: If GPU memory utilization has headroom, try larger batch sizes
- **Concurrency tuning**: Adjust concurrency of CPU-bound stages (try at least 2 values)

**Do NOT recommend `output_order="completion"`** — it is only needed when output ordering matters (e.g., evaluation jobs). For performance optimization, do not use it.

Include both parameter-only changes and structural code changes in the recommendations.

5. **Output a JSON block** at the end with your structured assessment:

```json
{
  "metrics": {
    "step_time_ms": <number or null>,
    "ttfb_s": <number or null>,
    "sm_utilization_pct": <number>,
    "cpu_utilization_pct": <number>,
    "data_readiness_pct": <number or null>,
    "bottleneck_stage": "<stage name>",
    "notes": "<one-line summary>"
  },
  "proposed_experiments": [
    {
      "name": "<short_snake_case_name>",
      "description": "<what we are testing>",
      "hypothesis": "<why this should help>",
      "category": "parameter_tuning|structural_change",
      "launch_command": "<full torchx command with $IMAGE placeholder>"
    }
  ]
}
```
