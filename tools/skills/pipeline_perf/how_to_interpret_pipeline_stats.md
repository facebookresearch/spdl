# How to Interpret Pipeline Stats

## Event Name Parsing

SPDL events follow this naming convention:

### Runtime stats (logged every ~59 seconds)

Format: `pipeline:{stage_name}_{metric_suffix}`

Where `{stage_name}` is like `0:1:download[96]` (pipeline_id:stage_index:func_name[concurrency]) and queue names append `_queue`.

| Metric Suffix | Description | Unit |
|--------------|-------------|------|
| `_ave_time` | Average task execution time | seconds |
| `_num_tasks` | Number of tasks completed (including failures) | count |
| `_num_failures` | Number of failed tasks | count |
| `_qps` | Queue throughput (items/second) | items/s |
| `_put` | Average time waiting to put result in downstream queue | seconds |
| `_get` | Average time waiting to get item from upstream queue | seconds |
| `_num_items` | Items that passed through the queue | count |
| `_occupation_rate` | Fraction of time queue was non-empty (0.0-1.0) | ratio |

### Extracting stage names from event strings

Use this regex to parse events: `pipeline:(\d+:\d+:.+?)_(ave_time|num_tasks|num_failures)$` for task metrics, and `pipeline:(\d+:\d+:.+?)_(qps|put|get|num_items|occupation_rate)$` for queue metrics.

Group stages by stripping the metric suffix. Each stage will have both task metrics and queue metrics.

## Quick Reference: Queue Health

| Scenario | Queue Put Time | Queue Get Time | Occupancy |
|---|---|---|---|
| Pipeline fast enough | High (queues full) | Low (data ready) | High (~1.0) |
| Pipeline too slow | Low (rarely full) | High (data unavailable) | Low (~0.0) |

**Sink queue data readiness is the single most important metric** — it directly equals the time the training loop waits for data.

## Bottleneck Analysis Heuristics

### Understanding Queue Metrics

Each stage has an **output queue** that sits between it and the next stage. The queue metrics describe this output queue:

- `_occupation_rate`: Fraction of time the output queue is **non-empty** (0.0 = always empty, 1.0 = always has items)
- `_put`: Time the stage spends waiting to **put** its result into its output queue (high = output queue is full, downstream is slow)
- `_get`: Time the **next** stage spends waiting to **get** an item from this queue (high = queue is often empty, this stage is slow)

**Key insight**: A stage's output queue occupation rate tells you the **balance between that stage (producer) and the next stage (consumer)**:
- **High occupancy (near 1.0)**: Producer is faster than consumer. Data is always available. The producer is NOT the bottleneck — look downstream.
- **Low occupancy (near 0.0)**: Producer is slower than consumer. The consumer frequently finds the queue empty and waits. The producer IS the bottleneck for downstream stages.

### 1. Execution Time Bottleneck
The stage with the highest `_ave_time` is the primary compute bottleneck. Compare relative execution times — if one stage is 10x slower than others, it dominates pipeline latency. But also account for concurrency: a stage with 0.2s avg time and concurrency 16 has effective throughput of ~80 items/s, while a stage with 0.01s avg time and concurrency 1 has ~100 items/s.

### 2. Backpressure Detection
**Symptoms**: High `_put` wait time on a stage AND high `_occupation_rate` (near 1.0) on that stage's own output queue.
**Meaning**: This stage's output queue is full because the downstream stage can't keep up. The stage is blocked trying to enqueue results. The downstream stage is the actual bottleneck.
**Fix**: Increase concurrency of the downstream stage, or optimize the downstream stage's processing.

### 3. Data Starvation Detection
**Symptoms**: High `_get` wait time on a stage's queue AND low `_occupation_rate` on that same queue.
**Meaning**: The queue is frequently empty. The stage producing into this queue is too slow — downstream stages are idle waiting.
**How to trace the bottleneck**: Walk upstream through the pipeline. If stage N's output queue has low occupancy, stage N is the slow producer. Check stage N's `_ave_time` and concurrency — it may need higher concurrency or optimization.
**Fix**: Increase concurrency of the slow upstream stage, or optimize its processing.

**Important**: Do NOT confuse a queue with high occupancy for starvation. High occupancy means data is plentiful — the opposite of starvation. If a stage has high `_get` wait but its upstream queue has HIGH occupancy, the wait is likely lock contention or dequeue overhead, not starvation.

### 4. Identifying the Pipeline Bottleneck Stage
Walk through the pipeline from source to sink and look at each stage's output queue occupancy:
- Stages **before** the bottleneck have high output queue occupancy (data accumulates because the bottleneck slows everything downstream)
- The **bottleneck stage** itself has LOW output queue occupancy (it can't produce fast enough to keep its output queue full)
- Stages **after** the bottleneck also have low-to-moderate output queue occupancy (they're limited by the bottleneck's throughput)

The transition from high-to-low occupancy pinpoints the bottleneck.

### 5. Failure Analysis
**Symptoms**: Non-zero `_num_failures` for any stage.
**Severity**: Calculate failure ratio = `num_failures / num_tasks`. Above 5% is concerning; above 20% is critical.
**Action**: Failures cause retries and waste compute. Investigate error logs for the failing stage.

### 6. Throughput Consistency
Compare `_qps` across all stages. In a healthy pipeline, QPS should be roughly equal (accounting for aggregate/disaggregate stages that change item count). Large divergence indicates a bottleneck stage.

### 7. Straggler Detection (multi-rank)
Compare `_ave_time` for the same stage across ranks. Ranks with significantly higher (>2x) execution times are stragglers, often caused by:
- Slow network/storage on specific hosts
- Data skew (some ranks get harder examples)
- Resource contention

Also compare `_occupation_rate` and `_get` times across ranks for the same queue. If a specific rank has lower occupancy and higher get-wait on the same queue, upstream stages on that rank are slower — indicating a rank-specific bottleneck.

## Output Format

Present results as a structured markdown report:

```markdown
## SPDL Pipeline Performance Report

**Rank**: <rank> | **Ranks Available**: <total_ranks>

### Pipeline Architecture

| # | Stage | Type | Concurrency | Source |
|---|-------|------|-------------|--------|
| 0 | download | PipeConfig | 96 | preprocess.py:407 |
| 1 | decode | PipeConfig | 10 | preprocess.py:412 |

### Performance Summary

| Stage | Avg Time (s) | p50 (s) | p90 (s) | p99 (s) | QPS | Failures | Occupancy | Put Wait (s) | Get Wait (s) |
|-------|-------------|---------|---------|---------|-----|----------|-----------|-------------|-------------|
| download | 0.15 | 0.14 | 0.18 | 0.25 | 42.3 | 0 | 0.85 | 0.001 | 0.05 |
| decode | 0.82 | 0.79 | 0.95 | 1.20 | 12.1 | 3 | 0.32 | 0.42 | 0.001 |

### Bottleneck Analysis

- **Primary bottleneck**: `decode` — highest avg execution time (0.82s), causing backpressure upstream
- **Data starvation**: None detected
- **Failures**: 3 failures in `decode` stage (0.1% failure rate — acceptable)

### Recommendations

1. Increase `decode` concurrency (currently 10) — this is the pipeline bottleneck
2. Consider profiling `decode` with SPDL diagnostic mode for optimal concurrency
```
