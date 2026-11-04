# Phase 3: Parameter Sweep

Read the installed skill's bundled `workflow.md` and `development_record.md`
first for the global phase gates, production-launch analysis and persistent
record protocol. Use [architectures.md](architectures.md) for
Phase 2 design/attribution; this file is the final search over a **finalized,
speed/quality/reliability-validated architecture**, not a migration recipe.

## Freeze the Search Contract

Keep the architecture/source, processing functions, data/cache contract,
ordering/RNG behavior, model and training topology fixed. Record the exact
architecture control, effective overrides, allocated CPU/memory/GPU resources
and measurement windows. Choose only parameters implicated by Phase 2
queue/bottleneck evidence. Minimal settings needed to make an architecture safe
and runnable belong in Phase 2; a broad sweep does not.

Training batch size, node/rank count, global batch, gradient accumulation and
global sample/optimizer schedule are **not casual loader knobs**. Changing them
requires an explicit decision and separately matched end-to-end resource/cost
and original-model quality validation. Transport grouping is not permission to
change training batching. A structural bottleneck discovered during the sweep
means explicitly returning to Phase 2, then freezing a new validated control.

## Select Parameters and Account for Interactions

- **Stage concurrency:** `.pipe(fn, concurrency=N)` bounds in-flight tasks.
  It does not create N threads or measure CPU load. Async I/O can overlap without
  one thread per request; sync work can queue behind executor capacity.
- **Default executor:** `.build(num_threads=N)` sizes a shared pool for stages
  without a dedicated executor. It is not required to exceed every stage's
  concurrency. Account for simultaneous sync stages, not just the largest one.
- **MP workers and per-worker settings:** account for
  `max_workers * per_worker_stage_concurrency` and each worker's executor plus
  decoder/library threads. Include every rank and orchestration/feeder/transfer
  thread sharing a host. Do not confuse invocation capacity with actual CPU use.
- **Prefetch/lookahead and sink depth:** use only settings supported by the
  finalized implementation. Buffers can absorb variability but also increase
  first-result latency, retained/pinned memory and stale in-flight work. A deeper
  sink does not repair a producer's sustained capacity deficit or trainer GIL
  interference. Avoid unbounded buffering.
- **Region transport buffers:** the input and output `.to(..., buffer_size=N)`
  markers group transfers independently and transparently to stage functions.
  Keep training batching fixed. Input fill latency, whole-transfer worker
  assignment and multiplied in-flight memory can offset lower IPC overhead.
- **Regulated-GC interval:** only if Phase 2 retained a measured collection
  policy, tune its interval against correlated pause/rank-tail and memory/view
  reclamation evidence. Do not silently change policy while sweeping intervals.
- **Shared-memory arena sizing:** if already part of the architecture, size
  segment/count or ring capacity against measured unit bytes, alignment,
  queue depth and consumer-held aliases. A pool unit can contain multiple
  binaries. Slow-consumer backpressure/lifetime and cleanup are required checks;
  increasing capacity is not a substitute for diagnosing retained views.

Use the **actual task budget** and measured end-to-end impact. Generic CPU 40%
or cores/8 claims are not unconditional Phase 2/3 promotion vetoes. Increased
CPU/memory can be an acceptable documented speed/cost tradeoff within allocation
and reliability constraints; report it rather than hiding it or inferring
contention from a utilization percentage.

## Concurrency Search: Lower Values Matter

Start from the finalized control. Use local screens to explore justified lower
and higher settings, refine the promising region, and check interactions only
where evidence supports them. There is no fixed mandatory experiment count or
universal range. Keep the source/override of each result attributable and revisit
the control when noise, cache state or hardware changes cloud the comparison.

For cheap copying, demux or disaggregation, scheduling, locks and cache-line
contention can dominate useful work. More tasks can raise per-item latency
faster than concurrency and reduce realized throughput. Stage
`concurrency / service_time` is only an approximate capacity clue: queue waits,
shared executors and changing service-time distributions matter. Select from
**end-to-end training throughput/complete step time**, not that ratio or a
single isolated latency.

### Demux and Decode

If Phase 2 retained separate demux/decode stages, tune them independently while
preserving the split. SPDL's `Demuxer` documentation notes that construction
probes streams through FFmpeg decoder initialization; some external codecs
serialize on the process-wide `codec_mutex`. Lower demux concurrency is therefore
a useful candidate, not a universal cap of three or a guarantee for every codec
and FFmpeg version. Look for lock/service-time evidence in the workload.

For a retained NVDEC stage, tune stage concurrency and its dedicated executor
against the actual device/codec, decoder capacity, memory and competing GPU
work. Do not assume a fixed A100/H100 engine count. A pool's `max_workers` and
pipe concurrency are separate settings; threads waiting for hardware can waste
host resources without helping training. Switching CPU/GPU decode placement or
splitting a fused decoder is an architecture decision, not part of this sweep.

## CPU Video Decoder Threads

SPDL documents **one FFmpeg decoder thread by default** for `load_video` /
`decode_packets`. Override through `decode_config`, with string-valued options:

```python
import spdl.io


decode_config = spdl.io.decode_config(decoder_options={"threads": "2"})
frames = spdl.io.load_video(video_data, decode_config=decode_config)
```

`"1"` retains the documented default; a fixed value such as `"2"` or `"4"`
requests that thread count; `"0"` lets FFmpeg choose. These are candidate examples,
not mandatory trials or guaranteed single-video speedups. Support and scaling
depend on codec, resolution, decoder and library version. Auto threads may use
more cores and lose end-to-end despite a fast microbenchmark.

Co-tune with the finalized stage/worker configuration. A useful **thread-demand
estimate**, not a CPU-utilization formula, is
`workers * decode_concurrency_per_worker * decoder_threads`, summed across ranks
sharing a host; default/dedicated executors can limit active decoders and other
stages/library threads add demand. When raising decoder threads, consider lower
stage concurrency to avoid oversubscription, then measure actual CPU, memory,
service time and complete step time. Do not reject solely at a generic CPU
percentage or choose solely from isolated decode latency.

## Analyze Every Production Trial and Accept the Final Combination

Record every production launch/attempt, including failures/stops, and analyze
it when SPDL telemetry appears and again after completion. Use the `spdl` CLI
as the primary source where available and the bundled platform fetching
reference. Collect source/stage/sink QPS, service-time distributions, queue
put/get waits, occupancy, failures and cross-rank variation; normalize
aggregate/disaggregate units. Use **minimum sink occupancy across all ranks**,
low/empty-window fractions and worst-rank evidence, not rank averages. Missing
queue telemetry is an explicit gap. Read
[pipeline_stats.md](pipeline_stats.md) for
interpretation; a non-empty sink alone proves neither health nor a GIL cause.

Compare realized end-to-end throughput/complete step time, CPU and memory over
matched intervals/scopes. Separate startup/cold-cache cost from warm throughput
and report both relevant tradeoffs. Repeat the winning settings against a fresh
architecture control to establish a **repeatable speed win** before full
accuracy non-degradation validation against the **original model baseline**.
Use the established evaluation and uncertainty/acceptance criteria, not invented
tolerances. Data/interface smoke checks precede speed trials.

Keep a speed-only result `speed_validated` / `accuracy_pending` until quality and
reliability pass. Revalidate the final cumulative combination, not just isolated
settings. Update the shared record's summary, inventory and reproducible final
recipe with exact sources/overrides, validated gains, resource costs, quality
evidence, rejected alternatives and unresolved limits before recommending
production defaults.
