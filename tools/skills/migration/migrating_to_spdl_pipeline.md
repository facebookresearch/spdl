# Migrating Data Loaders to SPDL

## Campaign obligations

- Before Phase 1 tests/launches, read the bundled `workflow.md` and `development_record.md` references for the indexed record and every-production-launch SPDL queue analysis.
- After parity, that workflow owns architecture/model optimization, neutral hygiene, validated stacking and final sweeps; this guide is the single baseline procedure. Preserve each phase's gates and keep temporary campaign documents out of reusable skill dependencies.

## 1. Preserve the baseline

- Capture the original loader/model revision, effective data/configuration, sampler/sharding, seed, batching, workers/prefetch, cache state, CPU allocation, training topology and comparison windows. Keep the original path reproducible.
- Construct SPDL orchestration with `PipelineBuilder` or a complete `PipelineConfig`, choosing the closest execution/ownership graph. For an original multiprocessing loader, a multiprocessing region is often closest; do not force threading or MTP first.
- Compose the SPDL pipeline from the original implementation's **functions and classes** for sampling, lookup/cache, decode, transforms, fused collate and transfer. Reuse the processing building blocks, not the original loader's orchestration. Ordinary trainer-interface adaptation is compatible with this requirement. If extraction is necessary, share it between both paths and compare against the untouched original.
- Preserve batch membership/order, partial-batch policy, worker-local state/RNG, epoch/restart behavior, constructor probes, pinning and H2D boundaries. Record unavoidable deviations; loss alone does not prove equivalence. For TorchTNT constructor probes, read [constructor_probe.md](constructor_probe.md).
- Do not split/rewrite collate, change decoder/cache formats, introduce RPC/GPU preprocessing/custom memory pools, change training topology/batch size, or run broad tuning before parity. Fused work may already avoid allocation/copy/coordination overhead; a cache-heavy local path may lose to RPC serialization/network cost.

## 2. Iterate locally and record joint parity evidence

Use matched local original-loader/SPDL tests as the fast inner loop; separate detailed consistency checks from timed runs.

- **Data:** compare sample IDs/order, output values/shapes/dtypes, lengths, masks/padding, controlled augmentation/RNG, epoch/probe/restart behavior and applicable quality checks. Exercise actual lookup/decode/transform/collate and process boundaries, not unrelated synthetic work or repeated cached output batches.
- **Matched setup:** use the same machine/CPU allocation, representative data, sampler/seed, batch count/size, workers/prefetch, cache state, warmup and consumer work. Choose windows before examining results; repeat to resolve noise.
- **Paired point:** report `(CPU utilization, throughput)` over the same interval. Include the full loader/harness process tree: frontend/orchestration, all workers and native/feeder/pin threads. Use one CPU normalization and exclude unrelated colocated work from both arms.
- **Accounting:** aggregate user+system CPU-seconds divided by elapsed wall time gives core equivalents; divide by the same allocated core count for utilization percent. Throughput is completed equivalent samples/batches divided by that same wall time. Missing worker accounting or different windows are not a valid pair.
- **Acceptance:** points must be close within repeat-run variation or an established tolerance; comparable/better throughput with no greater CPU demand is acceptable. Materially more CPU to match throughput fails initial migration parity, even below a generic host-CPU budget. Do not invent a tolerance or hide unresolved variability.
- **Training confirmation:** local tests accelerate iteration but do not replace matched end-to-end training and model-quality validation. Compare completed work/complete step time with the same data, model, resources and declared windows. Separate startup/cold-cache from warm/E2E scopes; retain recurring costs within their proper scope.
- **Baseline receipt:** record component reuse/deviations, correctness results, exact local/job/revision/configuration references, paired CPU/throughput values, memory metric/scope, windows/coverage and pass/unresolved verdict. A runnable pipeline, full sink, near-zero fetch wait or loader-only QPS is not parity evidence.

Repair attributed adapter, scheduling, state-ownership or process-boundary overhead before parity with the smallest semantics-preserving change, then repeat these checks. This is necessary migration repair, not permission for speculative optimization.

## 3. Diagnose a non-empty sink with slower training

A 100% occupation rate means the sink is non-empty, not necessarily full. With lower end-to-end throughput it prioritizes **main/trainer-process GIL contention** as a hypothesis, not a proven cause; slowed consumption can let the sink refill despite frontend interference.

- Confirm matched progress/complete-step windows and all-rank minimum sink occupancy, low/empty-window fractions, QPS and put/get waits. Cross-rank averages can hide starvation.
- Inspect blocking summaries/blind spots and capture actual GIL-holder Python/native stacks across all trainer-process threads. Correlate source/feeder serialization, result deserialization, pinning, callbacks and model work with the slowdown.
- Distinguish an actual holder from a named event-loop victim such as `disaggregate`, isolated workers from the trainer, and one-time bootstrap from recurring queue-feeder serialization. Separately attribute synchronous work blocking the asyncio loop. Missing profiling means unresolved attribution; do not tune blindly.
- If recursive request pickling is the confirmed regression, restore compact process-boundary inputs and worker-side materialization while retaining data/processing semantics. Do not remove a named victim based on a warning alone.
- Recheck correctness, CPU cost and complete-work throughput after repair. Initial optimization remains blocked until joint parity passes; model/communication/memory/transfer stalls may coexist.

## 4. Record capacity and hand off after parity

- Plot original/minimal-SPDL loader-only capacity **without training load**: x = consistently normalized CPU utilization; y = throughput and memory on labelled axes/panels. Match representative inputs/cache/consumer work and report worker/resource settings, intervals/repeats and shared-memory accounting; summed shared RSS is not unique memory.
- At least **2x throughput at comparable resource cost** makes fewer training nodes a separate later candidate, not scaling proof or permission to change Phase 1 topology. Validate that candidate's global-batch/schedule/data semantics, GPU/communication behavior, end-to-end cost/speed and original-baseline quality.
- Save the passing minimal SPDL source/configuration and evidence. Post-parity MP/MT/MTP comparisons may be empirical architecture exploration; other remedies require measured bottleneck evidence. Model accuracy uses the original baseline, and subsequent experiments start from the validated cumulative top with immediate-parent speed comparisons. Final parameter sweeps wait until architecture is finalized.
- Optional construction examples are in [pipeline_patterns.md](pipeline_patterns.md); load them only when implementing the relevant shape. They are not baseline prerequisites.
