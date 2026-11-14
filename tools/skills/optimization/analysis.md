# Choose the Next Optimization from Evidence

Read [workflow.md](workflow.md) for acceptance and the explicit architecture-exploration/neutral-hygiene exceptions. This file owns analysis and remedy selection; [pipeline_stats.md](pipeline_stats.md) owns loader-signal interpretation.

Before this procedure, perform the skill's bundled `async_boundary_audit.md` startup audit of recurring network-fetch call chains. Follow wrappers to the actual standard-library/native implementation; check per-call event-loop/client lifecycle costs, including `asyncio.run`, before selecting architecture or tuning concurrency. Source inspection exposes a candidate cost; runtime attribution establishes whether it matters.

## Decision procedure

1. Bind observations to the exact source/package/configuration, job/version/attempt, resources and completed-work window. Check whether the comparison is valid before ranking it; preserve unavailable or contaminated scopes explicitly.
2. Separate loader delivery/transfer, frontend host interference, forward, backward, parameter update, collectives, callbacks/checkpointing and epoch transitions. Identify **exposed critical-path time**, not just an operation that is slow in isolation.
3. Record the largest measured cost, rank/thread/process scope, supporting stats/trace/timing links, plausible cause and uncertainty. Do not declare model-bound behavior from full sinks or GPU activity: kernels may be layout/copy work and sampled queues may miss starvation.
4. Select a remedy from the knowledge base **for that cost**. State the expected mechanism and which measurement should change. Screen the triggers in [technique_catalog.md](technique_catalog.md) against the evidence, then load only the corresponding execution/media/hygiene/architecture reference; do not march through the catalog.
5. Make an attributable candidate on the validated parent and run basic data/interface checks. For model changes, first use the short native cached-batch gate in [cached_model_benchmark.md](cached_model_benchmark.md); after it passes, compare full-epoch E2E speed/resources with the real loader restored and **re-profile**. Loader changes require uncached loader/E2E evidence. A cached improvement alone is not a winning E2E critical-path optimization.
6. Once speed is repeatably better, check original-baseline model-quality non-degradation. The workflow decides acceptance/stacking and preserves provisional states. Return the concise result to the main agent for planning and record updates.

If evidence is insufficient, the next action is measurement or attribution—not a familiar optimization. An MP/MT/MTP comparison may instead be classified as architecture exploration because its benefit is implementation-dependent; compare bounded resources and processing semantics, then analyze every trial. This exception does not justify unrelated decoder/cache/model changes without evidence.

## Loader analysis before technique selection

- Fetch and analyze **SPDL pipeline performance statistics**, not only training logs or `next(dataloader)` latency: source/stage/sink QPS, service times, queue put/get waits, occupancy, failures and cross-rank variation. Use [pipeline_stats.md](pipeline_stats.md) and the statistics-acquisition tooling available in the deployment environment.
- Normalize item units at aggregate/disaggregate boundaries and relate queue behavior to completed trainer progress. Use minimum sink occupation across all ranks and worst-rank/empty-window evidence; average readiness cannot prove absence of starvation.
- Full/non-empty sinks with slower training prioritize actual main-process GIL-holder/asyncio attribution. Determine whether source/dispatch/serialization, result restoration, pinning, H2D or model/callback Python is interrupting progress; don't remove the named victim.
- Use function-level timings or holder/native stacks to distinguish decode/transform work, cache/storage wait, copy/allocation, lock/IPC and executor contention. Match memory, CPU and cache windows; a hot stage is relevant only if its cost is exposed to the consumer.
- After an architecture trial, explain the observed throughput/resource change using this evidence. Missing queue/holder information is an unresolved analysis gap, not proof that more threads or remote processing is required.

## Model analysis before technique selection

- Use representative **CPU/GPU traces or supported phase/operator/kernel measurements** to rank exposed forward, backward, optimizer/parameter-update and communication costs. Account for asynchronous GPU execution and overlapping work; summing overlapping kernel durations is not step critical-path time.
- Attribute kernels to actual operators/functions, not activity alone. Check time in copies/layout conversions, host launch gaps, synchronization/collectives, memory pressure and callbacks before claiming FLOP saturation or model-bound behavior.
- Match runtime/package, model/data/batch/schedule and the selected warm/complete-work scopes. Hold the validated loader fixed for a model candidate. If delivery starvation is unresolved, resolve it first or explicitly record a separate independent hypothesis.
- Select fusion/compilation, a specific update/optimizer implementation or supported 8-/16-bit precision only when the trace/timing explains what it would address. Record changed math/precision/runtime and hardware support; the technique name is not justification.
- Validate the intended path and short-step benefit first with [cached_model_benchmark.md](cached_model_benchmark.md), retaining the native optimizer and distributed NCCL. Warm compiler/autotune state and use mutation-safe real-batch replay with acquisition stopped; verify actual changed kernels/dtypes/layout rather than flags. Once the cached gate passes, run full-epoch E2E validation and original-baseline accuracy checks under the established protocol; do not invent margins.

## Instrumentation and optional headroom

- Prefer existing telemetry; add only missing, low-overhead measurements equally to control and candidate. Time completed equivalent work with monotonic clocks; detailed tensor hashing/dumps belong outside performance timing.
- Instrumentation must not silently change GC, caching, batching, source order, model math or architecture. Hygiene policies are separately recorded candidates. Use framework-supported callbacks, not a rewrite of trainer internals.
- For model experiments, `CacheDataLoader` is the default short performance assay described in [cached_model_benchmark.md](cached_model_benchmark.md). Separately, cached-output diagnostics can estimate conditional loading headroom but cannot validate the bypassed loader. Verify bounded fill/replay and explicit worker shutdown; neither use supplies training-quality evidence or the loader-only CPU/throughput capacity curve.
- A small headroom estimate is not itself a causal diagnosis or a mandate to compile the model. Combine it with actual queue/trainer/GPU evidence. Compilation startup versus warm behavior and recurring overhead remain separate scopes.

## Compact analysis handoff

Return: **measured bottleneck and evidence → selected relevant remedy/mechanism (or missing measurement) → expected affected signal → exact validated parent → result and re-profile outcome → quality/state → commit/job/analysis links**. For architecture exploration, label the comparison and state the observed result/cause instead of inventing a predicted bottleneck.
