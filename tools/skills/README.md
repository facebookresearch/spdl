# SPDL Migration and Optimization Guidance

Use this index to load only the guidance for the current decision. Migration first preserves the original implementation's functions/classes and verifies data/resource/performance parity. Afterward, architecture comparisons are empirical experiments; other optimization techniques address a measured bottleneck.

## Migration

- [Baseline migration](migration/migrating_to_spdl_pipeline.md): closest architecture, component reuse, local CPU/throughput comparisons and end-to-end parity.
- [Constructor-probe parity](migration/constructor_probe.md): worker/RNG state and startup-probe semantics when applicable.
- [Construction patterns](migration/pipeline_patterns.md): optional execution examples, not baseline prerequisites.
- [New pipeline authoring](authoring/building_pipelines.md): greenfield construction when no existing loader is being migrated.

## Optimization

- [Workflow](optimization/workflow.md): phases, empirical architecture exception, acceptance/quality, stacking and final sweep.
- [Analysis](optimization/analysis.md): identify loader bottlenecks using pipeline statistics and model bottlenecks using traces/timings; select remedies and re-profile.
- [Cached-first model benchmarks](optimization/cached_model_benchmark.md): short warmed native training with real GPU-ready `CacheDataLoader` replay before full-epoch E2E and original-baseline quality validation; preserve distributed communication and correctness.
- [Pipeline statistics](optimization/pipeline_stats.md): timing scopes, all-rank readiness, starvation/backpressure and GIL/event-loop attribution.
- [Architectures](optimization/architectures.md): multiprocessing, multithreading and MTP comparisons and process-state constraints.
- [Execution remedies](optimization/execution_optimizations.md): measured I/O, dispatch, serialization and copy costs.
- [Media remedies](optimization/media_optimizations.md): measured decode/transfer and placement costs.
- [Hygiene](optimization/hygiene.md): validated neutral H2D, foreground-prefetch and GC decisions.
- [Parameter sweep](optimization/parameter_sweep.md): tune only after architecture is finalized.
- [Local testing](optimization/local_testing.md): paired data/resource tests and CPU/throughput/memory curves.

## Campaign operation

- [Orchestration](optimization/orchestration.md): dedicated isolated approach owners, job coordination and concise handoffs.
- [Development record](optimization/development_record.md): indexed phase records/one-pagers and concise attempts with evidence links.

Keep each topic in one canonical file. Load techniques for the identified cost, not as a mandatory checklist; every empirical architecture trial still needs post-analysis and acceptance evidence. The guidance contains the lessons themselves rather than requiring temporary campaign documents.
