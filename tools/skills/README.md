# SPDL Migration and Optimization Guidance

Use this index to load only the guidance for the current decision. Migration first preserves the original implementation's functions/classes and verifies data/resource/performance parity. Afterward, architecture comparisons are empirical experiments; other optimization techniques address a measured bottleneck.

## Startup audit

- [Hidden async-to-sync network boundaries](async_boundary_audit.md): follow recurring fetch wrappers to the actual client/event-loop/transport implementation before migration or optimization; measure lifecycle overhead and preserve native async ownership and baseline evidence; fix a found bridge with a lean dataset and execution in the pipeline, not a dataset-owned pool or loop.

## Migration

- [Baseline migration](migration/migrating_to_spdl_pipeline.md): closest architecture, component reuse, local CPU/throughput comparisons and end-to-end parity.
- [Constructor-probe parity](migration/constructor_probe.md): worker/RNG state and startup-probe semantics when applicable.
- [Construction patterns](migration/pipeline_patterns.md): datasets hold data and functions while the pipeline owns execution; optional execution examples, not baseline prerequisites.
- [New pipeline authoring](authoring/building_pipelines.md): greenfield construction when no existing loader is being migrated.

## Optimization

- [Workflow](optimization/workflow.md): phases, empirical architecture exception, acceptance/quality, stacking and final sweep.
- [Analysis](optimization/analysis.md): identify loader bottlenecks using pipeline statistics and model bottlenecks using traces/timings; select remedies and re-profile.
- [Cached-first model benchmarks](optimization/cached_model_benchmark.md): short warmed native training with real GPU-ready `CacheDataLoader` replay before full-epoch E2E and original-baseline quality validation; preserve distributed communication and correctness.
- [Pipeline statistics](optimization/pipeline_stats.md): timing scopes, all-rank readiness, starvation/backpressure and GIL/event-loop attribution.
- [Architectures](optimization/architectures.md): multiprocessing, multithreading and MTP comparisons and process-state constraints.
- [Technique catalog](optimization/technique_catalog.md): symptom-indexed input-pipeline candidates with stable IDs and correctness/resource guards; screen triggers after each analysis.
- [Model optimization catalog](optimization/model_techniques/index.md): a separate, curated index for model-execution mechanisms. Load only families matching compute, layout, dispatch, memory or distributed-wait profiles; preserve numerical and trained-quality checks.
- [Execution remedies](optimization/execution_optimizations.md): measured I/O, dispatch, serialization and copy costs.
- [Media remedies](optimization/media_optimizations.md): measured decode/transfer and placement costs.
- [GC runtime optimization](optimization/gc_optimization.md): establish and retain validated regulation across experiments; evaluate heap-scan/straggler improvements and safe process-owned freeze/unfreeze. Measured gains are optimizations, neutral retention is hygiene.
- [Neutral hygiene](optimization/hygiene.md): validated H2D/foreground-prefetch cleanup and baseline-policy retention without a measured performance gain.
- [Parameter sweep](optimization/parameter_sweep.md): tune only after architecture is finalized.
- [Local testing](optimization/local_testing.md): paired data/resource tests and CPU/throughput/memory curves.

## Campaign operation

- [Orchestration](optimization/orchestration.md): dedicated isolated approach owners, job coordination and concise handoffs.
- [Development record](optimization/development_record.md): indexed phase records/one-pagers and concise attempts with evidence links.

Keep each topic in one canonical file. Load techniques for the identified cost, not as a mandatory checklist; every empirical architecture trial still needs post-analysis and acceptance evidence. The guidance contains the lessons themselves rather than requiring temporary campaign documents.
