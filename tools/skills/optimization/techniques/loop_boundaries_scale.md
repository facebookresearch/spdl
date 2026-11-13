# Training-loop boundaries and scale

Candidates for the work around the input path at evaluation, epoch, checkpoint and resume boundaries, rank synchronization and side work on the critical path, and GPU-count changes once the loader no longer limits the job. Each entry is a candidate to test against its trigger, not a guaranteed gain.

[Catalog index](../technique_catalog.md)

**`distributed_evaluation_sharding`**
- Trigger: every rank evaluates the full validation set at an evaluation boundary.
- Technique: for a finite holdout indexed `0..N-1`, rank `r` of `R` can evaluate indices `r, r + R, r + 2R, ...` below `N` instead of each rank reading all `N` rows. Keep the global row index with each prediction. Gather predictions, restore that index order, and run the original metric over the same complete set. Use the SPDL pipeline only for each rank's assigned source and transforms.
- Guards: every eligible example must appear exactly once; distributed samplers that pad to an equal length can create duplicates. Handle uneven tails and empty shards with compatible collective participation. Preserve evaluation preprocessing/random state and metric aggregation; averaging per-rank nonlinear metrics is generally not equivalent to computing the original metric over all predictions.
- SPDL reference: A rank-local {py:class}`spdl.pipeline.PipelineBuilder` graph can reuse the evaluation source/transforms and {py:func}`spdl.io.transfer_tensor_h2d` path. Dataset sharding, distributed prediction gathering, order restoration and metrics remain application responsibilities; SPDL does not choose the correct evaluation partition automatically.

**`epoch_boundary_work_audit`**
- Trigger: faster training exposes fixed per-epoch callback, export or barrier costs.
- Technique: remove redundant model initialization in export, skip barriers on epochs with no dependent work, and align export cadence with its consumers once the owner confirms that no one uses the intermediate artifacts.
- Guards: Do not remove synchronization needed by other ranks or omit artifacts another consumer needs. Compare the same evaluation/checkpoint schedule and include shutdown and final drains in completion time.

**`fewer_gpus_same_global_batch`**
- Trigger: after loader fixes, host CPU and memory have headroom and per-GPU throughput is the cost driver.
- Technique: compare fewer data-parallel ranks with a proportionally larger local batch while preserving the model recipe's effective global batch. Without accumulation, `global_batch = ranks * local_batch`; with accumulation, include that factor too. For example, changing 64 ranks with local batch `b` to 32 ranks with local batch `2b` keeps the total unchanged only when the rest of the update schedule is unchanged. Retune worker count and I/O concurrency for the greater demand per remaining rank.
- Guards: larger local batches change memory, kernel shapes, normalization and random/order behavior. Preserve optimizer updates, scheduler, sample coverage and quality, and include the measured resource duration/cost of completed runs. Do not reduce model-parallel groups or other topology axes under a data-parallel batching rule. More samples per GPU alone does not establish a faster or cheaper completed run.
- SPDL reference: Retune the existing {py:class}`spdl.pipeline.defs.ProcessPoolExecutorConfig` worker count and {py:meth}`spdl.pipeline.PipelineBuilder.pipe` concurrency as separate knobs after the rank topology changes. SPDL does not preserve global batch or quality on behalf of the trainer; the original model recipe and matched full-run validation remain the reference.
