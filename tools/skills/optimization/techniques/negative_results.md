# Negative results

These counterexamples are not universal prohibitions. **Attempt** describes what was tried, and **Outcome** says what happened. **Not tested** marks a proposed change that was not run at the stated scope; **Related outcomes** are from separate tests. **Guards and when to revisit** explain what to check before trying the idea again.

Choose the topic that matches the proposed change. A component speedup, higher utilization or lower memory is not a substitute for faster completion of the same valid task. Conversely, higher CPU or memory use alone does not disqualify a faster run that fits its limits. Keep the resource and quality evidence.

[Catalog index](../technique_catalog.md) · {ref}`Compare loader implementations <compare-loader-implementations>`

## Loader layout and remote offload counterexamples

- [Loader layout and remote offload counterexamples](negative_results/loader_layout_and_remote_offload_counterexamples.md): loader-only swaps, splitting already-fused pipeline work and remote offload over an efficient local cache.

## Component speed and materialization counterexamples

- [Component speed and materialization counterexamples](negative_results/component_speed_and_materialization_counterexamples.md): shared-transformer memory savings, native-parser expansion, local operator timing, utilization and dynamic-padding gains without faster completion.

## Buffering and sample mixing counterexamples

- [Buffering and sample mixing counterexamples](negative_results/buffering_and_sample_mixing_counterexamples.md): decoded-episode memory tradeoffs, worker-local packing changes and extra transport capacity while queues already hold ready batches.

## Trainer pauses and communication counterexamples

- [Trainer pauses and communication counterexamples](negative_results/trainer_pauses_and_communication_counterexamples.md): reduction-bucket and distributed-mode regressions, and proposed forced collections without a recurring GC tail.

## Worker concurrency counterexamples

- [Worker concurrency counterexamples](negative_results/worker_concurrency_counterexamples.md): unequal fixed-product worker layouts and diminishing returns near CPU, memory or storage saturation.
