# Transform compute

Reduce avoidable work in input decoding, feature preparation, augmentation and sampling. These are candidates for an SPDL stage or an equivalent input path, not guaranteed speedups; keep the consumed data and training semantics fixed unless explicitly testing a representation change.

[Catalog index](../technique_catalog.md)

**`per_row_cpu_generator_seeding`** (reproducibility)
- Trigger: batches change with worker scheduling, or seeding inside workers affects CUDA generators.
- Technique: use a dedicated CPU `torch.Generator` for each row's augmentation and derive its seed deterministically from the run seed, rank, epoch and stable row identity (or a step/position that is unchanged by worker scheduling). Pass that generator explicitly to supported random operations. This makes a row's draw independent of which worker happens to process it; do not use Python's process-randomized `hash()` as the seed mapping. Avoid global `torch.manual_seed` in workers because it also changes CUDA generator state.
- Guards: preserve the baseline's intended random distribution, distinct rank streams and checkpoint/resume behavior. Rows that legitimately recur need the same repeat/epoch identity policy in both paths. Validate repeated runs under different worker scheduling; explicit generators do not control unrelated random libraries automatically.
