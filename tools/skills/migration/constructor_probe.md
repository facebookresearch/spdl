# Constructor-Probe and Worker-State Parity

Read when migrating a child of `torchtnt.utils.data.MultiDataLoader`.

- Its constructor calls `next(iter(child_loader))` to reject empty children. With persistent PyTorch workers this is not a harmless peek: initial dispatch receives `W * P` tasks, returns a batch and refills a task. Up to `W * P + 1` batches can execute and be discarded before real training; worker dataset/cache/RNG state persists. `W=12, P=2` illustrates 25, not a universal count.
- Bypassing the constructor or wrapping a continuous SPDL iterator can alter numerical, startup and cache behavior. Abandoning the probe iterator may leave an epoch in flight; a plain iterable wrapper does not establish parity.
- Reproduce the probe by default. If removal is intentional, record it as a separate semantic decision with quality validation, not proof of bit identity.
- Exact emulation uses the persistent worker pool, drains/discards probe work, resets sampler epoch 0, realigns dispatch as needed and preserves worker-local state. The baseline refill boundary is after CPU pinning and before external H2D; `transfer_tensor_h2d` combines these operations, so a separate pin-only boundary may be necessary.
- In correctness runs compare worker IDs/call counts, sampled indices, pre-transform RNG fingerprints and post-collate tensor hashes for the probe and first training batches. Keep detailed dumps outside performance timing.
- Verify allocation/copy behavior too: a custom collate need not use `default_collate`'s shared-storage `out=` path. Reproduce source epoch length, partial batches, restart policy and cancellation/shutdown semantics; do not assume an iterator adapter supplies them automatically.
