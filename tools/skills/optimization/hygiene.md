# Preferred Hygiene and Ownership Checks

Read [workflow.md](workflow.md) for the separate **validated neutral hygiene** acceptance path. These are preferred after initial parity if matched performance/resources, correctness, model quality and reliability do not regress; label neutral results neutral, not speed gains. This small hygiene set is not permission to apply unrelated known optimizations blindly.

## Background SPDL H2D

- Prefer background `spdl.io.transfer_tensor_h2d` when the source/device/stream and tensor/shared-buffer lifetimes are correct and checks are neutral or faster. Retain the original path if it regresses.
- The helper pins tensors, uses a thread-local CUDA stream and synchronizes its transfer stream before returning. Overlap comes from background execution relative to training, not from pretending the call returns before DMA completes.
- Stable one-thread transfer ownership avoids crossing cached stream state between threads. Check `LOCAL_RANK`, CPU pinning, stream dependencies, retained aliases and shared-output release. Use the appropriate conversion/transfer for native buffers rather than calling a tensor helper indiscriminately.

## Avoid redundant foreground-prefetch

- When SPDL already produces GPU-ready batches and owns lookahead/transfer, prefer disabling redundant TorchTNT foreground-prefetch. The inspected AutoUnit interface uses `enable_prefetch=False`; use the owning project's actual option (`foreground_prefetch` where that is its public configuration), not an invented universal flag.
- That flag still calls `move_data_to_device`. Check the remaining traversal, stream handoff and lifetime behavior; it does not automatically remove all no-ops. Do not disable required transfer for CPU output or add an unsafe identity override.
- Compare the actual foreground trace, throughput, CPU and all-rank tails. Keep the original loader's transfer/overlap policy in Phase 1; initial parity is not obtained by changing that policy silently.

## Regulated GC

- Prefer a supported callback or explicit collection policy when neutral or faster with bounded memory and correct lifecycle behavior. Do not change GC as part of measurement-only instrumentation.
- TorchTNT's inspected `GarbageCollector(step_interval=...)` disables automatic collection, performs generation-1 collection on steps and full collection at the interval, and re-enables collection in its normal end hooks. Verify the version's exceptional/cancellation lifecycle and process scope instead of assuming restoration always happens.
- Equal step intervals across ranks do not guarantee simultaneous pauses. Compare actual collection events, allocation/cyclic objects, shared-view reclamation, tails, memory growth and complete-work elapsed costs. Do not hide recurring pauses using median step time.
- Retaining the policy is Phase 2 hygiene; **tune its interval in Phase 3** using [parameter_sweep.md](parameter_sweep.md). Do not mandate an arbitrary fixed interval.

## Handoff

Keep one neutral hygiene change attributable, record its measured non-regression and quality/reliability evidence, and stack only after those gates pass. New candidates use the updated validated top. Use [orchestration.md](orchestration.md) for isolated approach ownership and [development_record.md](development_record.md) for a concise attempt entry with detailed evidence linked.
