# Neutral Hygiene and Ownership Checks

Read [workflow.md](workflow.md) for the neutral-hygiene acceptance path. The [GC policy and outcome rules](gc_optimization.md#baseline-retention-and-outcome) define required regulation and distinguish neutral retention from performance gains. Preserve correctness, resources, quality and reliability; do not apply unrelated changes blindly.

## Background SPDL H2D

- Prefer background `spdl.io.transfer_tensor_h2d` when the source/device/stream and tensor/shared-buffer lifetimes are correct and checks are neutral or faster. Retain the original path if it regresses.
- The helper pins tensors, uses a thread-local CUDA stream and synchronizes its transfer stream before returning. Overlap comes from background execution relative to training, not from pretending the call returns before DMA completes.
- Stable one-thread transfer ownership avoids crossing cached stream state between threads. Check `LOCAL_RANK`, CPU pinning, stream dependencies, retained aliases and shared-output release. Use the appropriate conversion/transfer for native buffers rather than calling a tensor helper indiscriminately.

## Avoid redundant foreground-prefetch

- When SPDL already produces GPU-ready batches and owns lookahead/transfer, prefer disabling redundant TorchTNT foreground-prefetch. The inspected AutoUnit interface uses `enable_prefetch=False`; use the owning project's actual option (`foreground_prefetch` where that is its public configuration), not an invented universal flag.
- That flag still calls `move_data_to_device`. Check the remaining traversal, stream handoff and lifetime behavior; it does not automatically remove all no-ops. Do not disable required transfer for CPU output or add an unsafe identity override.
- Compare the actual foreground trace, throughput, CPU and all-rank tails. Keep the original loader's transfer/overlap policy in Phase 1; initial parity is not obtained by changing that policy silently.

## GC policy

Read [gc_optimization.md](gc_optimization.md) before selecting or changing GC policy; it owns the apply-and-retain requirement, performance diagnosis and freeze/unfreeze lifecycle safeguards. `gc.freeze()` remains a workload-dependent option, not a mandatory call for every process.

## Handoff

Keep each change attributable and include its measured outcome and correctness/resource/reliability evidence in the handoff. Use [orchestration.md](orchestration.md) for isolated approach ownership and [development_record.md](development_record.md) for concise attempt records.
