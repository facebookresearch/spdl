# Compilation, graph replay and host dispatch

Compilation turns selected tensor code into an optimized implementation; graph replay records GPU work so later calls can submit it with less host dispatch. Neither replays arbitrary Python work. Choose stable boundaries, keep dynamic state explicit and include setup costs in the appropriate measurement scope.

**Host dispatch** is the CPU work that Python, the framework and the driver do to enqueue each GPU kernel. When kernels are short, the GPU can finish one before the host enqueues the next and then sits idle. A **host readback** or synchronization, such as `.item()`, makes the host wait for queued GPU work, which stops it from running ahead. In `torch.compile`, a **graph break** ends the traced region and runs the following Python eagerly before tracing resumes. A **guard** is a check on inputs, shapes or Python values that decides whether existing compiled code can be reused; a failed guard triggers recompilation.

Choose the topic whose symptoms match the measured bottleneck. Open only that reference for technique entries and their compilation, state, replay and training guards.

[Model catalog](index.md)

## Compile boundaries and reuse

- [Compile boundaries and reuse](graphs_dispatch/compile_boundaries_and_reuse.md): mixed-wrapper graph breaks, repeated-block cold compilation and unsupported custom-kernel boundaries.

## Compile shape and schedule contracts

- [Compile shape and schedule contracts](graphs_dispatch/compile_shape_and_schedule_contracts.md): changing dimensions or schedule values trigger recompilation, while tensor-value branches and data-derived sizes disrupt tracing.

## Host dispatch and CUDA replay

- [Host dispatch and CUDA replay](graphs_dispatch/host_dispatch_and_cuda_replay.md): repeated optimizer-state traversal, non-finite verdict or reporting readbacks and launch gaps between short kernels.
