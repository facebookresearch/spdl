# Process runtime environment

Choose the topic whose symptoms match the measured bottleneck. Open only that
reference; the technique IDs, Trigger/Technique/Guards detail and applicability
caveats live in the topic files, not in this routing page.

[Catalog index](../technique_catalog.md)

## Process layout and the GIL

- [Process layout and GIL contention](process_runtime/process_layout_and_gil.md): worker-process
  placement, free-threaded Python, native GIL release and trainer handoff work.
- [Row dispatch and handoff](process_runtime/row_dispatch_and_handoff.md): worker-local
  fetch/transform, light-work IPC bypass, completion ordering and repeated
  context/tree/copy overhead.

## Executors, admission and host CPU

- [Executors and admission](process_runtime/executors_admission.md): nested native pools,
  separate I/O executors, fetch limits, remote-pool sizing and completion-first
  admission.
- [Host CPU and memory isolation](process_runtime/host_cpu_isolation.md): co-located
  input work starving trainer threads, and memory-bandwidth or topology contention.

## Memory and allocators

- [Memory and allocators](process_runtime/memory_allocators.md): page retention/purge,
  allocator arena counts and compressed-swap admission budgets.

## Garbage collection

- [Garbage collection](process_runtime/garbage_collection.md): cyclic-GC policy, freezing,
  pressure-gated collection and traceback-free timeout handoff. Follow the
  [GC policy](../gc_optimization.md), including lifecycle and prior-state guards.

## Resource lifetime and fault containment

- [Resource lifetime](process_runtime/resource_lifetime.md): persistent worker/client owners,
  epoch/batch-scoped buffers, descriptor cleanup and native-codec fault isolation.
