# Batch assembly and transport

Choose the topic whose symptoms match the measured bottleneck. Open only that
reference; technique entries and their applicability guards live in the topic
files, not in this routing page.

These techniques change how samples become batches and how their bytes reach the trainer. **Collation** assembles samples into the arrays/tensors a consumer expects; **IPC** (interprocess communication) moves payloads or storage handles between processes. A stored row or record is not necessarily one sample, and an input chunk or worker-transport group is not necessarily one training batch. Preserve those mappings when changing where assembly happens.

Separate the copies: writing samples directly into final shared storage differs from building a private batch and copying it into an arena. A receiver may then view that arena or copy out into its own storage. Pinning can add another host copy before host-to-device (H2D) transfer; none of these host-side changes removes H2D itself. Measure construction, allocation and communication together in the complete pipeline.

[Catalog index](../technique_catalog.md)

## SPDL reference implementations

SPDL's {doc}`shared-memory arena guide </pipeline_guides/memory_and_transport/shared_memory_arenas>` describes two optional transport backends for {py:func}`spdl.pipeline.iterate_in_subprocess`: {py:class}`spdl.pipeline.SharedMemorySegmentPool` returns supported payloads as shared views, while {py:class}`spdl.pipeline.SharedMemoryRingBuffer` copies them into independent receiver-owned buffers. Both reuse bounded transport storage, but both still copy producer payloads into that storage. They are reference paths to compare when allocation or IPC is exposed, not requirements or guaranteed throughput improvements.

The pool does not by itself make an arbitrary collator write directly into a reserved slot. Keep a working caller-owned direct-construction implementation when that is the measured win; do not replace it with copy-in arena transport and call the result copy-free. The public arena classes are experimental. Check the installed SPDL version, budget all local ranks and retained views, and compare the complete pipeline against the original path.

## Shared batch storage and lifetimes

- [Shared batch storage and lifetimes](batch_assembly_transport/batch_shared_storage_lifetimes.md): allocation, process-hop copies or storage ownership dominate.

## Numeric batch collation

- [Numeric batch collation](batch_assembly_transport/batch_numeric_collation.md): numeric construction or per-feature copies dominate.

## Compact batch payloads

- [Compact batch payloads](batch_assembly_transport/batch_compact_payloads.md): excessive payload bytes, objects or per-delivery overhead.

## Batch workload scheduling

- [Batch workload scheduling](batch_assembly_transport/batch_workload_scheduling.md): batch variability, rank skew or shape proliferation stalls training.

## Batch delivery and topology

- [Batch delivery and topology](batch_assembly_transport/batch_delivery_topology.md): delivery jitter, consumer backpressure or duplicated cross-host work.
