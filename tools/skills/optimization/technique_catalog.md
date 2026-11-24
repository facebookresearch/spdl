# Technique catalog: symptom-indexed candidates

The technique catalog brings together practical ways to improve SPDL input pipelines, along with lessons from approaches that did not help. The entries are organized around symptoms you may encounter, so you can find ideas relevant to your workload.

To choose a useful starting point, begin with the [analysis guidance](analysis.md): it helps you connect pipeline statistics and profiles to the cost limiting your training. Each entry is a **candidate** to test when your measurements match its trigger, rather than a checklist to apply in order.

Stable IDs identify screened and tested changes. Each entry explains when it applies, what to change and the checks needed before using it. There is no expected-gain field: compare the candidate with your own baseline. Source links are optional provenance; consult them when implementation, compatibility or correctness details need confirmation, not to copy a past speedup.

Before optimization, establish [migration parity](migrating_to_spdl_pipeline.md) by reusing the original components. If the replacement is slower, see {ref}`Compare loader implementations <compare-loader-implementations>` in the Optimization Guide. A faster path that loses data, changes effective batch or fails numerical/quality gates is not a promotable win. Use SPDL to compose and optimize the input pipeline. Baseline-specific adjustments preserve compatibility with components already in use.

## How to use the SPDL references

Where a public SPDL utility fits the mechanism, the area entry includes an **SPDL reference** with an API link or implementation example. These are options to compare with the existing pipeline, not required replacements or promises of a speedup. Check experimental status, the installed API version, data semantics and buffer lifetime before using one. Storage-client, allocator, GC and model-specific changes may still need the existing application's implementation; the catalog does not invent an SPDL API for them.

## How to screen

1. Gather SPDL stage statistics, **minimum sink occupancy across all ranks**, actual GIL-holder/native stacks, host file/anonymous/swap memory, total worker/thread CPU, per-rank GPU memory, and trainer/boundary timings. A full sink and low fetch wait do not rule out trainer-thread interference.
2. Use the trigger table to open only matching area files. Read the technique and guards before proposing an entry. Log each picked ID with the matching observation, and each inapplicable ID with the reason, in the campaign record's [technique screening log](development_record.md#technique-screening-log).
3. Rank candidates by exposed critical-path cost. Test independent hypotheses on separate forks of the validated parent. Preserve data coverage, randomization, checkpoint/resume and batch/update semantics.
4. Re-profile and compare matched native runs. Count cold preparation, startup, final drains, evaluation/checkpoints and realized resources in the complete-run result; validate quality against the original model baseline.

## ID to area file

The area files explain **Trigger** (when it may fit), **Technique** (what to change and why), **Guards** (correctness and tradeoffs), and an optional **SPDL reference**. [Counterexamples](techniques/negative_results.md) separate the **Attempt** and **Outcome** from **Guards and when to revisit**. If the change was not run, look for **Not tested** instead of **Attempt**. Deployment-specific mechanisms and internal source links belong in the deployment source map, not in these shared files.

### Cache and storage

| ID | File | One-line trigger |
|---|---|---|
| `direct_io_cache_reads` | [cache_storage/cache_reads_and_lookup_concurrency.md](techniques/cache_storage/cache_reads_and_lookup_concurrency.md) | Local disk-cache pages drive reclaim, swap, OOM, or progressively slower epochs. |
| `raw_cache_format` | [cache_storage/cache_payload_encoding.md](techniques/cache_storage/cache_payload_encoding.md) | `torch.save`/`torch.load` overhead, in-memory NPY copies from `numpy.load`, or many small cache reads dominate input cost. |
| `zero_copy_decompress` | [cache_storage/cache_payload_encoding.md](techniques/cache_storage/cache_payload_encoding.md) | Decompression, deserialization, and avoidable buffer copies dominate loader CPU. |
| `slim_stored_rows` | [cache_storage/cache_payload_encoding.md](techniques/cache_storage/cache_payload_encoding.md) | Unused fields or unnecessarily wide values inflate acquisition and cache footprint. |
| `cache_capacity_all_disks` | [cache_storage/cache_reads_and_lookup_concurrency.md](techniques/cache_storage/cache_reads_and_lookup_concurrency.md) | Unused dedicated mounts leave the working set larger than active cache capacity. |
| `lookup_rows_per_call` | [cache_storage/cache_reads_and_lookup_concurrency.md](techniques/cache_storage/cache_reads_and_lookup_concurrency.md) | Sequential batched dataset/cache lookups each wait for their slowest remote cache miss. |
| `lookup_concurrency_from_latency` | [cache_storage/cache_reads_and_lookup_concurrency.md](techniques/cache_storage/cache_reads_and_lookup_concurrency.md) | Measured in-flight work and round-trip latency cannot sustain trainer demand. |
| `serialize_non_thread_safe_client` | [cache_storage/cache_reads_and_lookup_concurrency.md](techniques/cache_storage/cache_reads_and_lookup_concurrency.md) | Concurrent callers corrupt or fail a process-local client. |
| `memoize_deterministic_intermediates` | [cache_storage/cache_reuse_and_sample_locality.md](techniques/cache_storage/cache_reuse_and_sample_locality.md) | Repeated transforms or evaluation recompute identical expensive intermediates. |
| `worker_stable_shard_membership` | [cache_storage/cache_reuse_and_sample_locality.md](techniques/cache_storage/cache_reuse_and_sample_locality.md) | Per-worker caches duplicate samples and lose reuse across epochs. |
| `atomic_cache_publication` | [cache_storage/cache_payload_encoding.md](techniques/cache_storage/cache_payload_encoding.md) | Interrupted or disk-full writes expose corrupt published cache archives. |
| `shared_mmap_catalogue` | [cache_storage/cache_metadata_catalogues.md](techniques/cache_storage/cache_metadata_catalogues.md) | Ranks and spawned workers duplicate a large Python metadata graph. |
| `eager_catalogue_row_materialization` | [cache_storage/cache_metadata_catalogues.md](techniques/cache_storage/cache_metadata_catalogues.md) | Repeated columnar row conversion creates periodic production stalls. |
| `lazy_native_frames_small_detection_cache` | [cache_storage/video_cache_and_storage_layouts.md](techniques/cache_storage/video_cache_and_storage_layouts.md) | Retained full-resolution decoded video dominates host memory. |
| `sequential_sampling_for_locality` | [cache_storage/cache_reuse_and_sample_locality.md](techniques/cache_storage/cache_reuse_and_sample_locality.md) | Random traversal destroys locality in a workload that may not need shuffled samples. |
| `normalized_entity_rows_late_expansion` | [cache_storage/cache_metadata_catalogues.md](techniques/cache_storage/cache_metadata_catalogues.md) | Pair rows repeatedly store or process the same entity history. |
| `temporal_video_storage_representation` | [cache_storage/video_cache_and_storage_layouts.md](techniques/cache_storage/video_cache_and_storage_layouts.md) | Separate image files or raw arrays multiply storage and loading work. |
| `video_subclipping` | [cache_storage/video_cache_and_storage_layouts.md](techniques/cache_storage/video_cache_and_storage_layouts.md) | Short annotations repeatedly seek through long source videos. |
| `file_page_cache_residency_policy` | [cache_storage/cache_reads_and_lookup_concurrency.md](techniques/cache_storage/cache_reads_and_lookup_concurrency.md) | Logical cache hits still reach disks, or resident file pages cause memory pressure. |
| `row_group_statistics_pruning` | [cache_storage/cache_metadata_catalogues.md](techniques/cache_storage/cache_metadata_catalogues.md) | Blocks are read and decompressed only to reject their rows. |
| `shared_source_frame_filter_views` | [cache_storage/cache_metadata_catalogues.md](techniques/cache_storage/cache_metadata_catalogues.md) | Several filtered datasets copy the same metadata table instead of sharing it. |
| `episode_locality_with_shuffle` | [cache_storage/video_cache_and_storage_layouts.md](techniques/cache_storage/video_cache_and_storage_layouts.md) | Random video-frame order repeatedly seeks/reopens episode videos while training still needs shuffling. |
| `persistent_pid_owned_manifest_handles` | [cache_storage/cache_reuse_and_sample_locality.md](techniques/cache_storage/cache_reuse_and_sample_locality.md) | Per-row manifest reopening repeatedly pays expensive setup. |
| `intra_frame_storage_random_access` | [cache_storage/video_cache_and_storage_layouts.md](techniques/cache_storage/video_cache_and_storage_layouts.md) | Sparse video-frame requests repeatedly decode from distant keyframes. |
| `chunkwise_cache_scan` | [cache_storage/cache_payload_encoding.md](techniques/cache_storage/cache_payload_encoding.md) | Joining cached JSON-lines chunks and rescanning for checksums dominates parsing. |
| `bounded_reusable_media_cache` | [cache_storage/cache_reuse_and_sample_locality.md](techniques/cache_storage/cache_reuse_and_sample_locality.md) | Repeated timestamp/frame windows reopen the same video while retained readers/decoders exhaust memory. |

### Process runtime environment

| ID | File | One-line trigger |
|---|---|---|
| `allocator_retain_freed_pages` | [process_runtime/memory_allocators.md](techniques/process_runtime/memory_allocators.md) | Workers repeatedly allocate and free large buffers, producing high page-fault or kernel time. |
| `cap_native_thread_pools` | [process_runtime/executors_admission.md](techniques/process_runtime/executors_admission.md) | Native pool fan-out or lock spinning wastes host CPU across loader workers; image-library threading can also slow a threaded path. |
| `persistent_pipeline_and_workers` | [process_runtime/resource_lifetime.md](techniques/process_runtime/resource_lifetime.md) | Warm epochs are slow, or worker/session startup recurs at each epoch. |
| `process_workers_for_gil_bound_rows` | [process_runtime/process_layout_and_gil.md](techniques/process_runtime/process_layout_and_gil.md) | A threads-only path is slower while decoding, serialization or augmentation holds the GIL, or main-process input work delays dispatch. |
| `epoch_cycle_collection` | [process_runtime/garbage_collection.md](techniques/process_runtime/garbage_collection.md) | GPU memory grows through reference cycles, or garbage collection stalls batch production. |
| `split_io_executor_from_cpu_stages` | [process_runtime/executors_admission.md](techniques/process_runtime/executors_admission.md) | Fetch is slow while decode time matches the reference, and blocking reads may be queuing behind CPU stages in a shared executor. |
| `asymmetric_loader_pool_sizing` | [process_runtime/executors_admission.md](techniques/process_runtime/executors_admission.md) | A remote preprocessing pool is sized to mean demand even though trainer starvation costs more than idle workers. |
| `freeze_long_lived_python_objects` | [process_runtime/garbage_collection.md](techniques/process_runtime/garbage_collection.md) | Garbage collection repeatedly scans stable training objects or inherited worker objects. |
| `memory_gated_cycle_collection` | [process_runtime/garbage_collection.md](techniques/process_runtime/garbage_collection.md) | Explicit garbage collection runs frequently despite ample memory. |
| `protect_trainer_cpu_scheduling` | [process_runtime/host_cpu_isolation.md](techniques/process_runtime/host_cpu_isolation.md) | Co-located input workers starve the trainer threads that schedule GPU work. |
| `memory_bandwidth_topology_isolation` | [process_runtime/host_cpu_isolation.md](techniques/process_runtime/host_cpu_isolation.md) | The same work costs more cycles per batch without an I/O explanation. |
| `free_threaded_python_loader` | [process_runtime/process_layout_and_gil.md](techniques/process_runtime/process_layout_and_gil.md) | GIL-bound threaded stages use few cores, and process workers would be too costly. |
| `native_binding_gil_release` | [process_runtime/process_layout_and_gil.md](techniques/process_runtime/process_layout_and_gil.md) | A native scan running on a Python thread still stalls other Python work. |
| `hoist_decorated_context_factories` | [process_runtime/row_dispatch_and_handoff.md](techniques/process_runtime/row_dispatch_and_handoff.md) | Free-threaded callbacks resume late while decorator or context construction recurs on the hot path. |
| `timeout_sentinel_not_exception` | [process_runtime/garbage_collection.md](techniques/process_runtime/garbage_collection.md) | Repeated read timeouts coincide with growing GPU memory. |
| `immediate_purge_avoids_madv_free_loss` | [process_runtime/memory_allocators.md](techniques/process_runtime/memory_allocators.md) | Native loaders segfault during lazy `MADV_FREE` purge. |
| `fetch_admission_backpressure` | [process_runtime/executors_admission.md](techniques/process_runtime/executors_admission.md) | A producer rejects fetch requests, and retries consume CPU. |
| `completion_first_memory_scheduling` | [process_runtime/executors_admission.md](techniques/process_runtime/executors_admission.md) | Admission control reads stale completed-but-uncollected results as memory pressure; total charged memory is not gated. |
| `zswap_aware_loader_memory_budget` | [process_runtime/memory_allocators.md](techniques/process_runtime/memory_allocators.md) | Memory-bound producer admission limits work on hosts with compressed swap already enabled and usable headroom. |
| `bypass_multiprocessing_for_light_work` | [process_runtime/row_dispatch_and_handoff.md](techniques/process_runtime/row_dispatch_and_handoff.md) | Queue overhead or worker startup dominates light input work, including evaluation starts and checkpoint restore. |
| `completion_order_without_head_of_line_blocking` | [process_runtime/row_dispatch_and_handoff.md](techniques/process_runtime/row_dispatch_and_handoff.md) | Ready batches wait behind slower earlier batches because output is ordered. |
| `minimize_trainer_gil_reacquisition` | [process_runtime/process_layout_and_gil.md](techniques/process_runtime/process_layout_and_gil.md) | The producer queue is full, but the trainer waits at batch handoff. |
| `acquire_transform_same_worker` | [process_runtime/row_dispatch_and_handoff.md](techniques/process_runtime/row_dispatch_and_handoff.md) | Raw media bytes dominate process-boundary traffic. |
| `cheap_batch_tree_type_checks` | [process_runtime/row_dispatch_and_handoff.md](techniques/process_runtime/row_dispatch_and_handoff.md) | Batch traversal builds attribute lists to detect named tuples. |
| `shallow_copy_owned_rows` | [process_runtime/row_dispatch_and_handoff.md](techniques/process_runtime/row_dispatch_and_handoff.md) | `deepcopy` dominates row handoff. |
| `gil_free_window_scheduling` | [process_runtime/process_layout_and_gil.md](techniques/process_runtime/process_layout_and_gil.md) | Loader-thread conversion and pickling hold the GIL in long C calls, freezing Python preparation before native evaluation. |
| `bound_allocator_arena_count` | [process_runtime/memory_allocators.md](techniques/process_runtime/memory_allocators.md) | Allocator arenas multiply across workers. |
| `epoch_scoped_async_resource_lifetime` | [process_runtime/resource_lifetime.md](techniques/process_runtime/resource_lifetime.md) | Memory grows as abandoned iterators or tasks retain epoch data. |
| `batch_owned_stateful_transform_buffers` | [process_runtime/resource_lifetime.md](techniques/process_runtime/resource_lifetime.md) | Shared stateful transform buffers mix carry-over between batch indices, causing incorrect chunks and preparation spikes. |
| `close_unlinked_media_descriptors` | [process_runtime/resource_lifetime.md](techniques/process_runtime/resource_lifetime.md) | Deleted temporary media files remain held open. |
| `native_codec_process_safety` | [process_runtime/resource_lifetime.md](techniques/process_runtime/resource_lifetime.md) | Duplicate native codec linkage crashes decoding, or corrupt media crashes or hangs native metadata probes. |

### Batch assembly and transport

| ID | File | One-line trigger |
|---|---|---|
| `collate_into_shared_memory` | [batch_assembly_transport/batch_shared_storage_lifetimes.md](techniques/batch_assembly_transport/batch_shared_storage_lifetimes.md) | Large worker batches pay private-to-shared copies or serialized array IPC. |
| `recycled_output_slots` | [batch_assembly_transport/batch_shared_storage_lifetimes.md](techniques/batch_assembly_transport/batch_shared_storage_lifetimes.md) | Fresh shared-output or device-buffer allocation recurs on every batch. |
| `registered_pinned_slots` | [batch_assembly_transport/batch_shared_storage_lifetimes.md](techniques/batch_assembly_transport/batch_shared_storage_lifetimes.md) | Large subprocess-hop and host-pinning copies delay GPU consumption. |
| `tail_only_padding` | [batch_assembly_transport/batch_numeric_collation.md](techniques/batch_assembly_transport/batch_numeric_collation.md) | Padding initializes a full output before rows overwrite most of it. |
| `batch_indices_before_boundary` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Per-row metadata or cheap item work repeatedly crosses worker/executor boundaries. |
| `vectorized_collate_steps` | [batch_assembly_transport/batch_numeric_collation.md](techniques/batch_assembly_transport/batch_numeric_collation.md) | Per-example objects, scalar tensors or row-by-feature loops dominate collation. |
| `sink_buffer_depth` | [batch_assembly_transport/batch_delivery_topology.md](techniques/batch_assembly_transport/batch_delivery_topology.md) | Ready-batch jitter empties the sink, or oversized buffers exhaust resident memory. |
| `broadcast_load_in_replica_group` | [batch_assembly_transport/batch_delivery_topology.md](techniques/batch_assembly_transport/batch_delivery_topology.md) | Data-identical ranks redundantly read and decode the same microbatch. |
| `buffered_length_bucketing` | [batch_assembly_transport/batch_workload_scheduling.md](techniques/batch_assembly_transport/batch_workload_scheduling.md) | Variable-length iterable inputs cause preprocessing outliers and collective waiting. |
| `hybrid_local_remote_preprocessing` | [batch_assembly_transport/batch_delivery_topology.md](techniques/batch_assembly_transport/batch_delivery_topology.md) | Remote preprocessing dominates cost while trainer-host CPUs remain underused. |
| `rank_local_complete_input_pipeline` | [batch_assembly_transport/batch_delivery_topology.md](techniques/batch_assembly_transport/batch_delivery_topology.md) | A centralized source/collator bottlenecks many trainer hosts. |
| `payload_cut_and_native_rate_metadata` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Unused fields or oversampled metadata inflate assembly, IPC and device-transfer bytes. |
| `native_numeric_buffer_bridge` | [batch_assembly_transport/batch_numeric_collation.md](techniques/batch_assembly_transport/batch_numeric_collation.md) | Numeric ingestion or reductions create large nested Python object graphs before arrays. |
| `broadcast_constant_frame_metadata` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Frame-constant metadata is repeatedly materialized as dense lists or arrays. |
| `direct_thread_safe_output_handoff` | [batch_assembly_transport/batch_delivery_topology.md](techniques/batch_assembly_transport/batch_delivery_topology.md) | Already-ready terminal batches pay event-loop submission and resumption latency. |
| `multiplex_sources_before_shared_stages` | [batch_assembly_transport/batch_delivery_topology.md](techniques/batch_assembly_transport/batch_delivery_topology.md) | Same-type datasets duplicate stage graphs and compete for threads. |
| `linear_keyed_jagged_slicing` | [batch_assembly_transport/batch_numeric_collation.md](techniques/batch_assembly_transport/batch_numeric_collation.md) | Keyed jagged slicing repeatedly concatenates growing per-key prefixes. |
| `reuse_column_vectors_in_feature_merge` | [batch_assembly_transport/batch_numeric_collation.md](techniques/batch_assembly_transport/batch_numeric_collation.md) | Feature merging copies usable normalized column payloads. |
| `torch_filesystem_tensor_sharing` | [batch_assembly_transport/batch_shared_storage_lifetimes.md](techniques/batch_assembly_transport/batch_shared_storage_lifetimes.md) | An existing PyTorch tensor-IPC baseline has descriptor-heavy queue spikes under light work. |
| `background_shape_grouped_rebatching` | [batch_assembly_transport/batch_workload_scheduling.md](techniques/batch_assembly_transport/batch_workload_scheduling.md) | Main-thread shape grouping, batch assembly or IPC causes large variable pauses. |
| `workload_weighted_rebatching` | [batch_assembly_transport/batch_workload_scheduling.md](techniques/batch_assembly_transport/batch_workload_scheduling.md) | Equal-size batches have unequal history/sequence work, and task-safe regrouping can balance heavy and light items. |
| `token_budget_balanced_packs` | [batch_assembly_transport/batch_workload_scheduling.md](techniques/batch_assembly_transport/batch_workload_scheduling.md) | Fixed sample counts create unequal token, frame or pixel work across ranks. |
| `per_step_global_length_election` | [batch_assembly_transport/batch_workload_scheduling.md](techniques/batch_assembly_transport/batch_workload_scheduling.md) | Independently sampled video lengths make synchronized steps wait for longest-rank work. |
| `round_packed_length_to_bucket` | [batch_assembly_transport/batch_workload_scheduling.md](techniques/batch_assembly_transport/batch_workload_scheduling.md) | Variable packed lengths create excessive shape specialization or compiler warmup. |
| `immutable_metadata_bank_ids` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Stable repeated side tensors dominate IPC. |
| `bitpacked_mask_transport` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Binary masks dominate IPC, pinning or H2D bytes. |
| `compact_transport_metadata_dtypes` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Range-bounded counts and indices consume large transfer payloads. |
| `same_dtype_concat_then_cast` | [batch_assembly_transport/batch_numeric_collation.md](techniques/batch_assembly_transport/batch_numeric_collation.md) | Concatenation copies and converts individual homogeneous pieces repeatedly. |
| `coalesce_tensor_records_before_ipc` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Thousands of tensor objects and storage descriptors dominate batch IPC. |
| `macro_loader_batches_micro_steps` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Input/transfer overhead dominates small optimizer steps. |
| `typed_native_batch_boundary` | [batch_assembly_transport/batch_compact_payloads.md](techniques/batch_assembly_transport/batch_compact_payloads.md) | Generic dynamic batch arguments incur repeated per-feature native/scripted conversion. |
| `per_consumer_shared_reader_queues` | [batch_assembly_transport/batch_delivery_topology.md](techniques/batch_assembly_transport/batch_delivery_topology.md) | One slow independent inference consumer backpressures every shared-reader consumer. |

### GPU placement and transfer

| ID | File | One-line trigger |
|---|---|---|
| `gpu_terminal_collate` | [gpu_placement_transfer/gpu_input_placement.md](techniques/gpu_placement_transfer/gpu_input_placement.md) | Final CPU assembly or frontend transforms duplicate host copies and starve compute. |
| `h2d_cache_depth_and_threads` | [gpu_placement_transfer/pinned_transfer_ownership.md](techniques/gpu_placement_transfer/pinned_transfer_ownership.md) | Serial IPC/H2D or background-copy lifetime/thread proliferation delays or corrupts training. |
| `pinned_d2h_outputs` | [gpu_placement_transfer/transfer_calls_and_readbacks.md](techniques/gpu_placement_transfer/transfer_calls_and_readbacks.md) | Pageable output copies stall the GPU-feeding thread for seconds. |
| `numa_local_input_placement` | [gpu_placement_transfer/gpu_input_placement.md](techniques/gpu_placement_transfer/gpu_input_placement.md) | Input allocation and transfer cross NUMA sockets and create remote-memory stalls or rank skew. |
| `preserve_uint8_until_normalization` | [gpu_placement_transfer/gpu_input_placement.md](techniques/gpu_placement_transfer/gpu_input_placement.md) | Early float conversion expands IPC and H2D media payloads fourfold. |
| `record_consumer_stream_lifetime` | [gpu_placement_transfer/pinned_transfer_ownership.md](techniques/gpu_placement_transfer/pinned_transfer_ownership.md) | Cross-stream storage reuse corrupts transferred or converted tensors. |
| `gpu_resident_dataset_indexing` | [gpu_placement_transfer/gpu_input_placement.md](techniques/gpu_placement_transfer/gpu_input_placement.md) | A dataset fits GPU memory but batches repeatedly collate and transfer it from CPU. |
| `pin_buffers_on_rank_device` | [gpu_placement_transfer/pinned_transfer_ownership.md](techniques/gpu_placement_transfer/pinned_transfer_ownership.md) | Background registration creates unintended GPU-zero contexts for nonzero ranks. |
| `pin_memory_effectiveness_audit` | [gpu_placement_transfer/pinned_transfer_ownership.md](techniques/gpu_placement_transfer/pinned_transfer_ownership.md) | Pinning is configured but custom containers or reconstruction deliver pageable H2D sources. |
| `supply_known_gpu_output_sizes` | [gpu_placement_transfer/transfer_calls_and_readbacks.md](techniques/gpu_placement_transfer/transfer_calls_and_readbacks.md) | GPU output construction reads totals back to CPU when exact sizes are already known. |
| `transfer_unique_consumed_storages` | [gpu_placement_transfer/transfer_calls_and_readbacks.md](techniques/gpu_placement_transfer/transfer_calls_and_readbacks.md) | Unused fields, exact-object aliases or small strided views cause redundant device copies or oversized pinning. |
| `coalesce_small_h2d_copies` | [gpu_placement_transfer/transfer_calls_and_readbacks.md](techniques/gpu_placement_transfer/transfer_calls_and_readbacks.md) | Host submission for many small copies delays the consumer, and bounded packing can cost less than the calls it removes. |
| `bounded_native_h2d_batch_calls` | [gpu_placement_transfer/transfer_calls_and_readbacks.md](techniques/gpu_placement_transfer/transfer_calls_and_readbacks.md) | Many small pinned-tensor copy submissions dominate host H2D time. |
| `pinned_small_d2h_readbacks` | [gpu_placement_transfer/transfer_calls_and_readbacks.md](techniques/gpu_placement_transfer/transfer_calls_and_readbacks.md) | Small pageable metadata readbacks serialize GPU collation. |

### Transform compute

| ID | File | One-line trigger |
|---|---|---|
| `per_row_cpu_generator_seeding` | [transform_compute/row_rng_and_identity_sampling.md](techniques/transform_compute/row_rng_and_identity_sampling.md) | Worker-dependent augmentation streams obstruct reproducible scheduling changes; seed each row using stable coordinates and rank. |
| `select_view_and_fields_before_tensorize` | [transform_compute/image_transform_work_elimination.md](techniques/transform_compute/image_transform_work_elimination.md) | Unused views and fields are decoded, tensorized or augmented before being discarded. |
| `move_trainer_side_preproc_off_critical_path` | [transform_compute/transform_placement_and_chunking.md](techniques/transform_compute/transform_placement_and_chunking.md) | Synchronous trainer-side input preparation stalls steps while loader capacity is spare. |
| `fused_per_feature_preproc_pass` | [transform_compute/feature_preparation_and_materialization.md](techniques/transform_compute/feature_preparation_and_materialization.md) | Many small feature operators repeatedly traverse the same decoded input. |
| `parse_shared_fields_once` | [transform_compute/feature_preparation_and_materialization.md](techniques/transform_compute/feature_preparation_and_materialization.md) | Multiple targets parse the same row metadata independently. |
| `allocation_free_preproc_loops` | [transform_compute/feature_preparation_and_materialization.md](techniques/transform_compute/feature_preparation_and_materialization.md) | Temporary allocations, deep copies or exact-size buffer growth dominate preprocessing. |
| `lower_resolution_forward_preprocessing` | [transform_compute/image_transform_work_elimination.md](techniques/transform_compute/image_transform_work_elimination.md) | Dense high-resolution reprojection dominates input CPU and starves GPUs. |
| `lean_cpu_augmentation` | [transform_compute/image_transform_work_elimination.md](techniques/transform_compute/image_transform_work_elimination.md) | Simple CPU augmentations pay generic tensor dispatch and redundant scaling. |
| `offline_pretokenization` | [transform_compute/input_encoder_reuse_and_packing.md](techniques/transform_compute/input_encoder_reuse_and_packing.md) | On-the-fly input encoding creates rank imbalance and low GPU utilization. |
| `stateful_chunk_encoding` | [transform_compute/input_encoder_reuse_and_packing.md](techniques/transform_compute/input_encoder_reuse_and_packing.md) | Overlapping input chunks repeatedly encode the same left context. |
| `gil_releasing_dataframe_decode` | [transform_compute/feature_preparation_and_materialization.md](techniques/transform_compute/feature_preparation_and_materialization.md) | Threaded DataFrame decode-and-transform stops scaling; compare a compatible backend and measure actual parallelism. |
| `scalar_timebase_without_fraction_objects` | [transform_compute/media_decode_work_reduction.md](techniques/transform_compute/media_decode_work_reduction.md) | Per-record rational objects are allocated only to obtain scalar timebases. |
| `bypass_empty_or_identity_stages` | [transform_compute/feature_preparation_and_materialization.md](techniques/transform_compute/feature_preparation_and_materialization.md) | Empty stage lists or identity masks and geometry still execute scaffolding. |
| `zero_copy_raw_wav_window_views` | [transform_compute/media_decode_work_reduction.md](techniques/transform_compute/media_decode_work_reduction.md) | In-memory PCM uses general demux and decode for a sample window. |
| `decoder_admission_and_stage_split` | [transform_compute/media_decode_work_reduction.md](techniques/transform_compute/media_decode_work_reduction.md) | Demux latency increases nonlinearly with admission concurrency. |
| `in_memory_encoded_media_decode` | [transform_compute/media_decode_work_reduction.md](techniques/transform_compute/media_decode_work_reduction.md) | Encoded media is written to temporary files and reopened under filesystem pressure. |
| `decode_scale_sample_fusion` | [transform_compute/media_decode_work_reduction.md](techniques/transform_compute/media_decode_work_reduction.md) | High-resolution frames are buffered only to be resized and sparsely retained. |
| `offset_only_nonempty_mask` | [transform_compute/feature_preparation_and_materialization.md](techniques/transform_compute/feature_preparation_and_materialization.md) | String filtering scans payload bytes solely to test emptiness. |
| `constant_time_identity_sampling` | [transform_compute/row_rng_and_identity_sampling.md](techniques/transform_compute/row_rng_and_identity_sampling.md) | Fallback sampling linearly scans a huge identity pool for every sample. |
| `blockwise_transform_temporaries` | [transform_compute/image_transform_work_elimination.md](techniques/transform_compute/image_transform_work_elimination.md) | Whole-image temporaries and page faults dominate transform memory and CPU. |
| `coalesce_chunk_dedup_readbacks` | [transform_compute/transform_placement_and_chunking.md](techniques/transform_compute/transform_placement_and_chunking.md) | Every transform chunk incurs a CPU readback for deduplication. |
| `gpu_transform_chunk_granularity` | [transform_compute/transform_placement_and_chunking.md](techniques/transform_compute/transform_placement_and_chunking.md) | Small input-transform chunks multiply device launches. |
| `cpu_precompute_sync_heavy_inputs` | [transform_compute/transform_placement_and_chunking.md](techniques/transform_compute/transform_placement_and_chunking.md) | Tiny device input operations pay disproportionate launch and synchronization cost. |
| `offsets_only_sparse_normalization` | [transform_compute/feature_preparation_and_materialization.md](techniques/transform_compute/feature_preparation_and_materialization.md) | Sparse normalization repeatedly constructs ranges and casts offsets. |
| `bounded_sparse_video_decode_plan` | [transform_compute/media_decode_work_reduction.md](techniques/transform_compute/media_decode_work_reduction.md) | Sparse frame requests repeatedly seek overlapping video segments. |
| `sequence_packing_with_segment_isolation` | [transform_compute/input_encoder_reuse_and_packing.md](techniques/transform_compute/input_encoder_reuse_and_packing.md) | Padding independent documents wastes encoder work. |
| `resize_and_cap_before_expansion` | [transform_compute/image_transform_work_elimination.md](techniques/transform_compute/image_transform_work_elimination.md) | Expansion materializes pixels or media that later stages discard. |

### Training-loop boundaries and scale

| ID | File | One-line trigger |
|---|---|---|
| `distributed_evaluation_sharding` | [loop_boundaries_scale/evaluation_boundary_work.md](techniques/loop_boundaries_scale/evaluation_boundary_work.md) | Every rank evaluates the full validation set. |
| `epoch_boundary_work_audit` | [loop_boundaries_scale/evaluation_boundary_work.md](techniques/loop_boundaries_scale/evaluation_boundary_work.md) | Fixed per-epoch callbacks, exports, barriers or metric work leave GPUs idle after training steps become faster. |
| `fewer_gpus_same_global_batch` | [loop_boundaries_scale/rank_coordination_and_scaling.md](techniques/loop_boundaries_scale/rank_coordination_and_scaling.md) | A loader-fixed job has host headroom, and per-GPU throughput dominates cost. |
| `rank0_side_work_off_critical_path` | [loop_boundaries_scale/rank_coordination_and_scaling.md](techniques/loop_boundaries_scale/rank_coordination_and_scaling.md) | Rank 0 straggles on side I/O or loader-statistics collectives. |
| `async_eval_input_and_output_sinks` | [loop_boundaries_scale/evaluation_boundary_work.md](techniques/loop_boundaries_scale/evaluation_boundary_work.md) | The evaluation main thread blocks on input loads or output writes. |
| `bounded_chunked_evaluation_input` | [loop_boundaries_scale/evaluation_boundary_work.md](techniques/loop_boundaries_scale/evaluation_boundary_work.md) | Whole-video decoding inflates host memory. |
| `reuse_distributed_setup_group` | [loop_boundaries_scale/rank_coordination_and_scaling.md](techniques/loop_boundaries_scale/rank_coordination_and_scaling.md) | The loader creates an extra process group for control messages. |
| `keep_batch_control_state_on_cpu` | [loop_boundaries_scale/loader_checkpoint_progress.md](techniques/loop_boundaries_scale/loader_checkpoint_progress.md) | CPU batch-control logic reads a GPU-resident scalar, forcing synchronization. |
| `reset_progress_for_nonstateful_resume` | [loop_boundaries_scale/loader_checkpoint_progress.md](techniques/loop_boundaries_scale/loader_checkpoint_progress.md) | A restarted input cursor and restored mid-epoch progress can disagree; resetting the counter is not exact resume. |
| `throttle_loader_state_snapshots` | [loop_boundaries_scale/loader_checkpoint_progress.md](techniques/loop_boundaries_scale/loader_checkpoint_progress.md) | Loader state is rebuilt or synchronized on every step. |
| `remove_debug_device_syncs` | [loop_boundaries_scale/rank_coordination_and_scaling.md](techniques/loop_boundaries_scale/rank_coordination_and_scaling.md) | Timing code synchronizes the device in every phase. |
| `index_only_loader_checkpoint` | [loop_boundaries_scale/loader_checkpoint_progress.md](techniques/loop_boundaries_scale/loader_checkpoint_progress.md) | Checkpoints include immutable cached payloads. |
| `single_rank_input_sync_fastpath` | [loop_boundaries_scale/rank_coordination_and_scaling.md](techniques/loop_boundaries_scale/rank_coordination_and_scaling.md) | A single-rank job runs end-of-data synchronization collectives. |

### Startup

| ID | File | One-line trigger |
|---|---|---|
| `bounded_parallel_worker_spawn` | [startup/worker_launch_and_imports.md](techniques/startup/worker_launch_and_imports.md) | Workers are started one at a time, delaying the first batch; serialization under spawn can compound the cost. |
| `spawn_mmaps_by_descriptor` | [startup/startup_payload_loading.md](techniques/startup/startup_payload_loading.md) | Spawn serialization copies full mappings or record graphs into each child. |
| `intern_native_singleton_wrappers` | [startup/loader_graph_initialization.md](techniques/startup/loader_graph_initialization.md) | Duplicate native-singleton wrappers create quadratic initialization or cleanup. |
| `flat_transform_module_scripting` | [startup/loader_graph_initialization.md](techniques/startup/loader_graph_initialization.md) | Thousands of repeated per-feature modules delay input-transform scripting. |
| `parallel_independent_decode_units` | [startup/startup_payload_loading.md](techniques/startup/startup_payload_loading.md) | Serial decode of independently addressable compressed groups delays startup. |
| `preloaded_forkserver_workers` | [startup/worker_launch_and_imports.md](techniques/startup/worker_launch_and_imports.md) | Each spawned worker repeats heavy CPU-only imports. |
| `fork_safe_worker_bootstrap` | [startup/worker_launch_and_imports.md](techniques/startup/worker_launch_and_imports.md) | Repeated launcher imports, inherited thread locks, or costly worker bootstrap require workload-scoped start-method and lifecycle fixes. |
| `lazy_budgeted_eval_source` | [startup/startup_payload_loading.md](techniques/startup/startup_payload_loading.md) | Eager evaluation materializes a large corpus before training despite a small per-cadence budget. |
| `pickle_native_loader_graph` | [startup/loader_graph_initialization.md](techniques/startup/loader_graph_initialization.md) | Dynamic callable capture dominates loader graph serialization and bootstrap. |

### Model execution adjacent to the input path

| ID | File | One-line trigger |
|---|---|---|
| `device_resident_matching` | [model_execution.md](techniques/model_execution.md) | Per-sample CPU matching blocks the training host on device-to-host copies. |
| `reduced_precision_compute` | [model_execution.md](techniques/model_execution.md) | Traces or matched cached runs identify exposed model compute cost. |
| `norm_layout_materialization` | [model_execution.md](techniques/model_execution.md) | Layout conversions or strided operands make an evidenced model region costly. |
| `fused_copy_widen` | [model_execution.md](techniques/model_execution.md) | Separate materialization and BF16-to-FP32 widening of strided device activations duplicate copy/conversion work. |
| `selective_kernel_replacement` | [model_execution.md](techniques/model_execution.md) | A specific operator/shape has exposed cost in the measured path. |

### Negative results

| ID | File | One-line trigger |
|---|---|---|
| `loader_swap_without_data_layout` | [negative_results/loader_layout_and_remote_offload_counterexamples.md](techniques/negative_results/loader_layout_and_remote_offload_counterexamples.md) | A loader replacement OOMs on inline images or remains slow after lookup conversion because transform threading still contends. |
| `shared_jit_transformer_negative` | [negative_results/component_speed_and_materialization_counterexamples.md](techniques/negative_results/component_speed_and_materialization_counterexamples.md) | Sharing compiled preprocessing modules saves memory but can add latency without improving throughput. |
| `component_win_without_training_win` | [negative_results/component_speed_and_materialization_counterexamples.md](techniques/negative_results/component_speed_and_materialization_counterexamples.md) | A faster input component does not improve measured training throughput. |
| `python_materialization_erases_parser_gain` | [negative_results/component_speed_and_materialization_counterexamples.md](techniques/negative_results/component_speed_and_materialization_counterexamples.md) | Immediate Python-object expansion erases a faster native parser's benefit. |
| `throughput_not_utilization` | [negative_results/component_speed_and_materialization_counterexamples.md](techniques/negative_results/component_speed_and_materialization_counterexamples.md) | Aggregation settings are chosen by GPU utilization rather than useful completion. |
| `byte_budgeted_episode_shuffle` | [negative_results/buffering_and_sample_mixing_counterexamples.md](techniques/negative_results/buffering_and_sample_mixing_counterexamples.md) | Each worker processes one episode at a time, limiting batch diversity; buffering more episodes costs host memory. |
| `worker_local_packing_quality_gate` | [negative_results/buffering_and_sample_mixing_counterexamples.md](techniques/negative_results/buffering_and_sample_mixing_counterexamples.md) | Changing the number of packing workers can hurt model quality even when the overall mixture and training loss match. |
| `length_sorted_dynamic_padding` | [negative_results/component_speed_and_materialization_counterexamples.md](techniques/negative_results/component_speed_and_materialization_counterexamples.md) | Length sorting shortens local iteration timings without improving wall-clock progress in the measured window. |
| `fused_stage_split_negative` | [negative_results/loader_layout_and_remote_offload_counterexamples.md](techniques/negative_results/loader_layout_and_remote_offload_counterexamples.md) | Splitting efficient fused work introduces scheduling or handoff overhead that can outweigh overlap gains, especially with short training steps. |
| `local_cache_remote_offload_negative` | [negative_results/loader_layout_and_remote_offload_counterexamples.md](techniques/negative_results/loader_layout_and_remote_offload_counterexamples.md) | Remote CPU/cache offload is proposed over an efficient local filesystem cache. |
| `distributed_model_tuning_negative` | [negative_results/trainer_pauses_and_communication_counterexamples.md](techniques/negative_results/trainer_pauses_and_communication_counterexamples.md) | Communication settings (bucket size, static graph, bucket views) are proposed once communication becomes a larger share after input bottlenecks are reduced. |
| `forced_gc_without_tail_negative` | [negative_results/trainer_pauses_and_communication_counterexamples.md](techniques/negative_results/trainer_pauses_and_communication_counterexamples.md) | Periodic forced collection is proposed without a recurring steady-state GC tail or unexplained late spikes. |
| `well_fed_transport_change_negative` | [negative_results/buffering_and_sample_mixing_counterexamples.md](techniques/negative_results/buffering_and_sample_mixing_counterexamples.md) | A larger sink or shared transport is proposed while boundary queues already feed most ranks and only the slowest rank lags. |
| `process_concurrency_extremes_negative` | [negative_results/worker_concurrency_counterexamples.md](techniques/negative_results/worker_concurrency_counterexamples.md) | Process count and per-process item concurrency are treated as interchangeable when only their product is held fixed. |
| `worker_saturation_negative` | [negative_results/worker_concurrency_counterexamples.md](techniques/negative_results/worker_concurrency_counterexamples.md) | Input remains limiting while added workers approach CPU, non-reclaimable-memory or shared-storage limits; total memory occupancy alone is insufficient. |

## Maintenance

Add a new stable ID only for a distinct mechanism, with a plain-language trigger, concrete intervention and essential safety/quality guards. Merge duplicate support into existing IDs; do not rename them. Keep all organization-specific names and evidence links outside the shared catalog. Explain the input/output representation, concrete intervention and guards so an OSS reader can understand and apply the idea without opening a private source diff; add a small reference example when the mechanism is otherwise ambiguous. Public SPDL API references are options to compare, not required replacements. For counterexamples, separate what was tried from the outcome and conditions for revisiting it; mark unrun proposals as **Not tested**. Omit historical gain summaries; keep source links as optional provenance rather than required reading.

The index is a shortlist, not the full explanation. In an area entry, explain the usual situation and input representation, why work is repeated or expensive, and how the proposed change removes it. Trigger and Technique can be longer when that makes the reasoning clear to a fresh reader; use a concrete before/after example when helpful. Make Guards follow naturally from that reasoning while retaining unique correctness, lifetime, resource, quality and negative-result checks. Do not replace the explanation with a dense checklist of technical terms.

Keep this entry point screenable: put complete entries in the matching area file and update the ID-to-file table. New model-execution mechanisms belong in the separate [model optimization catalog](model_techniques/index.md); this input-focused catalog retains its historical input-adjacent entries without duplicating them. Model hypotheses can be investigated alongside input work; use the measured path for attribution and preserve quality gates. Every bundle consuming this index must include the sibling `techniques/` and `model_techniques/` directories and resolve their relative links.
