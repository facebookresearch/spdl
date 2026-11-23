# Compact batch payloads

This topic covers index dispatch, compact metadata, shared identifiers, bit packing, flat tensor records and typed boundaries. Open this topic for excessive payload bytes, objects or per-delivery overhead.

[Catalog index](../../technique_catalog.md) · [Batch assembly and transport](../batch_assembly_transport.md)

**`batch_indices_before_boundary`: batch compact indices before worker dispatch**
- Trigger: the main process serializes row dictionaries, materializes rows individually, or pays executor overhead for cheap per-item work.
- Technique: group compact row indices before the process or executor boundary, sending one index vector instead of a queue message and materialized dictionary per row. In the worker, select the same rows with one vectorized columnar `take` or equivalent batch lookup, then fill the batch from native buffers. For indices `[7, 2, 7]`, lookup must return row 7, row 2 and row 7 again in that order. Grouping dispatch amortizes messages; bulk lookup separately avoids repeated row extraction, and requires an API that actually supports it.
- Guards: preserve order, repeated indices, filtering, epoch boundaries and resume state, including the original row-to-sample mapping. Budget larger in-flight batches explicitly. Cached assembly timing does not measure uncached acquisition or complete training.
- SPDL reference: {py:meth}`spdl.pipeline.PipelineBuilder.aggregate` can group compact indices before {py:meth}`spdl.pipeline.PipelineBuilder.to` opens a {py:class}`spdl.pipeline.defs.ProcessPoolExecutorConfig` region. Fetch, transform and collate can stay worker-local; close the region with `MAIN_PROCESS` before the sink. `to(..., buffer_size=...)` groups transport, not training samples, and does not automatically vectorize the row lookup. The [architecture examples](../../architectures.md) provide an optional starting point.

**`payload_cut_and_native_rate_metadata`: transport only consumed information**
- Trigger: unused keys are still constructed, collated, pinned and transferred, or slowly changing metadata is expanded at the sample rate.
- Technique: trace the consumer's dependencies and remove unused fields before expensive assembly and IPC. For retained metadata, transport its original values and timing/index information rather than a dense repetition at every frame or sample; reconstruct the required expansion after the boundary with the existing alignment rule. For example, a clip-constant label can travel once with the clip length instead of once per frame. Use range-safe narrow dtypes. This reduces construction, copy and transfer bytes, but reconstruction remains part of the measured path.
- Guards: retain the full dependency closure for losses, transforms, evaluation and checkpoints. Check reconstruction, dtype and timebase semantics; reducing bytes does not authorize removing training information. Compare complete runs as well as steady steps.

**`broadcast_constant_frame_metadata`: retain broadcast views until materialization is needed**
- Trigger: metadata constant across frames or targets is repeatedly expanded into lists and arrays.
- Technique: keep the compact source and expose a broadcast view whose repeated dimensions address the same underlying values, rather than allocating one copy per frame or target. For example, a channel vector can be viewed as a frame × channel × target array with no new metadata payload for each frame/target. Defer dense materialization until an operator or transport actually requires contiguous owning storage. This removes repeated-value construction only along paths that accept the view; a serializer may still materialize it.
- Guards: a broadcast view is not independently writable storage. Preserve shape, timebase and reduction semantics; copy at an explicit mutation boundary. Include later materialization in the measurement rather than merely moving its cost.

**`immutable_metadata_bank_ids`: send identifiers for repeated immutable tensors**
- Trigger: repeated side tensors dominate interprocess traffic even though their content is stable.
- Technique: initialize a shared immutable rank-local bank once and assign stable identifiers to its entries. Send the identifier and any required slice descriptor with each batch instead of resending the tensor; the consumer resolves the identifier against the same manifest and gathers the needed views. Provide a normal payload fallback when bank identity or coverage does not match. This trades repeated per-batch bytes for one resident bank and lookup work; a gather that requires a new dense tensor still has a copy cost.
- Guards: verify manifest identity, mapping and bank lifetime. Isolate mutation, preserve the original batch contract and retain fallback behavior across evaluation and resume. Count the shared bank's resident memory rather than only the per-batch reduction.

**`bitpacked_mask_transport`: move binary masks in packed form**
- Trigger: binary masks dominate IPC, pinning or H2D bytes.
- Technique: pack boolean masks into bits before transport and unpack at the terminal GPU stage under an explicit shared representation contract. Each byte carries eight mask positions rather than one boolean per byte. For an illustrative least-significant-bit-first byte, `[True, False, True, False, False, True, False, True]` becomes `0b10100101`; position `i` is recovered as `bool((byte >> i) & 1)`. Carry the original dimensions and apply the same convention at both ends. This reduces transported bytes but adds pack/unpack work; the illustration defines a convention, not a GPU kernel or a requirement to use that bit order.
- Guards: require boolean input and supported width alignment; define bit order, padding bits and original dimensions at both ends. Preserve masks exactly and exercise evaluation as well as training. Include unpacking cost and do not assume byte reduction controls the complete input path.

**`compact_transport_metadata_dtypes`: narrow losslessly before transport**
- Trigger: counts or indices consume a large fraction of batch-transfer bytes despite bounded values.
- Technique: validate the full value range and use a lossless narrower integer representation before IPC and H2D, then widen only at consumers that require the original dtype. For example, counts known to lie in `[0, 65535]` can be represented in `uint16` if the transport and consumers support it; that representation cannot carry a negative sentinel. Fewer bytes travel across the boundary, but casting back and any consumer reconstruction remain part of the path.
- Guards: check overflow, signedness, sentinels and every downstream consumer, with a legacy fallback for unsupported ranges. Do not apply a lossy cast. Changes affecting distributed exchange need their own complete consumer-contract validation.

**`coalesce_tensor_records_before_ipc`: send flat tensors and offsets**
- Trigger: many tiny tensor objects and storage-descriptor transfers dominate each batch handoff.
- Technique: concatenate compatible record fields into a few flat tensors and send their offsets, shapes and field descriptors before IPC. The receiver reconstructs record views from that metadata without receiving one storage handle per original tensor. Illustratively, records `[[10, 11], [], [12]]` use `values=[10, 11, 12]` and `offsets=[0, 2, 2, 3]`; the repeated offset preserves the empty record. Keep separate buffers for incompatible dtypes or layouts. The saving is fewer objects/handles at the boundary, not necessarily fewer payload bytes or copy-free packing.
- Guards: preserve sample boundaries, keys, order, dtype and empty-record behavior. Define padding and view ownership explicitly and retain the original consumer contract. Include packing and reconstruction in the measurement; fewer handles alone are not sufficient.

**`macro_loader_batches_micro_steps`: amortize delivery without enlarging optimizer updates**
- Trigger: per-batch input and transfer overhead dominates small optimizer steps.
- Technique: load and transfer a macro-batch, then take consecutive micro-slices and run the original zero-grad → forward → backward → optimizer-update sequence separately for each slice. Do not accumulate all slices into one larger update. Loader/transfer setup is amortized across the slices, while the optimizer micro-step size and update count remain unchanged. Batch-dependent transforms or normalization must still operate at the original semantic boundary; merely slicing a macro-normalized result is not equivalent.
- Guards: preserve scheduler, callback, normalization and shuffle semantics. Track the unconsumed part of a macro-batch in checkpoint/resume state. Verify last-batch handling, update counts, sample conservation and quality; input batching must not silently change global batch or gradient weighting.

**`typed_native_batch_boundary`: bypass dynamic per-feature argument conversion**
- Trigger: converting generic `Dict[str, Any]` batch arguments taxes every feature at a native or scripted boundary.
- Technique: define stable typed feature groups and pass their tensors/buffers directly to a compatible native or scripted entry point. For example, a group can carry `(dense_tensor, sparse_values, sparse_offsets)` rather than repeatedly looking up and converting every field of a generic dictionary. Resolve schema and optional-field handling explicitly, retaining a dynamic fallback where needed. The saving is boundary lookup, dispatch and wrapper conversion; it does not remove feature computation or authorize changing what the native operator receives.
- Guards: preserve the exact container and output contract. Provide a fallback for missing or dynamic groups and unsupported schemas; do not specialize away valid optional fields. Include native-call and downstream costs when assessing the whole path.
