# Cache and storage

Choose the topic whose symptoms match the measured bottleneck. Open only that
reference; technique entries and their applicability guards live in the topic
files, not in this routing page.

These techniques reduce the bytes an input pipeline reads, the work it repeats, or the resources it keeps while preparing training data. A cache stores something already fetched or computed so a later request can reuse it. That something might be a downloaded file, a decoded record, metadata, or a live video reader; each has a different cost and lifetime. Measure first-time reads and repeated cache hits separately, including the cost of filling and retaining the cache.

Here, a **row** or **record** is one logical set of related values, such as a path, a label, and feature tensors. In a table, records are rows and the named values are **columns** or **fields**. A stored record is not necessarily one training sample: several records may form a sample, and one record may expand into several examples. Likewise, a cache entry may contain one record, a chunk, or a batch. Preserve these mappings when changing storage or batching.

[Catalog index](../technique_catalog.md)

## Cache payload encoding

- [Cache payload encoding](cache_storage/cache_payload_encoding.md): serialization, copies or incomplete cache entries dominate.

## Cache reads and lookup concurrency

- [Cache reads and lookup concurrency](cache_storage/cache_reads_and_lookup_concurrency.md): repeated misses, memory pressure or slow reads.

## Metadata catalogue layouts

- [Metadata catalogue layouts](cache_storage/cache_metadata_catalogues.md): metadata duplication or row preparation dominates.

## Cache reuse and sample locality

- [Cache reuse and sample locality](cache_storage/cache_reuse_and_sample_locality.md): repeated setup or poor locality defeats cache reuse.

## Video cache and storage layouts

- [Video cache and storage layouts](cache_storage/video_cache_and_storage_layouts.md): retained frames, video seeking or storage layout dominate.
