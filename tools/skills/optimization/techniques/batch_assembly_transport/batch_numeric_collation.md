# Numeric batch collation

This topic covers padding, vectorized assembly, native buffers, jagged slicing, column reuse and homogeneous casts. Open this topic when numeric construction or per-feature copies dominate.

[Catalog index](../../technique_catalog.md) · [Batch assembly and transport](../batch_assembly_transport.md)

**`tail_only_padding`: avoid initializing values that rows will overwrite**
- Trigger: padding fills an entire output with the pad value before copying rows over most of it.
- Technique: allocate the output without initializing it, copy row `i` into `output[i, :length_i]`, and fill only `output[i, length_i:]` with the padding value. The final contents are the same as fill-then-copy, but valid row positions are written once rather than twice. This can reduce memory traffic when rows occupy most of the padded window; it does not eliminate the row copy.
- Guards: write every output element before publishing the batch, including uncovered trailing dimensions and the entire window for an empty row. Preserve dtype, padding value, lengths, masks and the original truncation policy. Never expose uninitialized memory or a partially constructed buffer after an error, and do not reuse an output while consumers still reference it. Compare against the original collator, including short and empty rows; the avoided writes are greatest when rows nearly fill the window.

**Tail-only padding example.**

A NumPy reference implementation for rows of a common dtype and trailing shape:

```python
import numpy as np


def pad_rows(rows, max_length, pad_value=0):
    rows = list(rows)
    if not rows or max_length < 0 or rows[0].ndim < 1:
        raise ValueError("Expected nonempty rows and a nonnegative window")
    trailing_shape = rows[0].shape[1:]
    dtype = rows[0].dtype
    for row in rows:
        if row.ndim < 1 or row.shape[1:] != trailing_shape or row.dtype != dtype:
            raise ValueError("Rows must have the same dtype and trailing shape")
        if len(row) > max_length:
            raise ValueError("Row exceeds the window; handle truncation explicitly")
    out = np.empty((len(rows), max_length, *trailing_shape), dtype=dtype)
    for i, row in enumerate(rows):
        length = len(row)
        np.copyto(out[i, :length], row)
        out[i, length:] = pad_value
    return out
```

For rows of lengths three and one, a window of four and a padding value of zero produce `[a, b, c, 0]` and `[d, 0, 0, 0]`. The first row initializes one padding position, not four. This example allocates a private NumPy batch; direct construction in shared or pinned storage requires a separate allocation/ownership contract. There is no dedicated SPDL tail-padding API: a custom collator can be passed to {py:meth}`spdl.pipeline.PipelineBuilder.pipe` after batching, or retain the equivalent baseline implementation.

**`vectorized_collate_steps`: assemble a whole numeric batch rather than many tiny objects**
- Trigger: per-example DataFrames, scalar tensors, `.item()` round trips or Python row-by-feature loops dominate collation.
- Technique: keep compatible values in arrays, select the batch in bulk, and apply the same transform across its leading batch dimension. For fixed-width rows, `np.stack(rows)` followed by one compatible array-to-tensor conversion is a reference shape; for ragged rows, retain flat values and offsets or use a padding collator with explicit length/mask handling. Avoid materializing unused features. This is not permission to change a per-example random transform into one shared random draw.
- Guards: preserve missing/invalid-value rules, dtypes, integer bounds, feature order, transform order, bit-level packing, padding, masks and per-example augmentation randomness. Fall back for unsupported layouts or transforms. Preserve output ownership: a batch view cannot be mutated as though it were an independent copy. Compare representative outputs against the original collator and measure the whole row-to-batch path, not just one array operation. Worker-count, storage-format and device-offload bundles do not establish a collation-only benefit.
- SPDL reference: For compatible decoded media, {py:func}`spdl.io.convert_frames` accepts a sequence and creates a batch buffer; {py:func}`spdl.io.load_image_batch` combines loading and batch conversion for images. An optional {py:func}`spdl.io.cpu_storage` passed as `storage` can control the destination allocation; {py:func}`spdl.io.to_numpy` / {py:func}`spdl.io.to_torch` expose compatible buffers without an extra conversion copy. These are media-buffer reference paths, not a general replacement for numeric feature packing or baseline collation semantics. `cpu_storage` is not an IPC shared-memory arena.

**`native_numeric_buffer_bridge`: bypass nested Python numeric materialization**
- Trigger: numeric reads or grouped reductions first create many Python lists, dictionaries or scalar objects before constructing arrays.
- Technique: expose ownership-carrying native numeric buffers as array/tensor views instead of converting each scalar through Python. Variable-length rows use a flat values buffer and cumulative offsets: illustratively, `[[1, 2], [], [3, 4, 5]]` is `values=[1, 2, 3, 4, 5]`, `offsets=[0, 2, 2, 5]`, and row `i` occupies `values[offsets[i]:offsets[i+1]]`. Verified fixed-width records can instead be reshaped directly. Perform grouped reductions in native columnar operations while retaining null/validity information. The saving is avoided object allocation and conversion, not the elimination of required reads or reductions.
- Guards: preserve nulls, empty rows, offsets, schema, dtype and reduction semantics. Fixed-width reshape needs verified widths; views must keep their owners alive. Bulk ingestion and loader draining do not measure training end to end.

**`linear_keyed_jagged_slicing`: remove repeated per-key concatenation**
- Trigger: slicing keyed variable-length tensors repeatedly concatenates growing per-key data, introducing quadratic work.
- Technique: remove repeated per-key concatenation from the slicing path while preserving the consumer's keys, per-key/per-sample lengths, offsets and flat values. Each concatenation can copy an already assembled prefix again; removing that repeated work is the optimization, without prescribing a particular replacement algorithm. For illustration, in a key-major two-sample layout with keys `[a, b]` and lengths `[2, 0, 1, 2]`, selecting the second sample keeps lengths `[0, 2]` and only its two values for `b`. This example defines the required output, not an implementation recipe. Treat range-aware output assembly as a separate candidate requiring its own validation, not as an established part of eliminating repeated concatenation.
- Guards: check keys, lengths, offsets, empty features and ordering. In one comparison within a larger shared-buffer batching change, faster slicing reduced but did not eliminate a regression in the complete consumer path: encoding remained more expensive and CPU work per batch was higher. When reader concurrency is already capped, memory reductions need not offset that extra CPU work. Correctness checks on a combined path do not isolate the slicing algorithm, and a local slicing speedup is not a standalone end-to-end win. Validate slices and profile encoding and the complete consumer path before promoting the change.

**`reuse_column_vectors_in_feature_merge`: merge references instead of payload copies**
- Trigger: normalized features already reside in usable per-feature column buffers, but combining them into a batch/container copies every column's values again.
- Technique: build the merged container from references or lightweight descriptors for those existing vectors, rather than rematerializing their payloads. For example, the output can associate feature names with existing numeric vectors and their validity/offset buffers while leaving the bytes in place. The consumer still sees the required features in the original schema/order; only their storage assembly changes. This avoids a merge-time copy when later operators accept the retained encoding, not when the consumer requires a newly packed dense tensor.
- Guards: every downstream operator must accept the retained encoding and ownership contract. Preserve nulls, offsets, schema and mutation isolation, and keep backing storage alive through consumption. A later forced materialization can erase the saving.

**`same_dtype_concat_then_cast`: cast once after homogeneous assembly**
- Trigger: pieces with the same raw dtype are individually converted while being concatenated into a different output dtype, repeating conversion/allocation work during assembly.
- Technique: concatenate the homogeneous pieces in their raw dtype, then cast the combined buffer once. For example, assemble same-dtype integer arrays before the single required float conversion, rather than constructing a separate float array for every piece first. This separates assembly from conversion and can reduce per-piece overhead; it still writes a raw concatenated buffer and a converted output, so it does not inherently remove all copies.
- Guards: fall back for truly mixed types. Preserve output dtype, layout and conversion semantics, including invalid values and integer bounds. Account for the temporary concatenated buffer and peak memory rather than measuring conversion alone.
