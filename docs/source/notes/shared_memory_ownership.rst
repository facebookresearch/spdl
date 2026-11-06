Why shared arena outputs keep storage alive
=============================================

Start with :doc:`../optimization_guide/shared_memory_arenas` for what arenas save,
pool/ring choices, default CPU usage and the optional memory-pinning workflow.
This Development Note focuses on the technical obligation behind those choices:
**do not overwrite or unmap bytes that an output or in-flight transfer still
uses**. Storage lifetime is not just the lifetime of the Python wrapper that
first returned it; the same rule matters with or without pinning.

Pool views and ring copies have different lifetimes
-----------------------------------------------------

An output restored from a
:py:class:`~spdl.pipeline.SharedMemorySegmentPool` refers to the shared unit's
storage. **Reclaiming** means permitting a producer to reuse those bytes for a
later unit; it must wait while any consumer alias still uses them.

Consider a consumer keeping a view of an earlier batch::

    cpu_batch = next(iterator)       # A restored pool-backed Torch tensor.
    retained = cpu_batch.detach()    # Another Tensor using the same storage.
    del cpu_batch                   # Only the original Python wrapper is gone.
    # retained still reads the original values; this segment cannot be reused.
    del retained                    # Reuse is possible once ALL storage aliases go.

A weakref/finalizer on ``cpu_batch`` would be the wrong signal: it could fire
while ``retained`` still reads those bytes. Reshapes, ``untyped_storage()``,
``set_`` aliases, NumPy base chains and memoryview slices can also outlive the
original wrapper. The pool therefore waits for every export of the storage,
including exports that escape a failed restore operation.

Suppose unit 0 is still held by a consumer while unit 1 has already been released.
When the writer wraps around, it cannot skip unit 0 and treat unit 1's release as
permission to overwrite older storage. The reclaim point advances in order only
through consecutive released units. This in-order rule trades some reuse flexibility
for the guarantee that an earlier live view still sees its original values.

A :py:class:`~spdl.pipeline.SharedMemoryRingBuffer` instead uses a shared
first-in-first-out (FIFO) byte region. The receiver copies a unit into a new owning buffer before returning its
ring span for immediate reuse. An output retained after that point owns the
copy—not a live view of the source ring. Pinning changes that destination to a
pinned allocator-owned buffer; it does not remove ring copy-out or make retained
outputs consume the old ring span.

How the storage owner follows aliases
---------------------------------------

Native-backed arena views use the optional ``spdl.io.lib._buffer`` helper. The
`native buffer exporter implementation
<https://github.com/facebookresearch/spdl/blob/main/src/spdl/io/lib/buffer/register.cpp>`_
is the C++ ``Exporter`` introduced for these storage paths. Its immutable object
implements the buffer protocol. For shared
mappings it holds an actual ``Py_buffer`` export; for allocator-owned pinned
outputs it holds a hidden reference to the allocation owner. A native storage
consumer that retains this exporter keeps the underlying memory alive, even
when the original Tensor or view wrapper disappears.

The built-in Torch restore path retains the exporter itself for complete
contiguous binaries. Independent wrappers preserve offsets for slices. Derived
NumPy and memoryview aliases retain their backing export too. Once the last
exporter is destroyed, it releases the backing export/owner **before** notifying
caller-side lease bookkeeping that storage can be reused or unmapped.

Each buffer export has its own native release ticket. Setup owns rollback until
the native factory adopts that ticket before returning a successful view. After
adoption, a Python return error is not permission to release the hold: the error's
traceback or an escaped alias may still own the actual exported storage. Only
exporter destruction can retire the native-owned ticket, exactly once.

The exporter deliberately roots its callback and allocator graph until actual
storage destruction. Otherwise cyclic GC could clear a Python partial or bound
owner before the native destructor calls it. This is conservative: genuine
callback ownership cycles can remain retained and may need to be broken by the
caller. It is not a blanket promise of cycle reclamation or a Python-finalizer
fallback.

This is why the protection is more than a Python wrapper finalizer. Supported
Python 3.11 has no Python-defined buffer-export hooks; Python 3.12+ provides
PEP 688 hooks, but a replacement would still need a validated storage-owner
design.

The helper is loaded only at actual storage export/allocation. Ordinary pipeline
use has no mandatory IO/native dependency, and unpinned ring copy-out needs no
helper. Native pool views, including pageable ones, and pinned ring destinations
require SPDL IO. Its existing NumPy Python dependency may load on that path, but
creating the buffer exporter does not load Torch, CUDA or media-native
extensions. There is no unsafe finalizer fallback if the helper is unavailable.

Closing local use is not unlinking a name
-------------------------------------------

**Close** stops new local arena use; **unlink** removes the shared-memory name so
that new processes cannot attach by that name. Removing a name is different from
removing an already-mapped region. Existing exported views must stay readable
until their storage owners are released.

For a pool, live exports postpone reuse and local unmapping even after close and
unlink. The final export release can finish cleanup without reopening an
already-closed control segment. A pinned pool must also unregister each original
mapping before unmapping it. If unregister or transfer completion is uncertain,
the owner is retained for retry rather than risking memory reuse while CUDA
might still use it.

A ring output already owns its copy, so closing/unlinking the source ring does
not invalidate that output. Retaining many such outputs can nevertheless retain
many independent allocations. A fixed source-ring capacity does not bound all
consumer-held output memory.

When an arena is handed to ``iterate_in_subprocess``, its ownership transfers to
the returned iterable, which closes/unlinks it at teardown. Do not share that
same arena with another boundary or reuse it after teardown. Callers using the
low-level endpoints manage close/unlink themselves and must keep the storage
owners, not merely raw addresses.

CUDA work has its own completion rule
---------------------------------------

Keeping an exported CPU buffer alive is not proof that a CUDA copy has finished.
The native exporter does not track DMA, CUDA streams or cross-process slot
accounting. The synchronized :py:func:`spdl.io.transfer_tensor_h2d` retains the
source, destination and transfer stream until completion, including failure
paths. With raw nonblocking H2D, the caller must retain source owners until the
copy completes; allocator bookkeeping alone does not protect an externally
registered shared mapping. See :doc:`../optimization_guide/shared_memory_arenas` for stream and
failed-cleanup behavior.

External ``torch.frombuffer`` consumers
------------------------------------------

SPDL's built-in Torch restore path retains the immutable exporter directly.
External ``torch.frombuffer(memoryview)`` instead retains the supplied Python
memoryview object; it does not protect that wrapper against explicit
``release()`` or context-manager exit. Do not release a wrapper handed to an
external consumer while its native aliases still use it.

For a complete arena binary, retaining the exporter directly avoids that
particular wrapper-release hazard::

    tensor = torch.frombuffer(arena_view.obj, dtype=torch.uint8)
    arena_view.release()  # tensor's native storage still owns the exporter.

This is only for a **complete** binary: ``arena_view.obj`` describes the entire
export, not a sliced view's offset or shape. For slices/casts, keep the correct
layout and supply an independent ``memoryview(sliced_view)`` that is not
explicitly released while Torch uses it. Ordinary garbage collection of the
original wrapper is safe when a valid owning export remains. An arbitrary raw
pointer is not such an ownership reference.
