Shared-memory arenas
====================

Moving a large batch between CPU processes can cost more than putting an item on
a queue: payloads must be serialized, transport storage allocated, and bytes
copied or attached on the receiving side. Some framework transfers already share
payload storage, but creating fresh storage for every object still costs work.

A **shared-memory arena** is preallocated host storage that a producer and
consumer can both access and reuse across process-boundary transfers.
:py:class:`~spdl.pipeline.SharedMemorySegmentPool` and
:py:class:`~spdl.pipeline.SharedMemoryRingBuffer` reuse preallocated host memory
across many transfers. They reduce repeated transport allocations; a pool also
avoids receiving-side copies for supported views. This can reduce CPU data-transfer
work, but requires an explicit budget and storage-lifetime contract.

The producer writes supported binary fields into that shared storage. The queue
carries a small description of where the bytes are and how to restore them,
rather than another copy of every payload. The receiver can then read shared
views or copy into its own buffer, depending on the backend. Small values and
control information can still use ordinary transport; an arena does not make
every part of a nested value copy-free.

A **unit** is one yielded value together with its offloaded binary fields. A unit
can contain several tensors or arrays, so size an arena for all of those fields,
not just the largest leaf. Reusing storage saves allocation/copying work, but
also requires a rule for when the next value may overwrite that storage.

Pool views or ring copies
-------------------------

SPDL offers two existing arena backends through
:py:func:`spdl.pipeline.iterate_in_subprocess`:

- :py:class:`~spdl.pipeline.SharedMemorySegmentPool` rotates through a fixed set
  of equal-sized segments, one per unit. Supported restored tensors, arrays and
  views refer directly to the shared segment, avoiding a receiving-side copy.
  Retaining a view or storage alias keeps that segment occupied. In-order reuse
  cannot skip an earlier live unit just because a later one has been released.
- :py:class:`~spdl.pipeline.SharedMemoryRingBuffer` is a bounded first-in-first-out
  (**FIFO**) byte region. The receiver copies each unit into independent owning
  memory, then releases that unit's ring span. The caller can retain the result
  without retaining the source ring, at the cost of this receiving-side copy.

In their default ordinary CPU mode, pool views use shared **pageable** memory
and ring outputs use owning pageable buffers. Pageable means the operating
system may page memory out; it does not require CUDA. Both backends still perform
producer-to-arena writes. The pool avoids copy-out for supported views; the ring
keeps it deliberately so retained outputs do not stall its source FIFO.

When to use an arena and how to size it
--------------------------------------------------

First measure whether process-boundary allocation or copying is an important
cost in the application. Arenas are worth comparing for large, repeated payloads
with a predictable in-flight budget. Small messages, expensive worker computation
or long-lived outputs can change the tradeoff; a faster transfer mechanism is
not automatically a faster training job.

Choose a pool when the consumer can release shared views promptly and avoiding
copy-out matters. Choose a ring when independent owning outputs are more useful,
for example when a consumer keeps many results. A fixed ring capacity does not
bound those retained output allocations: they live outside the ring.

A pool's payload-capacity budget is ``count * segment_size``. Each segment must
fit one unit's offloaded fields, with alignment/control overhead allowed. Count
must cover queued units **and** aliases held by the consumer. A ring must fit its
largest unit and in-flight FIFO bytes. Budget across all workers/ranks and leave
host, shared-memory/tmpfs and cgroup headroom; logical reservation is not the
same thing as resident or total process memory.

CPU requirements and a basic example
--------------------------------------------------

Ordinary ``spdl.pipeline`` import/execution remains Python-only. Native-backed
pool views, **including pageable ones**, require optional SPDL IO support via
``spdl.io.lib._buffer``; install SPDL IO when using that storage path. The helper
is loaded only when storage is actually exported. Unpinned ring copy-out does not
need it. This is a storage-lifetime requirement, not a GPU requirement: CPU-only
pool producers and consumers do not need to initialize CUDA. The IO parent may
load its existing NumPy Python dependency; the native buffer helper itself has
no media, tensor or CUDA native dependency and has no unsafe finalizer fallback.

Create an arena in the parent, then give it to a subprocess iterable boundary.
This example uses four reusable segments for 4-MiB FP32 batches. It keeps the
ordinary pageable default and uses no GPU-transfer helper::

    from spdl.pipeline import SharedMemorySegmentPool, iterate_in_subprocess

    def make_batches():
        import torch
        # The worker creates this re-iterable collection once.
        return [torch.full((1024 * 1024,), i, dtype=torch.float32)
                for i in range(8)]

    def run_cpu():
        arena = SharedMemorySegmentPool(segment_size=4 * 1024 * 1024, count=4)
        batches = iterate_in_subprocess(
            make_batches, arena=arena, buffer_size=2, mp_context="spawn"
        )
        for cpu_batch in batches:
            print(cpu_batch.shape, int(cpu_batch[0]))
            del cpu_batch  # Do not hold this shared unit unnecessarily.
        # The returned iterable owns arena teardown; do not reuse arena elsewhere.

    if __name__ == "__main__":
        run_cpu()

Use a top-level picklable factory and the usual ``__main__`` guard with spawn.
The returned iterable creates the worker-side iterable once and reuses its worker
for subsequent iterations. A returned list can be iterated again; an exhausted
generator cannot. This example uses one arena for one producer/boundary. Ownership
transfers to the returned iterable, which closes/unlinks it at teardown; do not
hand that same arena to another boundary or reuse it afterwards.

For a FIFO with independent outputs, replace only the arena construction::

    from spdl.pipeline import SharedMemoryRingBuffer
    arena = SharedMemoryRingBuffer(capacity=16 * 1024 * 1024)

Storage lifetime is not wrapper lifetime
--------------------------------------------------

A pool-backed tensor can outlive the Python wrapper that first returned it:
``detach()``, retained storage, NumPy bases or memoryview slices may still refer
to its bytes. Early reuse or unmapping would corrupt those values. Ring outputs
own their copies, but retaining many of them retains many independent buffers.
Closing local use and unlinking a shared name are different from invalidating
live storage. See :doc:`../notes/shared_memory_ownership` in Development Notes
for the technical storage-owner, alias, cleanup and external-consumer rules.

Memory pinning
--------------

After choosing an arena for CPU transport, a GPU consumer may have another cost:
copying pageable output into pinned staging memory before transferring it to
CUDA. **Pinned** (page-locked) host memory stays resident while registered or
allocated as pinned. CUDA can use it as a host-to-device (**H2D**, CPU-to-GPU)
transfer source without first making another pinned CPU staging copy.

The existing arena constructors have an experimental ``pin_memory`` argument,
which defaults to ``False``. Set it to ``True`` only when the receiving workflow
benefits from GPU-transfer-ready host storage:

.. versionadded:: 0.9.0
   The ``pin_memory`` argument on the existing pool and ring buffer; the arena
   classes and ordinary CPU usage are not new in this version.

The two backends prepare receiving storage differently:

- A pinned pool registers each complete data mapping **once in the consumer**
  with CUDA and reuses that registration across later units. Registration
  page-locks existing shared memory rather than allocating another per-batch
  host buffer. CPU producer attachments do not register memory or initialize CUDA.
- A pinned ring still copies each unit out. That required copy goes directly
  into a Torch-allocator pinned owning buffer instead of a pageable bytearray.
  The source ring itself is not CUDA-registered, and retaining an output still
  does not retain its old ring span.

This avoids a redundant pinning/staging copy for supported already-pinned
outputs, not every CPU copy in the pipeline. The public
:py:func:`spdl.io.transfer_tensor_h2d` calls ``pin_memory()`` on its inputs;
for a supported already-pinned tensor it aliases the same storage. The helper
completes its transfer before returning. The :py:mod:`benchmark_pinned_arena`
example compares these paths with pageable controls in a fully checked workflow;
its results are not a promise of higher model-training throughput.

A GPU consumer variant
----------------------

Reuse ``make_batches`` from the CPU example, but select the pinned pool and this
consumer in a CUDA-capable environment::

    from spdl.io import transfer_tensor_h2d

    def run_gpu():
        arena = SharedMemorySegmentPool(
            segment_size=4 * 1024 * 1024, count=4, pin_memory=True
        )
        batches = iterate_in_subprocess(
            make_batches, arena=arena, buffer_size=2, mp_context="spawn"
        )
        for cpu_batch in batches:
            gpu_batch = transfer_tensor_h2d(cpu_batch, num_caches=0)
            # Pass gpu_batch to training on the selected device's default stream.
            print(gpu_batch.shape, gpu_batch.device)
            del cpu_batch, gpu_batch

    if __name__ == "__main__":
        run_gpu()

For an owning pinned ring output, use::

    arena = SharedMemoryRingBuffer(capacity=16 * 1024 * 1024, pin_memory=True)

Pinning initializes CUDA lazily in the actual consumer, using the device selected
by ``LOCAL_RANK``. Arena construction and CPU producer attachment do not initialize
CUDA. Use spawn: initializing CUDA after an unsafe fork is rejected. Only shared
names and allocation policy cross this boundary, not pointers, registrations
or CUDA objects. Native-backed pool views and pinned ring allocations require
the optional IO helper; unpinned ring copy-out remains helper-free.

Pinning costs and transfer safety
--------------------------------------------------

Keep ``pin_memory=False`` when data stays on CPU, setup is not repaid by the
workload, later transforms allocate pageable outputs anyway, or host memory is
scarce. Page-locking consumes resident memory the operating system cannot freely
reclaim. Try both modes in the same application instead of inferring a gain from
allocation policy alone.

Pool registration covers whole consumer mappings, not just the current item.
Previously sparse shared pages can become resident, so reserve host/shared-memory
capacity for the whole pool and allow control/alignment overhead. Pinned ring
outputs are separate allocations; many retained outputs can use more pinned
memory than the source ring's capacity. Host/tmpfs/cgroup checks catch some
impossible requests but do not replace budgeting across all workers and ranks.

A CUDA copy can still use host bytes after the initiating code has returned.
The synchronized H2D helper retains sources, destinations and its transfer stream
even if a later nested field raises. If draining the stream fails, those owners
stay quarantined until a successful retry proves completion. A pool must be
unregistered **before** unmapping; live aliases and H2D holds postpone cleanup.
Pinning/unregister errors are reported, not silently treated as pageable success;
repeated close can retry unresolved unregister. The technical details belong in
:doc:`../notes/shared_memory_ownership`.

Returned CUDA tensors are recorded on the selected device's default consumer
stream. Work on another stream still needs normal PyTorch readiness and
``record_stream`` handling. Raw nonblocking H2D outside the helper must retain
source owners until completion; external shared-memory registration alone is not
allocator event tracking.

The pinned-backing promise applies to supported restored views, not every field.
Inline leaves below the offload threshold, reconstructed owning bytes/bytearrays,
Packets and later transforms can be pageable. Empty fields have no transfer bytes;
pinning does not add otherwise unsupported dtypes such as bfloat16. Inspect actual
outputs and use the example's checks rather than assuming an entire nested batch
is pinned.
