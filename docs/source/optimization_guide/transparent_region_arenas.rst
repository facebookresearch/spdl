.. _transparent-region-arenas:

Explicit caller-owned process-region arenas
===========================================

Reducing memory copies can improve pipeline performance. This can matter when
training small models: short compute steps leave less time to hide data movement,
so repeated allocations and process-boundary copies can take up a meaningful
share of each step. If profiling points to those costs, try a reusable
:doc:`shared-memory arena <shared_memory_arenas>` at the affected boundary and
measure the result.

Pass an arena to :py:meth:`~spdl.pipeline.PipelineBuilder.to` to change how data
crosses a process boundary without changing the stage itself. The stage still
receives and returns ordinary values. The transport copies supported binary
leaves into reusable shared memory and restores them on the receiving side.
If the stage itself needs to write directly into shared memory, use caller-managed
allocation tickets over ordinary IPC instead; transparent transport does not
recover or discard those tickets.

Start with one worker
---------------------

Create the arena in the caller and pass it on the marker that returns output to
``MAIN_PROCESS``. Size each segment for the payload plus overhead, and allow
enough segments for queued results and any results the consumer retains.

.. code-block:: python

   from spdl.pipeline import PipelineBuilder, SharedMemorySegmentPool
   from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

   arena = SharedMemorySegmentPool(8 * 1024 * 1024 + 4096, 2)
   pipeline = (
       PipelineBuilder()
       .add_source(batch_ids)
       .to(ProcessPoolExecutorConfig(max_workers=1, mp_context="spawn"))
       .pipe(make_and_collate_private_batch)
       .to(MAIN_PROCESS, arena=arena)
       .add_sink(2)
       .build(num_threads=3)
   )
   with pipeline.auto_stop():
       for batch in pipeline.get_iterator():
           consume(batch)

Keep the ``arena`` reference and follow the retirement steps below before
closing or unlinking it. Exiting ``auto_stop()`` is not itself proof of worker
retirement.

Scale to more workers and boundaries
------------------------------------

For ``max_workers=N``, pass a list or tuple of N independently allocated arenas
instead of a single arena. For example, at the output marker for two workers::

   arenas = [SharedMemorySegmentPool(8 * 1024 * 1024 + 4096, 2) for _ in range(2)]
   builder.to(MAIN_PROCESS, arena=arenas)

The marker snapshots the sequence and checks its length, required methods,
and duplicate objects or known allocation aliases before starting workers.
Different handles are not a substitute for independent backing storage;
custom backends must provide their own single-producer/single-consumer
synchronization. See :py:class:`spdl.pipeline.ArenaProtocol` for the endpoint
contract.

An arena on the opening process marker carries inputs; an arena on the closing
``MAIN_PROCESS`` marker carries outputs. Use independent arenas for each
worker and direction. For adjacent process regions, close the first into
``MAIN_PROCESS`` and open the next with separate input arenas rather than
putting an arena on the marker between them. Buffers are separate for each
direction and flush partial transfers at epoch boundaries or end of stream.
Use ``arena=None`` for ordinary IPC. Arenas apply only at process-region
boundaries, not in main-only or subinterpreter regions.

Own and retire the arenas
-------------------------

The pipeline borrows arenas; the caller owns ``close()`` and ``unlink()``,
including after validation or startup fails. A fully delivered channel whose
workers have been confirmed reaped can be reused before caller cleanup.
For final cleanup, release unneeded results and verify that all the reported
workers have been reaped. Then close the original owning handles and unlink
only after retirement completes. Live tensor, array, or buffer views can delay
reuse or retirement and must remain valid after the pipeline stops. If cleanup
is incomplete or worker liveness is unknown, keep the owning references and
retry only once it is safe.

Cancellation can leave unread values or a publication that never reached the
queue. Do not try to discard such a publication out of order or attach a fresh
writer to that channel. Retire it with caller-owned cleanup and use new storage.
This differs from a fully delivered channel whose live views merely delay reuse.
If a custom backend can block indefinitely, the caller must arrange for its
public cancellation or terminal close to wake the blocked operation before
joining workers, while preserving live views. Do not unlink while worker
liveness is uncertain; a stop timeout does not guarantee their exit. See
:doc:`../notes/shared_memory_ownership` for more on storage lifetimes.

Pin memory at the receiver
--------------------------

The receiver controls pinning. An output reader runs in the main process; an
input reader runs in a worker. For GPU transfer in the main process, consider
pinning output storage. Do not assume that enabling pinning on inputs leaves
CPU producers CUDA-uninitialized.

Measure your application
------------------------

The :ref:`region arena benchmark results <region-arena-benchmark-results>`
show lower CPU time per batch with a pool in the measured workloads, with
higher memory use at larger batch sizes. Compare both throughput and memory
under your own workload before choosing a transport.
