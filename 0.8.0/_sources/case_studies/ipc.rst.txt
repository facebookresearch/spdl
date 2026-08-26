.. _ipc-cost:

The Cost of Inter-Process Communication
=======================================

.. py:currentmodule:: spdl.pipeline

Whenever data crosses a process boundary — a stage dispatched to a
:py:class:`~concurrent.futures.ProcessPoolExecutor`, a pipeline run with
:py:func:`run_pipeline_in_subprocess`, or the workers of a
:py:class:`torch.utils.data.DataLoader` — it is **copied** from one process to
another. That copy is not free, and for large payloads it is easy to spend more
time moving data between processes than doing useful work on it. This page
explains where that cost comes from, measures it, and describes ways to avoid it.
SPDL does not remove this cost automatically — it is something to be aware of and
to design around.

An example: the PyTorch DataLoader
----------------------------------

To make this concrete, let's take a conventional
:py:class:`~torch.utils.data.DataLoader` built around a large in-memory dataset.
Constructing TorchVision's ImageNet dataset object takes well under a second — it
scans the directory tree and builds a list of roughly 1.2 million
``(path, label)`` entries. But wrapping it in a ``DataLoader`` with a few workers
stalls the first iteration for **20+ seconds** before a single batch appears,
because the dataset object is **serialized and copied into every worker
process**. Building the list is cheap; shipping it across a process boundary,
once per worker, is not. The same copy is paid for any object sent across any
process boundary — the DataLoader just makes it easy to observe. In production,
dataloaders built without care for this have taken **more than 30 minutes** to
initialize, as the per-worker copy compounds with dataset size and worker count.

The :py:mod:`benchmark_ipc_dataloader` example isolates the effect: a dataset of
byte strings (standing in for an ``ImageFolder``'s paths) swept over total size
and worker count, timing the dataset **build** and the **startup** cost (iterator
creation through the first batch, when the workers are spawned and the dataset
shipped to each).

.. image:: ../_static/data/example_benchmark_ipc_dataloader.png
   :width: 100%

Building the dataset is cheap and roughly flat — no data crosses a boundary.
Startup cost grows with both payload size and worker count: each worker receives
its own serialized copy, so the transfer work scales with (payload size) ×
(worker count), reaching tens of seconds at the largest sizes and worker counts
even though ``__getitem__`` here does essentially nothing.

Why copying between processes is expensive
------------------------------------------

Two separate costs are hiding in that transfer.

**1. Serialization (pickling).** Processes do not share memory, so an arbitrary
Python object cannot simply be handed to another process — it must be converted
to a flat byte stream (:py:mod:`pickle`) on the sending side and reconstructed on
the receiving side. Both halves cost CPU, and the reconstruction on the receiving
side — allocating the objects again in the worker — is often the more expensive
one. A structure of many small objects, such as a large list of ``(path, label)``
entries, incurs this per object.

**2. Transport (the copy itself).** The resulting byte stream then has to travel
from one address space to another. By default this goes through a **pipe** (or a
socket-backed queue), and a pipe is a *stream*, not a shared region — so every
byte is physically copied twice: once from the sender's buffer into the kernel,
and once from the kernel into the receiver's buffer.

.. code-block:: text

     producer process              kernel                 worker process
    ┌─────────────────┐          ┌──────────┐            ┌─────────────────┐
    │ pickled dataset │  write() │   pipe   │   read()   │  rebuilt copy   │
    │ (~100 MB blob)  │ ───────► │  buffer  │ ─────────► │  (~100 MB blob) │
    └─────────────────┘  copy #1 └──────────┘   copy #2  └─────────────────┘
                        user→kernel            kernel→user

       the bytes never "move" — they are memcpy'd through the kernel, twice,
       and a whole second copy is rebuilt in the worker's address space

Chunking: a big payload through a small pipe
--------------------------------------------

The transport has a second cost, hidden below the API in how the operating
system actually moves the bytes. A pipe has a fixed capacity — on Linux the
default is **64 KiB** (16 pages of 4 KiB). It cannot hold a hundred-megabyte
payload, so the transfer cannot happen in one shot. Instead the payload is pushed
through the pipe **64 KiB at a time**: the writer fills the buffer, then *blocks*
until the reader has drained it, then fills it again. Producer and consumer
ping-pong like this until the whole payload is through.

.. code-block:: text

    pickled buffer (e.g. 100 MB)              pipe capacity = 64 KiB (16 × 4 KiB)
    ┌────────────────────────────┐
    │████████████████████████████│
    └─────────────┬──────────────┘
                  │ write() 64 KiB ───────────► [▓▓▓▓ pipe full ▓▓▓▓]
                  │                                     │ read() drains 64 KiB ─► worker
                  │ writer blocks until drained ◄───────┘
                  │ write() next 64 KiB ──────► [▓▓▓▓ pipe full ▓▓▓▓]
                  │                                     │ read() ─► worker
                  ▼                                     ▼
       ... repeat ~1,600 times for 100 MB (100 MB ÷ 64 KiB) ...

       every 64 KiB round-trip is a pair of read/write syscalls plus a
       block-and-wake — thousands of syscalls and context switches per payload

So moving one large payload through a pipe is not a single copy but *thousands*
of small ones, each with its own system-call and scheduler overhead, gated by the
64 KiB window. The two costs compound: pickling produces a large buffer, and the
pipe then drips that buffer across the boundary in tiny, blocking increments — on
both sides, once per worker.

.. _ipc-avoiding-the-cost:

Avoiding the cost
-----------------

There are three broad ways to pay less:

- **Don't cross the boundary at all.** SPDL's default is multi-threaded, not
  multi-process: threads share one address space, so passing data between stages
  is by reference — no pickling, no copy. When the heavy stages release the GIL,
  this is both the simplest and the fastest option. See :ref:`pipeline-parallelism`
  and :ref:`execution-models`.
- **Cross it fewer times.** When a process boundary is unavoidable (a GIL-bound
  stage, or isolating the loader from the training process), keep whole *regions*
  of the pipeline together in the workers rather than paying a round trip per
  stage, and hand only finished batches back. This is what the MP and MTP patterns
  do; see :ref:`execution-models` and the "cost of crossing a process boundary"
  discussion in :ref:`pipeline-parallelism`.
- **Make each crossing cheaper.** For the payloads you *do* send across, prefer
  types that move through shared memory instead of the pickle/pipe path.
  :py:class:`torch.Tensor` already does this: its multiprocessing reducer moves
  the tensor's storage through shared memory and pickles only a small handle, so
  a tensor transfers far more cheaply than its size suggests. Python's native
  ``bytes`` and NumPy arrays do **not** — they take the full pickle-and-copy path
  above — so a raw byte string is exactly the worst case, and the way to send one
  cheaply is to wrap it in a 1-D ``uint8`` tensor so it rides the tensor path. For
  the last bit of performance, :py:class:`SharedMemorySegmentPool` writes any
  payload into a pre-mapped shared-memory region so the bulk bytes never enter
  the pipe at all — see the :ref:`shared-memory arena case study
  <shared-memory-arena>`.

The common thread: a process boundary is a copy, and copies scale with what you
push across them. Keep the boundary count low and the per-crossing payload light,
and the IPC cost stops being the bottleneck.
