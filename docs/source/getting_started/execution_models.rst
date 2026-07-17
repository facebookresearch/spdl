.. _execution-models:

Execution Models: MT, MTP, and MP
=================================

.. currentmodule:: spdl.pipeline

The execution APIs from :ref:`pipeline-parallelism` -- thread and process
executors, the :py:meth:`~spdl.pipeline.PipelineBuilder.to` region API, and
:py:func:`~spdl.pipeline.run_pipeline_in_subprocess` -- are composable: the same
:py:class:`Pipeline` can run on a thread pool, across a pool of worker processes,
inside a subprocess, or a mixture of these. This page covers three compositions
we have tried and confirmed effective for keeping the GPU fed at high throughput:
MT, MTP and MP. Which one performs best depends on the project, so treat them as
starting points to benchmark rather than a fixed ranking.

All three are ways to run the *same* pipeline: the stages stay identical, and
only *where* each one executes changes. You choose that by how you build the
pipeline:

- **MT** (multi-threading) -- one process, a pool of worker threads.
- **MTP** (multi-threading in a subprocess) -- the whole pipeline runs in one
  subprocess and only finished batches cross back to the main process.
- **MP** (multi-processing) -- a pool of worker processes.

For each: the shape of the pattern, a minimal snippet, and when to reach for it.
The snippets use a continuous source
(:py:meth:`add_source(..., continuous=True) <spdl.pipeline.PipelineBuilder.add_source>`)
so the pipeline is reused across epochs instead of being rebuilt each one. See
:ref:`pipeline-parallelism` for the underlying mechanics and
:ref:`parallelism-performance` for a real-world tuning story.

Why the choice matters
----------------------

Much of a data loader runs Python, so the GIL (global interpreter lock) is in
play: while one thread holds it, no other thread in the same process can run
Python. A stage parallelizes across threads only to the extent its work
*releases* the GIL -- many I/O calls and array libraries (NumPy, PyTorch,
:py:mod:`spdl.io`) release it, but arbitrary Python does not. Whether a given
library releases the GIL is often not obvious up front, which makes the best
execution model an empirical question. The three models trade off differently:

- Threads share memory (no copying) but share one GIL.
- Processes each have their own interpreter and GIL (true parallelism) but must
  copy data across a process boundary.
- The subprocess model keeps thread-level sharing within the loader while
  isolating it from the process that runs the training loop.

For which libraries release the GIL, and what to do when a stage's function does
not, see :ref:`working-around-the-gil`.

There is a second consideration for GPU training: the **noisy-neighbour
effect**. The training loop drives the GPU from the main process, issuing the
next kernel as soon as the previous one finishes and the next batch is ready -- so
the GPU stays busy only if the CPU can respond to those moments promptly. If the
data loader is also burning CPU in that same process, it delays those launches and
the GPU is left waiting, which slows training down -- even when the loader's
throughput looks fine measured on its own. So for GPU training, *where* the
loader's CPU work runs matters as much as how fast it is: you want it off the
process that drives the GPU. This is what sets MTP (and an intermediate process in
MP) apart from plain MT, as noted below.

.. note::

   On free-threaded (no-GIL) builds of Python (3.13t / 3.14t) the GIL no longer
   serializes Python across threads, so the threaded stages in MT and MTP can
   run Python in parallel. How much that closes the gap to MP depends on the
   workload.

MT -- multi-threading
---------------------

The default. Each synchronous stage function is dispatched to a shared
:py:class:`~concurrent.futures.ThreadPoolExecutor` whose size you set with
``num_threads``. The front of the pipeline (source, async I/O, batching) runs on
the event loop; the CPU stages fan out to the thread pool and the results fan
back in.

.. code-block:: text

                         ┌────────┐
                         │ source │
                         └────┬───┘
                              │
              ┌───────────────┼───────────────┐             fan-out (async I/O, concurrent)
              ▼               ▼               ▼
         ┌─────────┐     ┌─────────┐     ┌─────────┐
         │download │     │download │ ... │download │
         └────┬────┘     └────┬────┘     └────┬────┘
              └───────────────┼───────────────┘             fan-in
                              ▼
                    ┌───────────────────┐
                    │ aggregate (batch) │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐             fan-out to N threads
              ▼               ▼               ▼
         ┌─────────┐     ┌─────────┐     ┌─────────┐
         │ thread 1│     │ thread 2│ ... │ thread N│
         │transform│     │transform│     │transform│
         └────┬────┘     └────┬────┘     └────┬────┘
              └───────────────┼───────────────┘             fan-in
                              ▼
                          ┌──────┐
                          │ sink │
                          └───┬──┘
                              ▼
                      ┌───────────────┐
                      │ training loop │
                      └───────────────┘

       one process, one GIL — threads parallelize only where the work releases it

.. code-block:: python

   from spdl.pipeline import PipelineBuilder

   pipeline = (
       PipelineBuilder()
       .add_source(source, continuous=True)
       .pipe(download, concurrency=...)   # async I/O, on the event loop
       .aggregate(batch_size)
       .pipe(transform, concurrency=N)    # sync CPU, on the thread pool
       .add_sink(buffer_size)
       .build(num_threads=N)              # size of the shared thread pool
   )

   for batch in pipeline:
       ...  # training loop

**Reach for MT when** the heavy stages release the GIL (media decode via
:py:mod:`spdl.io`, NumPy / PyTorch ops, network I/O) or when you run a
free-threaded build. It has the lowest overhead and passes data by reference.
See :ref:`pipeline-parallelism-custom-mt` for pinning a stage to its own
dedicated thread (for example, GPU transfer).

For GPU training, note the caveat above: because the loader threads share the
process with the training loop, pure MT keeps all of the loader's CPU work next
to the GPU-driving code and can trigger the noisy-neighbour effect. MTP addresses
this by moving the loader into a subprocess.

MTP -- multi-threading in a subprocess
--------------------------------------

MTP runs the *entire* loader -- source, I/O, transform, batching -- inside a
single subprocess with its own thread pool, and hands only finished batches back
to the main process through an IPC queue.
:py:func:`~spdl.pipeline.run_pipeline_in_subprocess` takes a pipeline config and
returns an iterable.

.. code-block:: text

     ┌───────────────────────────────────────────────────┐
     │  subprocess:  the full loader, same stages as MT  │
     └────────────────────────┬──────────────────────────┘
                              │
   ═════════════════════ process boundary ═════════════════════
                              │
                              ▼
     ┌───────────────────────────────────────────────────┐
     │  main process:  GPU transfer                      │
     └────────────────────────┬──────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────┐
     │  training loop                                    │
     └───────────────────────────────────────────────────┘
       only finished batches cross the boundary

.. code-block:: python

   from spdl.pipeline import PipelineBuilder, run_pipeline_in_subprocess

   builder = (
       PipelineBuilder()
       .add_source(source, continuous=True)
       .pipe(download, concurrency=...)
       .aggregate(batch_size)
       .pipe(transform, concurrency=N)
       .add_sink(buffer_size)
   )
   config = builder.get_config()

   # Run the whole pipeline in a subprocess; only batches cross back.
   src = run_pipeline_in_subprocess(config, num_threads=N)

   for batch in src:
       ...  # training loop, in the main process

Because the result is itself an iterable, you can build another pipeline on top
of it -- for example, to overlap GPU transfer in the main process while the
subprocess keeps loading:

.. code-block:: python

   src = run_pipeline_in_subprocess(config, num_threads=N)

   pipeline = (
       PipelineBuilder()
       .add_source(src, continuous=True)
       .pipe(gpu_transfer)
       .add_sink(...)
       .build(...)
   )

**Reach for MTP when** you are training on the GPU. Because the entire loader
runs in the subprocess, the main process spends its CPU on GPU kernel launches
rather than data loading -- directly avoiding the noisy-neighbour effect -- and
it has fewer Python objects to manage. This is why MTP is the recommended
production pattern; see :ref:`parallelism-performance`. A continuous source
matters most here: it keeps the subprocess pipeline warm across epochs and avoids
a per-epoch rebuild that would otherwise stall the GPU.

MP -- multi-processing
----------------------

When the bottleneck stage is pure-Python and holds the GIL, even MT and MTP
cannot run it across threads in parallel. Multi-processing sidesteps the GIL by
running a **region** of stages in a pool of worker processes, each with its own
interpreter. Mark the region with :py:meth:`~spdl.pipeline.PipelineBuilder.to`:
everything between ``.to(ProcessPoolExecutorConfig(...))`` and
``.to(MAIN_PROCESS)`` runs in the worker processes.

For GPU training, run the pipeline's orchestration -- source iteration, batching,
and dispatch to the workers -- in an **intermediate process** rather than the
main process, so the main process only issues GPU work. That is what the diagram
below shows, and it keeps the orchestration's CPU off the GPU-driving process
(the noisy-neighbour effect). Compose the region with
:py:func:`~spdl.pipeline.run_pipeline_in_subprocess` to get that intermediate
process:

.. code-block:: text

     ┌───────────────────────────────────────────────────┐
     │  intermediate process:  source → aggregate        │
     └────────────────────────┬──────────────────────────┘
                              │
   ═════════════════════ process boundary ═════════════════════
                              │
              ┌───────────────┼───────────────┐             fan-out to N worker processes
              ▼               ▼               ▼
         ┌─────────┐     ┌─────────┐     ┌─────────┐
         │ worker 1│     │ worker 2│ ... │ worker N│         separate interpreters,
         │  I/O +  │     │  I/O +  │     │  I/O +  │         no shared GIL
         │transform│     │transform│     │transform│
         └────┬────┘     └────┬────┘     └────┬────┘
              └───────────────┼───────────────┘             fan-in
                              │
   ═════════════════════ process boundary ═════════════════════
                              │
                              ▼
     ┌───────────────────────────────────────────────────┐
     │  main process:  training loop (GPU only)          │
     └───────────────────────────────────────────────────┘
       N worker processes · each its own interpreter & GIL · data crosses via IPC

.. code-block:: python

   from spdl.pipeline import PipelineBuilder, run_pipeline_in_subprocess
   from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

   builder = (
       PipelineBuilder()
       .add_source(source, continuous=True)
       .aggregate(batch_size)
       .to(ProcessPoolExecutorConfig(max_workers=N))  # heavy stages run in worker processes
       .pipe(download, concurrency=...)
       .pipe(transform, concurrency=...)
       .to(MAIN_PROCESS)
       .add_sink(buffer_size)
   )

   # Run the orchestration off the main process; the main process only
   # consumes batches and issues GPU work.
   src = run_pipeline_in_subprocess(builder.get_config(), num_threads=...)

   for batch in src:
       ...  # training loop (GPU) in the main process

(The region also works without the subprocess wrap -- calling ``.build()`` and
iterating it directly -- but then the main process runs the orchestration.)

The region's inputs and outputs cross a process boundary, so they must be
`picklable <https://docs.python.org/3/library/pickle.html#pickle-picklable>`_;
values passed between stages *inside* the region do not.
**Reach for MP when** a CPU-bound Python stage that does not release the GIL
dominates. The cost is IPC (data is copied across the boundary) and higher
memory (each worker is a full interpreter). See :ref:`pipeline-parallelism` for
the region and per-stage mechanics (including how a region composes with
``run_pipeline_in_subprocess``) and the picklability rules.

Choosing between them
---------------------

- Start with **MT** when the heavy stages release the GIL, or you run
  free-threaded Python -- it is the simplest option. MT is also the right call
  when data starvation is severe and the noisy-neighbour effect is not a concern:
  reach for MT first to raise throughput, then optimize the stage functions, and
  if that makes loading fast enough, MT alone may suffice.
- For GPU training, prefer **MTP** so the loader's CPU work stays off the process
  that drives the GPU (the noisy-neighbour effect).
- If a pure-Python, GIL-holding stage dominates, move it into an **MP** region so
  it runs across processes. For GPU training, run that region in an intermediate
  process (or inside MTP) so the main process still only issues GPU work.

Because a ``Pipeline`` can also be the source of another ``Pipeline``, these
models compose: an MP region inside an MTP subprocess, an MT front feeding a
dedicated GPU-transfer thread, and so on. When experimenting, this flexibility
makes it easy to switch between multi-threading, multi-processing, and mixtures
of the two.
