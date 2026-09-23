Migrating from Grain
====================

`Grain <https://github.com/google/grain>`_ and SPDL can both construct concurrent
input pipelines, but they expose different abstractions. Grain is dataset-first:
it preserves lazy random access through :py:class:`grain.MapDataset` and later
turns the dataset into an iterator. SPDL is stage-first: it connects a source,
independently scheduled operations, bounded queues, and a sink.

Migration is therefore a pipeline redesign, not an import replacement. SPDL is
a strong fit when the workload benefits from stage-specific concurrency or
async I/O. Its task and queue statistics provide the structured feedback needed
for manual tuning and agentic performance optimization. Features without a
built-in equivalent require an application-level adapter during migration.

Why SPDL uses a stage-first pipeline
------------------------------------

A production input pipeline often combines remote network reads, CPU decoding
or preprocessing, host-to-device transfer, accelerator computation, and
device-to-host transfer. These operations are limited by different resources,
so one concurrency setting is rarely appropriate for the entire pipeline.

For example, a remote read spends much of its lifetime waiting rather than
using a CPU and can often benefit from substantially more concurrency than a
CPU-bound transform. CPU work should keep the available cores busy, but
unbounded workers can delay the training loop through CPU contention and the
:doc:`noisy-neighbour effect <../optimization_guide/noisy_neighbour>`. Device
transfers have a different constraint again: accelerator copy engines commonly
allow one host-to-device and one device-to-host transfer to overlap with
computation, so transfer stages are usually kept at low concurrency and
scheduled independently in each direction.

SPDL represents these resource boundaries as pipeline stages. Each stage can
select its own execution model, concurrency, output ordering, and bounded queue,
which makes the scheduling policy correspond directly to the production data
path. A dataset abstraction is familiar and convenient for expressing samples
and transformations, but packaging heterogeneous work behind one dataset
iterator makes independent concurrency, buffering, and backpressure harder to
observe and tune.

SPDL also records per-stage task time, throughput, queue wait time, and queue
occupancy. These measurements expose the current bottleneck and the effect of a
configuration change. Besides supporting manual tuning, the structured feedback
can drive an agentic optimization loop that measures the pipeline, changes one
stage's concurrency or buffering, and validates the end-to-end result. See the
:doc:`SPDL optimization guide <../optimization_guide/index>` and its
:doc:`performance-statistics reference <../optimization_guide/stats>`.

How Grain is commonly used
--------------------------

Grain has two APIs:

* :py:class:`grain.DataLoader` combines a random-access data source, sampler,
  flat operation list, worker pool, and output buffer.
* The Dataset API chains :py:class:`grain.MapDataset` and
  :py:class:`grain.IterDataset` transformations. It preserves random access as
  long as possible, then adds threaded reads, prefetching, or multiprocessing
  when converted to an iterable dataset.

In large training codebases, the Dataset API is the more common integration
shape. Representative pipelines use it to:

* wrap custom file, record, Hive, or in-memory random-access sources;
* seed, globally shuffle, repeat, and shard a dataset;
* map tokenization, decoding, augmentation, and collation functions;
* filter invalid examples and batch or pack variable-length examples;
* mix multiple datasets with weights;
* prefetch with :py:class:`grain.ReadOptions` and run transforms with
  :py:class:`grain.MultiprocessingOptions`; and
* save and restore :py:class:`grain.DatasetIterator` or elastic-iterator state
  with a training checkpoint.

Grain does have multithreading: ``ReadOptions(num_threads=N)`` concurrently
reads and applies ``MapDataset`` transformations with a
``concurrent.futures.ThreadPoolExecutor`` while returning elements in input
order. Without ``mp_prefetch``, that pool runs in the main process and no worker
process is created. When ``mp_prefetch`` wraps the iterator, ``ReadOptions``
instead configures each worker process. Grain's Dataset API does not expose an
equivalent completion-order switch.

Equivalent pipeline structure
-----------------------------

The following single-rank example applies the same source lookup, transform,
ordering, and fixed-size batching contract in each library. A typical Grain
pipeline separates the random-access graph from iteration:

.. code-block:: python

   import grain.python as grain

   dataset = (
       grain.MapDataset.source(source)
       .seed(seed)
       .shuffle()
       .map(transform)
       .to_iter_dataset(grain.ReadOptions(num_threads=8))
       .batch(batch_size, drop_remainder=True)
   )

   for batch in dataset:
       train_step(batch)

The corresponding SPDL pipeline makes the execution stages explicit:

.. code-block:: python

   from spdl.pipeline import PipelineBuilder
   from spdl.source import DistributedRandomSampler
   from spdl.source.utils import embed_shuffle

   sampler = embed_shuffle(
       DistributedRandomSampler(
           len(source), rank=0, world_size=1, seed=seed
       )
   )

   pipeline = (
       PipelineBuilder()
       .add_source(sampler)
       .pipe(source, concurrency=8, output_order="input")
       .pipe(transform, concurrency=4, output_order="input")
       .aggregate(batch_size, drop_last=True)
       .pipe(collate, concurrency=1)
       .add_sink(4)
       .build(num_threads=12)
   )

   for batch in pipeline:
       train_step(batch)

Both versions emit fixed-size batches without reordering after sampling. A
distributed migration must additionally match the application's exact sharding
and shuffle contract. The SPDL form makes lookup and transform scheduling
independent, so each stage can be tuned for its own resource constraint.
Collation is serial here because the example assumes it is inexpensive; give
it separate measured concurrency when it performs substantial CPU work.

Component mapping
-----------------

.. list-table:: Grain to SPDL mapping
   :header-rows: 1
   :widths: 28 30 42

   * - Grain
     - SPDL
     - Migration note
   * - ``RandomAccessDataSource``
     - An index iterable plus ``pipe(source)``
     - Yield records directly if random access is unnecessary. Yield indices and
       pass the random-access source to a pipe when lookup needs concurrency.
   * - ``IndexSampler`` or ``ShardOptions``
     - :py:class:`spdl.source.DistributedRandomSampler` or
       :py:class:`spdl.source.DistributedDeterministicSampler`
     - Match rank, world size, draw count, seed, and drop behavior before tuning.
   * - ``MapDataset`` or ``IterDataset``
     - :py:class:`spdl.pipeline.Pipeline`
     - SPDL is streaming and does not preserve a random-access dataset view.
   * - ``map`` or ``MapTransform``
     - :py:meth:`spdl.pipeline.PipelineBuilder.pipe`
     - Give I/O, decode, tokenization, collation, and transfer separate stages.
   * - ``random_map`` or ``RandomMapTransform``
     - A source or ``pipe`` stage that carries an explicit per-record RNG
     - Derive the RNG from the global seed, epoch, rank, and stable sample
       identity instead of sharing mutable global state.
   * - ``filter`` or ``FilterTransform``
     - A ``pipe`` operation that returns ``None`` for rejected items
     - Measure rejection rates; filtering changes downstream batch boundaries.
   * - ``flat_map``
     - A generator operation, or ``pipe`` followed by
       :py:meth:`spdl.pipeline.PipelineBuilder.disaggregate`
     - Preserve ordering explicitly if downstream logic depends on it.
   * - ``batch`` or ``Batch``
     - :py:meth:`spdl.pipeline.PipelineBuilder.aggregate`, followed by a collate
       ``pipe``
     - Match ``drop_remainder`` with ``drop_last``. Use a custom
       :py:class:`spdl.pipeline.defs.Aggregator` for size-aware packing.
   * - ``repeat``
     - ``add_source(..., continuous=True)`` or
       :py:func:`spdl.source.utils.repeat_source`
     - Decide where epoch boundaries and reshuffling occur.
   * - ``shuffle``
     - A random sampler, optionally wrapped by
       :py:func:`spdl.source.utils.embed_shuffle`
     - This covers index shuffling, not arbitrary streaming-buffer shuffle.
   * - ``ReadOptions`` and prefetch
     - Per-stage ``concurrency``, ``add_sink(buffer_size)``, and
       ``build(num_threads)``
     - SPDL makes scheduling and buffer choices explicit for each stage.
   * - ``MultiprocessingOptions``
     - ``ProcessPoolExecutor`` for one stage, an SPDL process execution region,
       or a subprocess pipeline
     - Put only GIL-holding work across a process boundary; serialization can
       erase the gain.
   * - ``MapDataset.mix``
     - Application composition of iterables or maps
     - :py:class:`spdl.source.utils.MergeIterator` is one weighted-iterable
       implementation. Match stopping, exhaustion, repetition, seeding, and
       checkpoint behavior.
   * - Dataset statistics and profiling
     - Pipeline task statistics, queue statistics, custom
       :py:class:`spdl.pipeline.TaskHook` objects, and
       :py:func:`spdl.pipeline.profile_pipeline`
     - Stage boundaries expose bottlenecks and backpressure. Task hooks can add
       per-sample pre/post-processing and measurements.

Migration work for non-equivalent features
------------------------------------------

As of SPDL 0.7.0, implement and validate the following behaviors in application
code. These sketches show the relevant builder fragment and omit the common
``add_sink(...).build(...)`` suffix, application-specific failure handling, and
checkpoint storage.

* **Checkpointable iterator state and elastic resume.** SPDL does not provide a
  general ``DatasetIterator.get_state``/``set_state`` equivalent. Persist the
  source epoch and committed source offset, then reconstruct the sampler on
  resume. Track consumed source positions directly because filtering can break
  the relationship between batch count and source position:

  .. code-block:: python

     from itertools import islice

     state = {"epoch": epoch, "source_offset": committed_source_offset}
     save_with_training_checkpoint(state)

     def make_sampler(epoch):
         return DistributedRandomSampler(
             len(source),
             rank=rank,
             world_size=world_size,
             seed=base_seed + epoch,
         )

     class ResumedEpoch:
         def __iter__(self):
             sampler = make_sampler(epoch=state["epoch"])
             return islice(iter(sampler), state["source_offset"], None)

     builder = PipelineBuilder().add_source(ResumedEpoch())

  Rebuild the source with offset zero when advancing to the next epoch. The
  epoch-to-seed mapping is part of the checkpoint contract and must be identical
  before and after resume.

* **Lazy random access after transformations.** A running SPDL pipeline is an
  iterator, not an indexable dataset. Keep a separate transformed view when
  indexed debugging or evaluation requires one:

  .. code-block:: python

     class TransformedView:
         def __len__(self):
             return len(source)

         def __getitem__(self, index):
             return transform(source[index])

     debug_view = TransformedView()
     builder = (
         PipelineBuilder()
         .add_source(range(len(debug_view)))
         .pipe(debug_view.__getitem__, concurrency=8, output_order="input")
     )

* **Random-access and checkpointable mixture state.** Compose application
  iterables or maps according to the required semantics. ``MergeIterator`` is
  one ready-made weighted iterable, but it does not preserve Grain
  ``MapDataset.mix`` random access or expose its RNG and parent iterator states:

  .. code-block:: python

     from itertools import chain
     from spdl.source.utils import MergeIterator

     class Concatenated:
         def __iter__(self):
             return chain(source_a, source_b)

     concatenated_builder = PipelineBuilder().add_source(Concatenated())

     weighted = MergeIterator(
         [source_a, source_b],
         weights=[0.8, 0.2],
         stop_after=epoch_size,
         seed=epoch_seed,
     )
     weighted_builder = PipelineBuilder().add_source(weighted)

* **Automatic per-record random keys.** Inject an RNG in the source or a pipe
  stage, derived from stable record metadata. A shared mutable global RNG is
  unsafe when a stage has concurrency greater than one:

  .. code-block:: python

     import hashlib
     import random

     def load_with_rng(index):
         identity = f"{global_seed}:{epoch}:{rank}:{index}".encode()
         seed = int.from_bytes(
             hashlib.blake2b(identity, digest_size=8).digest(), byteorder="big"
         )
         return source[index], random.Random(seed)

     def random_transform(item):
         sample, rng = item
         return augment(sample, rng)

     builder = (
         PipelineBuilder()
         .add_source(sampler)
         .pipe(load_with_rng, concurrency=16)
         .pipe(random_transform, concurrency=8)
     )

* **Drop-in elastic global batching.** Global batch size, worker topology,
  resharding, and resume behavior must be expressed by the application:

  .. code-block:: python

     if global_batch_size % world_size:
         raise ValueError("global batch size must be divisible by world size")
     rank_batch_size = global_batch_size // world_size
     sampler = DistributedRandomSampler(
         len(source),
         rank=rank,
         world_size=world_size,
         ddp_drop_last_distributed_round=True,
     )
     builder = (
         PipelineBuilder()
         .add_source(sampler)
         .pipe(source, concurrency=8)
         .aggregate(rank_batch_size, drop_last=True)
     )

  Keeping distributed-round dropping enabled gives every rank the same source
  length. ``drop_last=True`` then removes an incomplete per-rank batch, so all
  ranks emit the same number of batches. Change either policy only when the
  application explicitly synchronizes uneven ranks.

* **Dataset transformation introspection.** SPDL reports runtime stage behavior,
  and a custom ``TaskHook`` can capture per-sample inputs, timing, or failure
  metadata. It does not reproduce Grain's index-level dataset debugging model:

  .. code-block:: python

     from contextlib import asynccontextmanager
     from time import perf_counter

     from spdl.pipeline import TaskHook, TaskStatsHook

     class SampleTiming(TaskHook):
         @asynccontextmanager
         async def task_hook(self, input_item=None):
             started = perf_counter()
             try:
                 yield
             finally:
                 record_latency(input_item, perf_counter() - started)

     def hook_factory(stage):
         hooks = [TaskStatsHook(stage)]
         if stage.stage_name == "decode":
             hooks.append(SampleTiming())
         return hooks

     pipeline = (
         PipelineBuilder()
         .add_source(source)
         .pipe(decode, concurrency=8, name="decode")
         .add_sink(4)
         .build(
             num_threads=24,
             task_hook_factory=hook_factory,
         )
     )

Packing is not a fundamental gap: implement token- or byte-budget packing with
a custom :py:class:`spdl.pipeline.defs.Aggregator`. Multiprocessing is also not a
gap, but SPDL asks you to choose the process boundary instead of applying one
worker setting to the whole dataset.

Migration workflow
------------------

1. **Write down the semantic contract.** Record the exact sample order, epoch
   length, sharding, shuffle seed, filtering, batch/drop behavior, random
   augmentation, mixture weights, and checkpoint/resume guarantees.
2. **Create a parity test.** Compare stable sample identifiers and batch shapes
   over multiple epochs and ranks. Test restart from a mid-epoch checkpoint if
   the Grain pipeline supports it.
3. **Move sampling into the source.** Start with the matching SPDL distributed
   sampler. Feed its indices to ``pipe(source)`` for automatic ``__getitem__``
   lookup, or make the source yield records directly.
4. **Split work by resource.** Use separate stages for remote read, decode,
   tokenization or augmentation, collation, and device transfer. This is the
   change that enables SPDL's stage-level optimization.
5. **Choose execution per stage.** Prefer async functions for async I/O and
   threads for native operators that release the GIL. For measured GIL-holding
   work, use ``pipe(..., executor=ProcessPoolExecutor(...),
   output_order="input")`` when order is required. A multi-stage
   ``to(ProcessPoolExecutorConfig(...))`` region instead emits in completion
   order, requires application-level reordering, and must be closed with
   ``to(MAIN_PROCESS)`` before adding the sink.
6. **Batch late enough to expose parallelism.** Aggregate records, then collate
   once. For variable-length data, use a custom aggregator and test its final
   flush and ``drop_last`` behavior.
7. **Bound every queue.** Start with small sink and inter-stage buffers. Large
   prefetch values can hide a bottleneck while increasing host memory and
   checkpoint replay distance.
8. **Tune from measurements.** Inspect queue occupancy and per-stage task time,
   then change one stage's concurrency at a time. Optimize end-to-end training
   step time, not loader microbenchmark throughput alone.
9. **Roll out with measured gates.** Compare output fingerprints, recovery
   behavior, and training metrics while moving traffic to the SPDL path.

.. seealso::

   :doc:`../optimization_guide/index`
      Measure stage statistics and tune concurrency.

   :doc:`../getting_started/execution_models`
      Choose threading, multiprocessing, or hybrid execution.

   :doc:`PyTorch migration guide <../migration/pytorch>`
      See another dataset-first loader translated into SPDL stages.
