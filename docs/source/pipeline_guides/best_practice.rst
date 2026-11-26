.. _best-practices:

Design and best practices
=========================

.. _avoid-creating-intermediate-tensors:

Batch frames before converting to tensors
-----------------------------------------

When frames for a batch have compatible shapes, combine them with
:py:func:`spdl.io.convert_frames` and convert the resulting batch buffer to a
PyTorch tensor. This avoids building a tensor per image and then stacking
those tensors. If the inputs that belong in the batch are already known,
:py:func:`spdl.io.load_image_batch` can decode and combine them in one call.

For a streaming source, decode each image, aggregate decoded frames and then
convert the batch:

.. code-block:: python

   import spdl.io
   import torch
   from spdl.pipeline import PipelineBuilder

   def decode_image(path: str) -> spdl.io.ImageFrames:
       return spdl.io.decode_packets(spdl.io.demux_image(path))

   def make_batch(frames: list[spdl.io.ImageFrames]) -> torch.Tensor:
       return spdl.io.to_torch(spdl.io.convert_frames(frames))

   def build_pipeline(paths):
       return (
           PipelineBuilder()
           .add_source(paths)
           .pipe(decode_image, concurrency=4)
           .aggregate(32)
           .pipe(make_batch)
           .add_sink(3)
           .build(num_threads=4)
       )

Check the input shape and decode options, and decide how to handle failed
samples explicitly. Aggregating after decoding groups the outputs that reach
the stage; it does not preserve fixed input batch membership if any items are
skipped. ``aggregate`` emits a final partial batch by default. Set
``drop_last=True`` only when dropping it matches the workload's batch policy.

.. seealso::

   :py:mod:`multi_thread_preprocessing`

.. _make-dataset-class-composable:

Make dataset components composable
----------------------------------

A dataset can provide a conventional ``__getitem__`` that returns a tensor
while exposing the same building blocks for a pipeline: an iterable of paths
or record identifiers, a fetch function, and a decode or preprocessing
function. Reusing these functions keeps label mapping and processing consistent
without forcing the pipeline to materialize a tensor per record.

For example, expose the image records as a sequence of ``(path, label)`` pairs
and define the two processing steps separately. This example reads local files;
a remote source can supply a different fetch function with the same output.

.. code-block:: python

   from collections.abc import Sequence
   from pathlib import Path

   def fetch(record: tuple[str, int]) -> tuple[bytes, int]:
       path, label = record
       return Path(path).read_bytes(), label

   def decode_item(item: tuple[bytes, int]) -> tuple[spdl.io.ImageFrames, int]:
       data, label = item
       frames = spdl.io.decode_packets(spdl.io.demux_image(data))
       return frames, label

The map-style dataset calls both functions and converts the result to a tensor
for callers that expect one tensor per sample:

.. code-block:: python

   class Dataset:
       def __init__(self, records: Sequence[tuple[str, int]]) -> None:
           self.records = records

       def __len__(self) -> int:
           return len(self.records)

       def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
           frames, label = decode_item(fetch(self.records[index]))
           return spdl.io.to_torch(spdl.io.convert_frames(frames)), label

The SPDL pipeline uses the same records and functions. It schedules fetch and
decode separately and reuses ``make_batch`` from above to convert the frames
only after aggregation:

.. code-block:: python

   def collate_images(items: list[tuple[spdl.io.ImageFrames, int]]) -> tuple[torch.Tensor, torch.Tensor]:
       frames, labels = zip(*items, strict=True)
       return make_batch(list(frames)), torch.tensor(labels)

   def build_labeled_pipeline(records: Sequence[tuple[str, int]]):
       return (
           PipelineBuilder()
           .add_source(records)
           .pipe(fetch, concurrency=8, output_order="input")
           .pipe(decode_item, concurrency=4, output_order="input")
           .aggregate(32)
           .pipe(collate_images)
           .add_sink(3)
           .build(num_threads=8)
       )

Preserve record order, labels, and sampling behavior when migrating an existing
loader. Use the same record sequence or sampler in both paths; the example
iterates over the records in the supplied order. Split stages only when the
added scheduling cost is worth the overlap or separate concurrency controls.

.. _dataset-holds-data:

Datasets hold data and functions; the pipeline owns execution
-------------------------------------------------------------

A dataset, or any data-access helper that stages call, is data and plain
functions, synchronous or async. It does not own resources that decide how the
work runs; the pipeline does. So a dataset should not create thread pools,
executors or event loops, wrap calls in ``asyncio.run`` or another
sync-over-async bridge, install or replace the running event loop's default
executor, or manage a cache behind its methods.

The rule is most often broken in two ways: a sync/async bridge inside the
dataset, and a cache managed by the dataset.

**Sync/async bridge.** Some network and storage clients expose only async
APIs, and their "sync" wrappers just run the async call on a private event
loop. A traditional data loader interface, such as a PyTorch
``Dataset.__getitem__`` or ``__getitems__``, is synchronous and has nowhere to
await them, so each call gets bridged with ``asyncio.run`` or a similar helper,
and thread pools are then added inside the dataset to get the throughput back:

.. code-block:: python

   class RecordDataset:
       def __init__(self, keys: Sequence[str], client) -> None:
           self.keys = keys
           self.client = client
           self.pool = ThreadPoolExecutor(max_workers=16)

       def __getitem__(self, index: int) -> bytes:
           return asyncio.run(self.client.fetch(self.keys[index]))

       def __getitems__(self, indices: list[int]) -> list[bytes]:
           return list(self.pool.map(self.__getitem__, indices))

This costs more than it appears to:

- ``asyncio.run`` creates and closes an event loop and shuts down its default
  thread pool on every call.
- A pool owned by the dataset, or a default executor swapped in on the
  pipeline's loop, makes the dataset's work and every other coroutine on that
  loop compete for threads that the pipeline cannot see or size.
- An async stage does not need pipeline threads: its work is carried by the
  library's own machinery, such as the client's threads or processes, or by
  the operating system's async I/O. Problems start when an async stage hides
  blocking work, for example ``asyncio.to_thread`` of a synchronous read. That
  work runs on the event loop's default executor, which the pipeline sizes for
  its synchronous stages (in a process or interpreter region, from their
  concurrency alone), so nothing is planned for it. Make such work a
  synchronous stage.

Use the async function itself as the data-access helper and make it an async
stage. The pipeline awaits it on its own event loop, and ``concurrency=`` sets
how many requests are in flight. ``client.fetch`` stands for the service's
async API; create a client that is bound to an event loop in the process that
runs the stage.

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(keys)
       .pipe(client.fetch, concurrency=32)
       .aggregate(32)
       .add_sink(3)
       .build(num_threads=1)  # the async stage uses no worker threads
   )

Such a client may run its async calls on its own runtime, for example a native
executor with a global thread pool. That resource is invisible to the pipeline,
but it belongs to the client and is configured independently of the pipeline;
the dataset still owns no execution resources.

**Dataset-owned cache.** A local-cache hit and a remote fetch have very
different costs, so they need different concurrency. That is only possible when
cache read, remote fetch and cache write are separate pipeline stages; a
dataset that hides its cache behind one method forces one concurrency on all
three. Blocking cache I/O is a synchronous stage and the remote fetch is an
async stage:

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(keys)
       .pipe(partial(read_cache, cache_dir), concurrency=8)   # sync: (key, bytes | None)
       .pipe(partial(fetch_missing, client), concurrency=32)  # async: awaits only on a miss
       .pipe(partial(write_cache, cache_dir), concurrency=4)  # sync: writes fetched items
       .aggregate(32)
       .add_sink(3)
       .build(num_threads=12)  # threads for the two sync stages (8 + 4)
   )

Concurrency is set on each stage with ``concurrency=``, and with ``executor=``
when a stage outside a process or interpreter region needs its own pool. Each
step appears as its own stage in the pipeline's statistics. Stages emit items
in completion order by default; pass ``output_order="input"`` to each stage
when batches must keep the key order. When an existing dataset owns execution
resources, keep the original loader as the baseline, move its execution into
stages as a separate change, and check that the outputs match.
