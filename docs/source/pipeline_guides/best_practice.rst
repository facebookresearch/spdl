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
