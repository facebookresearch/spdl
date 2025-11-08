Pipeline Stages
===============

.. py:currentmodule:: spdl.pipeline

:py:class:`Pipeline` is composed of multiple stages.
There are mainly three kinds of stages.

- Source
- Processing
- Sink (buffer)

Source
------

The source specifies the origin of data, which is typically file paths or URLs.
The source can be set with :py:meth:`PipelineBuilder.add_source` method.
The only requirement for the source object is that it must implement
:py:class:`~collections.abc.Iterable` or :py:class:`~collections.abc.AsyncIterable`
interface.

For example

- Load a list of paths from a file.

.. code-block::

   def load_path_from_file(input_path: str):
       with open(input_path, "r") as f:
           for line in f:
               if path := line.strip():
                   yield path

- Files in directories

.. code-block::

   def find_files(path: Path, ext: str):
       yield from path.glob(f'**/*{ext}')

- Asynchronously list files in remote storage

.. code-block::

   # Using some imaginary async network client
   async def list_bucket(bucket: str) -> AsyncIterator[str]:
       client = client.connect()
       async for route in client.list_bucket(bucket):
           yield route

.. note::

   Since the source object is executed in async event loop, if the source is
   ``Iterable`` (synchronous iterator), the source object must be lightweight
   and refrain from performing blocking operations.

   Running a blocking operation in async event loop can, in turn, prevent the
   loop from scheduling callbacks, prevent tasks from being canceled, and
   prevent the background thread from joining.

Processing
----------

Pre-processing is where a variety of operations are applied to the items passed
from the previous stages.

You can define a processing stage by passing an operator function (callable) to
:py:meth:`~PipelineBuilder.pipe`. You can also use :py:meth:`~PipelineBuilder.aggregate`
and :py:meth:`~PipelineBuilder.disaggregate` to stack and unstack multiple items.

The operator can be either an async function or a synchronous function.
It must take exactly one argument†, which is an output from the earlier
stage.

.. admonition:: Note†
   :class: note

   - If you need to pass multiple objects between stages, use tuple or define a
     protocol using :py:func:`~dataclasses.dataclass`.
   - If you want to use an existing function which takes additional arguments,
     you need to convert the function to a univariate function by manually
     writing a wrapper function or using :py:func:`functools.partial`.

The following diagram illustrates a pipeline that fetches images from remote
locations, batch decodes and sends data to GPU.

.. mermaid::

   flowchart TD
       A[Source] --> B(Acquire data)
       B --> C(Batch)
       C --> D(Decode & Pre-process &Transfer to GPU & Convert to Tensor)
       D --> E[Sink]

An implementation could look like this.
It uses :py:func:`spdl.io.load_image_batch`, which can decode and resize images
and send the decoded frames to GPU asynchronously.

.. code-block::

   >>> import spdl.io
   >>> from spdl.dataloader import PipelineBuilder
   >>>
   >>> def source() -> Iterator[str]:
   ...     """Returns the list of URLs to fetch data from"""
   ...     ...
   >>>
   >>> async def download(url: str) -> bytes:
   ...     """Download data from the given URL"""
   ...     ...
   >>>
   >>> def process(data: list[bytes]) -> Tensor:
   ...     """Given raw image data, decode, resize, batch and transfer data to GPU"""
   ...     buffer = spdl.io.load_image_batch(
   ...         data,
   ...         width=224,
   ...         height=224,
   ...         cuda_config=spdl.io.cuda_config(device_index=0),
   ...     )
   ...     return spdl.io.to_torch(buffer)
   >>>
   >>> pipeline = (
   ...     PipelineBuilder()
   ...     .add_source(source())
   ...     .pipe(download)
   ...     .aggregate(32)
   ...     .pipe(process)
   ...     .add_sink(4)
   ...     .build()
   ... )
   >>>
   >>>


Sink
----

Sink is a buffer where the results of the pipeline are accumulated.
A sink can be attached to a pipeline with :py:meth:`PipelineBuilder.add_sink` method.
You can specify how many items can be buffered in the sink.

Advanced: Merging Multiple Pipelines
-------------------------------------

For more complex data loading scenarios, you can merge outputs from multiple independent
pipelines using :py:class:`~spdl.pipeline.defs.MergeConfig`. This is useful when you need to:

- Combine data from different sources (e.g., multiple datasets or storage locations)
- Process different types of data in parallel and merge them downstream
- Build complex data loading patterns that go beyond linear pipeline structures

The :py:func:`~spdl.pipeline.defs.Merge` function creates a merge configuration that combines
outputs from multiple :py:class:`~spdl.pipeline.defs.PipelineConfig` objects into a single stream.

.. note::

   The merge mechanism is not supported by :py:class:`PipelineBuilder`. You need to use
   the lower-level :py:mod:`spdl.pipeline.defs` API and :py:func:`~spdl.pipeline.build_pipeline`
   function to build pipelines with merge nodes.

.. seealso::

   :ref:`Example: Pipeline definitions <example-pipeline-definitions>`
      Demonstrates how to build a complex pipeline with merge nodes, including how to
      combine multiple data sources and process them through a unified pipeline.
