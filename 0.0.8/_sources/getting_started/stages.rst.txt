Pipeline Stages
===============

.. py:currentmodule:: spdl.dataloader

:py:class:`Pipeline` is composed of multiple stages.
There are mainly three kind of stages.

- Source
- Processing
- Sink (buffer)

Source
------

Source specifies where the data are located. This is typically file paths or URLs.
The source can be set with :py:meth:`PipelineBuilder.add_source`
method. The only requirement for the source object is that it must implement
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

   # Using some imaginary client
   async def list_bucket(bucket: str) -> AsyncIterator[str]:
       client = client.connect()
       async for route in client.list_bucket(bucket):
           yield route

.. note::

   Since the source object is executed in async event loop, if the source is
   ``Iterable`` (synchronous iterator), the source object must be lightweight
   and refrain from performing blocking operation.

   Running a blocking operation in async event loop can, in turn, prevent the
   loop from scheduling callbacks, prevent tasks from being canceled, and
   prevent the background thread from joining.

Processing
----------

Pre-processing is where a variety of operations are applied to the items passed
from the previous stages.

You can define processing stage by passing an operator function (callable) to
:py:meth:`~PipelineBuilder.pipe`. (Also there is :py:meth:`~PipelineBuilder.aggregate`
method, which can be used to stack multiple items.)

The operator can be either async function or synchronous function. Either way,
the operator must take exactly one argument†, which is an output from the earlier
stage.

.. note::

   † If you need to pass multiple objects between stages, use tuple or define a
   protocol using :py:class:`~dataclasses.dataclass`.

The following diagram illustrates a pipeline that fetch images from remote
locations, batch decode and send data to GPU.

.. mermaid::

   flowchart TD
       A[Source] --> B(Acquire data)
       B --> C(Batch)
       C --> D(Decode & Pre-process &Transfer to GPU & Convert to Tensor)
       D --> E[Sink]

An implementation could look like this.
It uses :py:func:`spdl.io.async_load_image_batch`, which can decode and resize images
and send the decoded frames to GPU in asynchronously.

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
   >>> async def process(data: list[bytes]) -> Tensor:
   ...     """Given raw image data, decode, resize, batch and transfer data to GPU"""
   ...     buffer = spdl.io.async_load_image_batch(
   ...         data,
   ...         width=224,
   ...         height=224,
   ...         cuda_config=spdl.io.cuda_config(device_index=0),
   ...     )
   ...     return spdl.io.to_torch(buffer)
   >>>
   >>> pipeline = (
   ...     PipelineBuiler()
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

Sink is a buffer where the results of the pipeline is accumulated.
A sink can be attached to pipeline with :py:meth:`PipelineBuilder.add_sink` method.
You can specify how many items can be buffered in the sink.
