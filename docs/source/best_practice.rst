Best Practices
==============

Skip the use of unbatched tensor
--------------------------------

For efficient and performant data processing, it is advised to not create
an intermediate Tensor for each individual media object (such as single image),
instead create a batch Tensor directly.

We recommend decoding individual frames, then using :py:func:`spdl.io.convert_frames`
to create a batch Tensor directly without creating an intermediate Tensors.

If you are decoding a batch of images, and you have pre-determined set of images
that should go together into a same batch, use
:py:func:`spdl.io.load_image_batch` (or its async variant
:py:func:`spdl.io.async_load_image_batch`).

Otherwise, demux, decode and preprocess multiple media, then combine them with
:py:func:`spdl.io.convert_frames` (or :py:func:`spdl.io.async_convert_frames`).
For example, the following functions implement decoding and tensor creation
separately.

.. code-block::

   import spdl.io
   from spdl.io import ImageFrames

   def decode_image(src: str) -> ImageFrames:
       packets = spdl.io.async_demux_image(src)
       return spdl.io.async_decode_packets(packets)

   def batchify(frames: list[ImageFrames]) -> ImageFrames:
       buffer = spdl.io.convert_frames(frames)
       return spdl.io.to_torch(buffer)

They can be combined in :py:class:`~spdl.dataloader.Pipeline`, which automatically
discards the items failed to process (for example due to invalid data), and
keep the batch size consistent by using other items successefully processed.

.. code-block::

   from spdl.dataloader import PipelineBuilder

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(decode_image, concurrency=...)
       .aggregate(32)
       .pipe(batchify)
       .add_sink(3)
       .build(num_threads=...)
   )

.. seealso::

   :py:mod:`multi_thread_preprocessing`

Make Dataset class composable
-----------------------------

If you are publishing a dataset and providing an implementation of
`Dataset` class, we recommend to make it composable.

That is, in addition to the conventional ``Dataset`` class that
returns Tensors, make the components of the ``Dataset``
implementation available by breaking down the implementation into

* Iterator (or map) interface that returns paths instead of Tensors.
* A helper function that loads the soure path into Tensor.

For example, the interface of a ``Dataset`` for image classification
might look like the following.

.. code-block::

   class Dataset:
       def __getitem__(self, key: int) -> tuple[Tensor, int]:
           ...

We recommend to separate the source and process and make them additional
public interface.
(Also, as descibed above, we recommend to not convert each item into
``Tensor`` for the performance reasons.)

.. code-block::

   class Source:
       def __getitem__(self, key: int) -> tuple[str, int]:
           ...

   def load(data: tuple[str, int]) -> tuple[ImageFrames, int]:
       ...

and if the processing is composed of stages with different bounding
factor, then split them further into primitive functions.

.. code-block::

   def download(src: tuple[str, int]) -> tuple[bytes, int]:
       ...

   def decode_and_preprocess(data: tuple[bytes, int]) -> tuple[ImageFrames, int]:
       ...

then the original ``Dataset`` can be implemented as a composition

.. code-block::

   class Dataset:
       def __init__(self, ...):
           self._src = Source(...)

       def __getitem__(self, key:int) -> tuple[str, int]:
           metadata = self._src[key]
           item = download(metadata)
           frames, cls = decode_and_preprocess(item)
           tensor = spdl.io.to_torch(frames)
           return tensor, cls

Such decomposition makes the dataset compatible with SPDL's Pipeline,
and allows users to run them more efficiently

.. code-block::

   pipeline = (
       PipelineBuilder()
       .add_source(Source(...))
       .pipe(download, concurrency=8)
       .pipe(decode_and_preprocess, concurrency=4)
       ...
       .build(...)
   )
