Migrating from PyTorch DataLoader
=================================

If you are using :py:class:`torch.utils.data.DataLoader` and decided to
migrate to SPDL, there are couple of conceptual differences
you want to be aware of.

There are three distinguished components in PyTorch's DataLoader.

1. Sampler - Generates keys.
2. DataSet - Map an input key to a Tensor.
3. Collate function - Convert input Tensors to a batch Tensor.

One notable thing about this composition is that DataSet contains the logic
to convert the data source to Tensor, and it directly maps keys to Tensors.
It does not distinguish the steps for data acquisition, decoding and
preprocessing nor expose them.

To achieve high throughput, it is important to separate the operations of
different natures and configure them, but there is no native support for
that. We will come back to this later for improved performance.

Moving from Process to Thread
-----------------------------

The first question to ask is whether the DataSet class is re-usable.
That is, does the logic internal to DataSet release GIL? Is it thread-safe?
If the answere to both questions are yes, then we can start by reusing
the DataSet logic as-is in :py:class:`spdl.dataloader.Pipeline`.

TorchVision's DataSet implementations meet these requirements.
By default, TorchVision uses Pillow to load images in its DataSet implementations.
Pillow releases GIL, and is thread safe.

The following code snippets show how one can use PyTorch
:py:class:`~torch.utils.data.DataLoader` to load images, and
its equivalent using SPDL's :py:class:`~spdl.dataloader.Pipeline`.

.. code-block::

   import torch
   from torchvision.datasets import ImageNet
   from torchvision.transforms import Compose, PILToTensor, Resize

   dataset = ImageNet(
       "dataset_directory",
       transform=Compose([Resize((224, 224)), PILToTensor()]),
   )

   dataloader = torch.utils.data.DataLoader(
       dataset,
       batch_size=None,
       num_workers=num_workers,
   )

   for batch, classes in dataloader:
       ...

.. code-block::

   from spdl.dataloader import PipelineBuilder
   import torch
   from torchvision.datasets import ImageNet

   pipeline = (
       PipelineBuilder()
       .add_source(range(len(dataset)))
       .pipe(
           dataset.__getitem__,
           concurrency=num_workers,
           output_order="input",
       )
       .add_sink(prefetch_factor)
       .build(num_threads=num_workers)
   )

   with pipeline.auto_stop():
       for batch, classes in dataloader:
           ...

Running them with different concurrency, we get the following
performance.

.. include:: ../plots/migration_1.txt

It's good that decoding and transforms work in threads. Many PyTorch
operators release GIL, so there is a good chance that it works fine.

Batching
--------

Now we know that the base loading operation works, let's add batching.
We change the code as follow.


.. code-block::

   dataloader = torch.utils.data.DataLoader(
       dataset,
       batch_size=32,
       # ^^^^^^^^^^^^
       num_workers=num_workers,
   )

   for batch, classes in dataloader:
       ...

.. code-block::

   pipeline = (
       PipelineBuilder()
       .add_source(range(len(dataset)))
       .pipe(
           dataset.__getitem__,
           concurrency=num_workers,
           output_order="input",
       )
       .aggregate(32)
       # ^^^^^^^^^^^^
       .pipe(torch.utils.data.default_collate)
       # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       .add_sink(prefetch_factor)
       .build(num_threads=num_workers)
   )

Running this, we get the following result.

.. include:: ../plots/migration_2.txt

The batching gives both pipelines boost, but PyTorch DataLoader benefits more and
now SPDL is not as fast as PyTorch DataLoader.

So this is it? SPDL does not help? Not so fast.

Understanding the Performance Trend
-----------------------------------


Let's change the batch size to see how these data processing pipelines scale.

Runnig the same code while changing the batch size from 32 to 256, we get the following
result.

.. include:: ../plots/migration_3.txt

We can make the following observations.

1. PyTorch DataLoader does not scale well beyond 8-16 workers.
2. As the batch size increases, PyTorch DataLoader's peak performance drops.
3. SPDL Pipeline does not exhibit either issue.
4. The way SPDL Pipeline's performance scale is consistant for different batch sizes.

Why does the performance of PyTorch DataLoader degrade as the batch size increase, while
SPDL Pipeline has consistent performance?

One reason is the increased memory allocation and copy originated from the use of subprocess.

The following fiture shows detailed process the images are prepared in when using
PyTorch DataLoader.

.. include:: ../plots/multi_thread_preprocessing_chart_torch.txt

The batch created in a background process is serialized into byte string and written to
the memory space shared between the main process and the background process.

The main process de-serialize the byte string and re-create the batch. So the batch
Tensor is copied at least twice.

So as the size of the batch tensor grows, the main process needs to allocate and copy more
memory to fetch the batch.

In SPDL Pipeline, this is not the case because it uses threads to execute works concurrently.
Threads, unlike processes, share the memory space, so the batch tensor created by the background
thread can be directly used by the main thread. There is no need to copy it.

Switching to SPDL I/O
---------------------

In addition to the redundant memory copy originated from inter-process communication, there
are few more memory-related inefficiency in the pipelines.

The image data coming out of :py:class:`~torchvision.datasets.ImageNet` is Tensor, which is
contiguous memory format. In media processing, the data are often represented as incontiguouos
memories regions (image planes), and usually they are not in the size.

Loaidng JPEG file using Pillow, converting to RGB before resizing and converting to Tensor before
batching adds multiple of redundant memory allocations and copying.

In SPDL, we implemented the I/O module, which avoids these redundant memory allocations and
copying, and converts the data to contiguous format when batching the resulted images.

Let's swap the image decoding part and see how it helps.

We change the image decoding part (which was happening inside of
:py:class:`~torchvision.datasets.ImageNet` class) as follow.

First, we implement the function that decodes an image, but does not convert it to Tensor yet.
The result of :py:func:`spdl.io.decode_packets` is :py:class:`~spdl.io.ImageFrames`, which holds
the frame data decoded by FFmpeg as-is.

.. code-block::

    filter_desc = spdl.io.get_video_filter_desc(
        scale_width=224,
        scale_height=224,
    )

   def decode_image(path: str) -> ImageFrames:
       packets = spdl.io.demux_image(path)
       return spdl.io.decode_packets(packets, filter_desc=filter_desc)

Then we swap the image decoding function.

.. code-block::

   dataset = ImageNet(
       "dataset_directory",
       loader = decode_image,
       # ^^^^^^^^^^^^^^^^^^^
   )

Then we implement a batching function, which is somewhat equivalent to collate function, but
its inputs are not Tensor.
The :py:func:`spdl.io.convert_frames` function receives multiple Frame objects and create one
contiguous memory and copies the data from the Frames.
The result is a :py:class:`spdl.io.CPUBuffer` instance, which implements
`NumPy Array Interface <https://numpy.org/doc/stable/reference/arrays.interface.html>`_, so it
can be converted to PyTorch Tensor without copying data.
:py:func:`spdl.io.to_torch` function performs this operation.

.. code-block::

   def convert(items):
       frames, clsses = list(zip(*items))
       buffer = spdl.io.convert_frames(frames)
       tensor = spdl.io.to_torch(buffer).permute(0, 3, 1, 2)
       return tensor, clsses

Then we put them together to build the Pipeline.

.. code-block::

   pipeline = (
       PipelineBuilder()
       .add_source(range(len(dataset)))
       .pipe(
           dataset.__getitem__,
           concurrency=num_workers,
           output_order="input",
       )
       .aggregate(batch_size)
       .pipe(convert)
       .add_sink(prefetch_factor)
       .build(num_threads=num_workers)
   )

Running the pipelinn with different number of threads, we get the following result.

.. include:: ../plots/migration_4.txt

The new pipeline is faster than PyTorch DataLoader at all the level of concurrency
and the size of the batch tensor.

Summary
-------

We looked at how one can migrate from PyTorch DataLoader to SPDL Pipeline while
improving the performance of data loading.

The initial step can be mechanically applied, but to ensure that the resulting
pipeline is more perfomant, it is necessary to benchmark and adjust parts of the
pipeline.

We recommend to use :py:mod:`spdl.io` module for processing media data. It is
designed for scaling througput and has small memory footprint.

For more complete performance analysis on the SPDL-based pipelien, please refer
to :py:mod:`multi_thread_preprocessing`.
This example measures the pipeline with more processing and GPU data transfer.
