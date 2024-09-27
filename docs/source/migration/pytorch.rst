Practical Example
=================

If you are using :py:class:`torch.utils.data.DataLoader` and decided to
use to SPDL, there are couple of conceptual differences
you want to be aware of.

There are three distinguished components in PyTorch's DataLoader.

1. Sampler - Generates keys.
2. Dataset - Map an input key to a Tensor.
3. Collate function - Merge multiple Tensors into one batch Tensor.

One notable thing about this composition is that dataset contains the logic
to convert the data source to Tensor, and it directly maps keys to Tensors.
It prohibits the access to the data source or data itself, and it does not
distinguish the steps for data acquisition, decoding and preprocessing nor
expose them.

So as to achieve high throughput, it is important to separate the operations of
different natures and configure them, but there is no native support for
that in :py:class:`torch.utils.data.Dataset`.

Now let's look into a task of image loading for classification task, using
TorchVision's :py:class:`torchvision.datasets.ImageNet`.

Here is the base dataset instance we are going to use.

.. code-block::

   from torchvision.datasets import ImageNet
   from torchvision.transforms import Compose, PILToTensor, Resize

   dataset = ImageNet(
       "dataset_directory",
       transform=Compose([Resize((224, 224)), PILToTensor()]),
   )

The images are decoded in
:py:class:`torchvision.datasets.ImageNet.__getitem__` method, then
resized and converted to Tensor through :py:class:`~torchvision.transforms.Resize`
and :py:class:`~torchvision.transforms.PILToTensor` transforms.

Decoding images in threads
--------------------------

When using SPDL before free-threaded (a.k.a no-GIL) Python becomes available
and stable, we need to check whether the dataset instance can be used
in threaded environment in a performant manner.

- Does the internal loading logic release GIL?
- Is the dataset instance thread-safe?

If the answer to both questions is yes, then we can start by reusing
the dataset logic as-is in :py:class:`spdl.dataloader.Pipeline`.

TorchVision's dataset implementations meet these requirements.
By default, TorchVision uses Pillow to load images, and
Pillow releases GIL. The dataset is thread-safe.

The following code snippets show how one can use PyTorch
:py:class:`~torch.utils.data.DataLoader` to load images without batching.

.. code-block::

   import torch

   dataloader = torch.utils.data.DataLoader(
       dataset,
       batch_size=None,
       num_workers=num_workers,
   )

   for batch, classes in dataloader:
       ...


The following is an equivalent implementation using SPDL
:py:class:`~spdl.dataloader.Pipeline`.

.. code-block::

   from spdl.dataloader import PipelineBuilder

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

SPDL Pipeline is faster than PyTorch DataLoader at all stages,
which suggests that the operations executed here
(Image decoding, resising and Tensor conversions) all release GIL.

Many PyTorch operators (and also NumPy operators) release GIL,
so there is a good chance that they work fine in thread-based pipeline.

Batching images
---------------

Now we know that the basic loading operation works, we add batching.
In PyTorch DataLoader, providing `batch_size` enables batching
internally.

By default, it uses :py:func:`torch.utils.data.default_collate`
function.

.. code-block::

   dataloader = torch.utils.data.DataLoader(
       dataset,
       batch_size=32,
       # ^^^^^^^^^^^^
       num_workers=num_workers,
   )

   for batch, classes in dataloader:
       ...

In SPDL, we use :py:meth:`~spdl.dataloader.PipelineBuilder.aggregate`
method to buffer the results coming from upstream into a list.

The the buffered items need to be passed to collate function explicitly.

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

The batching gives huge boost to PyTorch DataLoader pipeline, while
SPDL becomes slightly slower than before.
Now, PyTorch DataLoader is faster than SPDL Pipeline.

This means that, simply running the existing data loading pipeline, which
is designed for multi-processing, does not improve the performance.

This might change with free-threaded, but that's the current state, and
also the reason why SPDL cannot provide a simple drop-in replacement of
PyTorch DataLoader class.

To understand how to workaround this, let's see how both pipelines scale.

Understanding the Performance Trend
-----------------------------------

We run the same pipeline while changing the batch size from 32 to 256 and see
how they scale.

.. include:: ../plots/migration_3.txt

.. include:: ../plots/migration_4.txt

We can make the following observations.

1. PyTorch DataLoader does not scale well beyond 8-16 workers,
   while SPDL Pipeline can achieve similar or higher throughput.
2. As the batch size increases, PyTorch DataLoader's peak throughput drops,
   while SPDL Pipeline sustain similar or even higher throughput.
3. The way SPDL Pipeline's performance scale is consistant for different batch sizes.

Why does the performance of PyTorch DataLoader drops as the batch size increase?
And why it's not the case for SPDL Pipeline?

Since they use the same operations, we can deduce that the difference is coming from
process and thread.
One of the major difference between process-based and thread-based data preparation
is the treatment of the batch after its creation.

The following figure shows in detail how images are prepared when using PyTorch
DataLoader.

.. include:: ../plots/multi_thread_preprocessing_chart_torch.txt

The batch created in a background process is serialized into byte string and written to
the memory space shared between the main process and the background process.
The main process then de-serialize the byte string and re-create the batch.

This means that the batch tensor is at least copied twice after its creation.

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
contiguous memory format. In media processing, the data are more often represented as
non-contiguouos memories regions (like separate image planes),
and usually they are not in the size.
For example, YUV420 format, which is one of the most commoly used format uses 12 bits per
pixel to store data, while RGB tensors uses 24 bits per pixel of contiguous memeory.

Loadidng JPEG files using Pillow converts the data to RGB.
This RGB data is then resized to the target resolution, and coverted to a Tensor representing
a single image. Finally, image tensors are copied into batch Tensor.

This process adds multiple of redundant memory allocations and copying.

Let's do a back-of-the-envelope calculation to see how wasteful this can be.
Suppose we want to create a batch of 32 images at 224x224 and all the source images are
720x480 YUV420 format.

One decoded image kept at YUV420 format occupies 720x480x12 = 4147200 bits ~= 0.52 MB.
When this images is converted to RGB, then it occupies twice the size of YUV420, 1.04 MB.
So converting images to RGB after loading consumes extra 32x0.52 = 16 MB of memory.

Next, once the image is resized to 224x224 RGB format, TorchVision/Pillow convert the data
to Tensors of single images. That is, 32 of contiguous 224x224x3 bytes ~= 4.8 MB is
allocated, copied but discarded when batch tensor is created.

Therefore, when using Pillow and TorchVision, creating a batch of 32 images at 224x224
(which is 4.8 MB) allocates, copies and discards redundant memory of about 20MB.

When this data is transfered from subprocess to the main process, 2x4.8 MB extra copies
are created.

In SPDL, we implemented the I/O module, which avoids these redundant memory allocations and
copying, and converts the data to contiguous format only when batching the resulted images.

The following figure illustrates the SPDL pipeline with this SPDL I/O module at the same
level of abstraction as the previous PyTorch DataLoader figure.

.. include:: ../plots/multi_thread_preprocessing_chart_spdl.txt

The image processing part has fewer steps than PyTorch DataLoader equivalent.

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

.. include:: ../plots/migration_5.txt

The new pipeline is faster than PyTorch DataLoader at all the level of concurrency
and the size of the batch tensor.

Summary
-------

We looked at how one can replace PyTorch DataLoader with SPDL Pipeline while
improving the performance of data loading.

The initial step can be mechanically applied, but to ensure that the resulting
pipeline is more perfomant, it is necessary to benchmark and adjust parts of the
pipeline.

We recommend to use :py:mod:`spdl.io` module for processing media data. It is
designed for scaling througput and has small memory footprint.

For more complete performance analysis on the SPDL-based pipeline, please refer
to :py:mod:`multi_thread_preprocessing`.
This example measures the pipeline with more processing and GPU data transfer.
