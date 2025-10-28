High-Level Loading Functions
=============================

Overview
--------

To load audio/video/image data from a file or in-memory buffer, you can use the following high-level functions:

- :py:func:`spdl.io.load_audio`
- :py:func:`spdl.io.load_video`
- :py:func:`spdl.io.load_image`

These functions provide a simple interface that handles the entire decoding pipeline internally.
They return a :py:class:`spdl.io.CPUBuffer` or :py:class:`spdl.io.CUDABuffer` object containing the decoded data as a contiguous memory region.

Basic Usage
-----------

Loading Audio
~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   # Load from file with minimum input
   buffer = spdl.io.load_audio("audio.mp3")

   # The buffer can be converted to different array types

   # NumPy array
   array = spdl.io.to_numpy(buffer)  # shape: (time, channel), dtype: float32

   # PyTorch tensor
   tensor = spdl.io.to_torch(buffer)

   # JAX array
   jax_array = spdl.io.to_jax(buffer)

   # Load from bytes
   data: bytes = download_from_remote_storage("audio.mp3")
   buffer = spdl.io.load_audio(data)

   # Load specific time window
   buffer = spdl.io.load_audio("audio.wav", timestamp=(5.0, 10.0))  # 5-10 seconds

Loading Video
~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   # Load from file with minimum input
   buffer = spdl.io.load_video("video.mp4")

   # The buffer can be converted to different array types

   # NumPy array
   array = spdl.io.to_numpy(buffer)  # shape: (time, height, width, channel), dtype: uint8

   # PyTorch tensor
   tensor = spdl.io.to_torch(buffer)

   # JAX array
   jax_array = spdl.io.to_jax(buffer)

   # Load from URL
   buffer = spdl.io.load_video("https://example.com/video.mp4")

   # Load specific time window
   buffer = spdl.io.load_video("video.mp4", timestamp=(0.0, 5.0))  # First 5 seconds

Loading Image
~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   # Load from file with minimum input
   buffer = spdl.io.load_image("image.jpg")

   # The buffer can be converted to different array types

   # NumPy array
   array = spdl.io.to_numpy(buffer)  # shape: (height, width, channel), dtype: uint8

   # PyTorch tensor
   tensor = spdl.io.to_torch(buffer)

   # JAX array
   jax_array = spdl.io.to_jax(buffer)

   # Load from bytes
   data: bytes = download_from_remote_storage("image.png")
   buffer = spdl.io.load_image(data)

Buffer Objects
--------------

The buffer objects returned by the loading functions implement standard array interface protocols:

- `The array interface protocol <https://numpy.org/doc/stable/reference/arrays.interface.html>`_ for CPU buffers
- `The CUDA array interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for CUDA buffers

This allows zero-copy conversion to commonly used array classes.

SPDL provides the following conversion functions:

- :py:func:`spdl.io.to_numpy` (CPU only)
- :py:func:`spdl.io.to_torch`
- :py:func:`spdl.io.to_jax` (CPU only)
- :py:func:`spdl.io.to_numba`

Default Output Formats
-----------------------

By default, the loading functions produce the following formats:

**Audio:**

- Sample format: 32-bit floating point (``float32``)
- Channel layout: Interleaved (channel-last)
- Shape: ``(time, channel)``

**Video:**

- Pixel format: RGB24 (interleaved)
- Layout: ``NHWC`` where ``N`` is time, ``C=3`` (RGB)
- Shape: ``(time, height, width, channel)``
- Data type: ``uint8``

**Image:**

- Pixel format: RGB24 (interleaved)
- Layout: ``HWC`` where ``C=3`` (RGB)
- Shape: ``(height, width, channel)``
- Data type: ``uint8``

.. note::

   The ``spdl.io`` module supports decoding images and videos into various color formats beyond RGB,
   including YUV420p, NV12, and other pixel formats. You can specify the desired pixel format using
   the ``pix_fmt`` parameter in :py:func:`spdl.io.get_video_filter_desc`. See :doc:`filtering` for details.

Customizing Output Format
--------------------------

You can customize the output format by providing a ``filter_desc`` parameter.
SPDL provides helper functions to construct filter descriptions:

- :py:func:`spdl.io.get_audio_filter_desc` - For audio preprocessing
- :py:func:`spdl.io.get_video_filter_desc` - For video and image preprocessing
- :py:func:`spdl.io.get_filter_desc` - For advanced custom filters

.. note::

   :py:func:`spdl.io.get_video_filter_desc` can be used for both image and video loading.

These helper functions allow you to specify common preprocessing operations such as:

- Resampling/rescaling
- Format conversion
- Cropping
- Frame rate adjustment
- Trimming to specific number of frames/samples

For detailed information about custom filter creation, see :doc:`filtering`.

Custom Audio Format Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   # Resample to 16kHz, convert to mono, change format to 16-bit signed integer
   buffer = spdl.io.load_audio(
       "audio.wav",
       filter_desc=spdl.io.get_audio_filter_desc(
           sample_rate=16_000,
           num_channels=1,
           sample_fmt="s16p",
           num_frames=80_000,
       )
   )
   array = spdl.io.to_numpy(buffer)

Custom Video Format Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   # Resize to 256x256, then crop to 224x224, adjust frame rate to 30fps
   buffer = spdl.io.load_video(
       "video.mp4",
       filter_desc=spdl.io.get_video_filter_desc(
           frame_rate=(30, 1),  # 30 fps as (numerator, denominator)
           scale_width=256,
           scale_height=256,
           scale_algo='bicubic',
           crop_width=224,
           crop_height=224,
           num_frames=10,
       )
   )
   tensor = spdl.io.to_torch(buffer)

Custom Image Format Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   # Note: get_video_filter_desc works for images too
   buffer = spdl.io.load_image(
       "image.jpg",
       filter_desc=spdl.io.get_video_filter_desc(
           scale_width=256,
           scale_height=256,
           crop_width=224,
           crop_height=224,
       )
   )
   array = spdl.io.to_numpy(buffer)

Transferring to GPU
-------------------

High-level loading functions accept a ``device_config`` parameter to automatically transfer decoded data to GPU:

.. code-block:: python

   import spdl.io

   # Create GPU device configuration
   cuda_config = spdl.io.cuda_config(device_index=0)

   # Audio
   audio_buffer = spdl.io.load_audio("audio.mp3", device_config=cuda_config)
   audio_tensor = spdl.io.to_torch(audio_buffer)  # CUDA tensor

   # Video
   video_buffer = spdl.io.load_video("video.mp4", device_config=cuda_config)
   video_tensor = spdl.io.to_torch(video_buffer)  # CUDA tensor

   # Image
   image_buffer = spdl.io.load_image("image.jpg", device_config=cuda_config)
   image_tensor = spdl.io.to_torch(image_buffer)  # CUDA tensor

.. note::

   The ``device_config`` parameter performs CPU decoding followed by GPU transfer.
   It does **not** use hardware-accelerated video decoding.

   For hardware-accelerated video decoding using NVIDIA's NVDEC hardware decoder, see :doc:`gpu_decoding`.

.. _custom-allocators:

Custom Memory Allocators
~~~~~~~~~~~~~~~~~~~~~~~~~

For better integration with PyTorch or other frameworks, you can specify custom memory allocators
via the ``allocator`` parameter in :py:func:`spdl.io.cuda_config`:

.. code-block:: python

   import spdl.io
   import torch

   # Use PyTorch's caching allocator
   cuda_config = spdl.io.cuda_config(
       device_index=0,
       allocator=(
           torch.cuda.caching_allocator_alloc,
           torch.cuda.caching_allocator_delete
       )
   )

   # Load and transfer using PyTorch's allocator
   buffer = spdl.io.load_video("video.mp4", device_config=cuda_config)
   tensor = spdl.io.to_torch(buffer)
   # Memory is managed by PyTorch's allocator

**Benefits of custom allocators:**

- Unified memory management with your framework
- Better memory pooling and reuse
- Reduced memory fragmentation

.. note::

   Custom allocators work with all GPU operations in SPDL, including hardware-accelerated
   video decoding. See :doc:`gpu_decoding` for more details.
