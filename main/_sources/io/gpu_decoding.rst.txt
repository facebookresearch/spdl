Hardware-Accelerated Video Decoding
====================================

SPDL supports hardware-accelerated video decoding using NVIDIA's NVDEC (NVIDIA Video Decoder).
Hardware-accelerated decoding can significantly speed up video processing workflows, especially when the decoded frames
are used for GPU-based operations like deep learning inference.

Overview
--------

Hardware-accelerated video decoding with NVDEC offers several advantages:

- **Hardware acceleration**: Offloads decoding from CPU to dedicated video decoder hardware
- **Zero-copy operations**: Decoded frames stay in GPU memory, avoiding CPU-GPU transfers
- **Built-in preprocessing**: Hardware-accelerated resize and crop operations with zero overhead
- **Direct CUDA buffer output**: Returns :py:class:`~spdl.io.CUDABuffer` directly, no conversion needed

**Key differences from CPU decoding:**

+---------------------------+--------------------------------+--------------------------------+
| Feature                   | CPU Decoding                   | Hardware Decoding (NVDEC)      |
+===========================+================================+================================+
| Output                    | :py:class:`~spdl.io.CPUBuffer` | :py:class:`~spdl.io.CUDABuffer`|
+---------------------------+--------------------------------+--------------------------------+
| FFmpeg filters            | ✓ Supported                    | ✗ Not supported                |
+---------------------------+--------------------------------+--------------------------------+
| Resize/Crop               | Via FFmpeg filters             | Hardware-accelerated (zero     |
|                           |                                | overhead)                      |
+---------------------------+--------------------------------+--------------------------------+
| Buffer conversion         | Required                       | Not required (direct output)   |
+---------------------------+--------------------------------+--------------------------------+
| Bitstream filtering       | Optional                       | **Required** for H.264/HEVC    |
+---------------------------+--------------------------------+--------------------------------+

Basic Usage
-----------

The simplest way to use hardware-accelerated decoding is with :py:func:`~spdl.io.decode_packets_nvdec`:

.. code-block:: python

   import spdl.io

   # Demux video packets
   packets = spdl.io.demux_video("video.mp4")

   # Decode using hardware acceleration
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0)
   )

   # Convert to PyTorch CUDA tensor (zero-copy)
   tensor = spdl.io.to_torch(buffer)
   # tensor is on GPU: tensor.device == torch.device('cuda:0')

Hardware-Accelerated Resize and Crop
-------------------------------------

NVDEC provides hardware-accelerated resize and crop operations with **zero overhead**.
These operations happen during decoding and don't require additional processing time.

Resizing
~~~~~~~~

Resize video to a specific resolution:

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")

   # Decode and resize to 256x256
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0),
       scale_width=256,
       scale_height=256
   )

   tensor = spdl.io.to_torch(buffer)
   # tensor.shape: (num_frames, 3, 256, 256)

**Important constraints:**

- Width and height must be **even numbers** (divisible by 2)
- Negative values are not allowed
- Aspect ratio is not preserved automatically (image will be stretched)

Cropping
~~~~~~~~

Crop video by specifying pixels to remove from each edge:

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")  # Original: 1920x1080

   # Crop 100 pixels from left, 200 from right, 50 from top, 50 from bottom
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0),
       crop_left=100,
       crop_right=200,
       crop_top=50,
       crop_bottom=50
   )

   tensor = spdl.io.to_torch(buffer)
   # Output size: (1920 - 100 - 200) x (1080 - 50 - 50) = 1620 x 980
   # tensor.shape: (num_frames, 3, 980, 1620)

**Crop parameters:**

- ``crop_left``: Pixels to remove from the left edge
- ``crop_right``: Pixels to remove from the right edge
- ``crop_top``: Pixels to remove from the top edge
- ``crop_bottom``: Pixels to remove from the bottom edge
- All values must be non-negative

Combining Crop and Resize
~~~~~~~~~~~~~~~~~~~~~~~~~~

Crop and resize can be combined for efficient preprocessing:

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")  # Original: 1920x1080

   # First crop to 1620x980, then resize to 224x224
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0),
       crop_left=100,
       crop_right=200,
       crop_top=50,
       crop_bottom=50,
       scale_width=224,
       scale_height=224
   )

   tensor = spdl.io.to_torch(buffer)
   # tensor.shape: (num_frames, 3, 224, 224)

**Processing order**: Crop is applied first, then resize.

Pixel Format Conversion
------------------------

NVDEC outputs video in NV12 format by default, but can convert to RGB during decoding:

Default Output (NV12)
~~~~~~~~~~~~~~~~~~~~~

By default, NVDEC outputs NV12 format (YUV 4:2:0 with interleaved UV plane):

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0)
   )

   tensor = spdl.io.to_torch(buffer)
   # tensor.shape: (num_frames, 1, height * 3 // 2, width)
   # Top 2/3: Y plane (luma)
   # Bottom 1/3: Interleaved UV plane (chroma)

RGB Conversion
~~~~~~~~~~~~~~

Convert to RGB format during decoding:

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0),
       pix_fmt="rgb"  # or "bgr"
   )

   tensor = spdl.io.to_torch(buffer)
   # tensor.shape: (num_frames, 3, height, width) for RGB or BGR

Post-Processing NV12 to RGB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, convert NV12 to RGB after decoding using :py:func:`~spdl.io.nv12_to_rgb`:

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")

   # Decode to NV12
   nv12_buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0)
   )

   # Convert NV12 to RGB on GPU
   rgb_buffer = spdl.io.nv12_to_rgb(
       [nv12_buffer],  # Must be a list
       device_config=spdl.io.cuda_config(device_index=0)
   )

   tensor = spdl.io.to_torch(rgb_buffer)
   # tensor.shape: (num_frames, 3, height, width)

Memory Management
-----------------

Custom memory allocators can be used with hardware-accelerated decoding via the ``allocator`` parameter
in :py:func:`spdl.io.cuda_config`. This feature is part of ``CUDAConfig`` and works with all GPU operations in SPDL.

For details on custom allocators, see the :ref:`Custom Memory Allocators <custom-allocators>` section in :doc:`basic`.

Streaming Decoding
------------------

For long videos, use streaming to avoid loading everything into memory:

.. code-block:: python

   import spdl.io

   device_config = spdl.io.cuda_config(device_index=0)

   # Stream video in chunks of 32 frames
   streamer = spdl.io.streaming_load_video_nvdec(
       "long_video.mp4",
       device_config,
       num_frames=32,
       post_processing_params={
           "scale_width": 224,
           "scale_height": 224,
       }
   )

   for buffers in streamer:
       # buffers is a list of CUDABuffer objects (NV12 format)
       # Convert to RGB
       rgb_buffer = spdl.io.nv12_to_rgb(buffers, device_config=device_config)
       tensor = spdl.io.to_torch(rgb_buffer)

       # Process tensor...
       # tensor.shape: (batch_size, 3, 224, 224)

.. _streaming-with-nvdec:

Low-Level Streaming with NvDecDecoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the low-level :py:class:`spdl.io.NvDecDecoder` class for streaming decoding, you **must manually apply**
bitstream filtering for H.264 and HEVC videos.

For hardware-accelerated streaming video decoding with manual control, use :py:class:`spdl.io.NvDecDecoder`:

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("video.mp4")
   codec = demuxer.video_codec

   # Initialize NVDEC decoder
   cuda_config = spdl.io.cuda_config(device_index=0)
   decoder = spdl.io.nvdec_decoder()
   decoder.init(cuda_config, codec)

   # IMPORTANT: Bitstream filtering is REQUIRED when using NvDecDecoder
   # Apply bitstream filtering for H.264/HEVC
   bsf = None
   if codec.name in ("h264", "hevc"):
       bsf = spdl.io.BSF(codec, f"{codec.name}_mp4toannexb")

   # Stream and decode
   for packets in demuxer.streaming_demux_video(num_packets=10):
       # Apply bitstream filter if needed
       if bsf is not None:
           packets = bsf.filter(packets)

       buffers = decoder.decode(packets)  # Returns list[CUDABuffer]

       # Convert from NV12 to RGB
       for buffer in buffers:
           rgb_buffer = spdl.io.nv12_to_rgb(buffer)
           tensor = spdl.io.to_torch(rgb_buffer)  # CUDA tensor
           # Process tensor...

   # Flush bitstream filter and decoder
   if bsf is not None:
       packets = bsf.flush()
       if len(packets):
           buffers = decoder.decode(packets)
           # Process remaining buffers...

   buffers = decoder.flush()
   for buffer in buffers:
       rgb_buffer = spdl.io.nv12_to_rgb(buffer)
       # Process buffer...

.. note::

   :py:func:`spdl.io.streaming_load_video_nvdec` handles bitstream filtering automatically
   for H.264 and HEVC codecs.

Complete Example
----------------

Here's a complete example combining all features:

.. code-block:: python

   import spdl.io
   import torch

   def decode_video_gpu(
       video_path: str,
       device_index: int = 0,
       target_size: tuple[int, int] = (224, 224),
       crop: tuple[int, int, int, int] | None = None,
   ) -> torch.Tensor:
       """
       Decode video using hardware acceleration with optional preprocessing.

       Args:
           video_path: Path to video file
           device_index: CUDA device index
           target_size: (width, height) for resizing
           crop: (left, right, top, bottom) pixels to crop, or None

       Returns:
           PyTorch CUDA tensor with shape (N, 3, H, W)
       """
       # Setup device config with PyTorch allocator
       device_config = spdl.io.cuda_config(
           device_index=device_index,
           allocator=(
               torch.cuda.caching_allocator_alloc,
               torch.cuda.caching_allocator_delete
           )
       )

       # Demux video
       packets = spdl.io.demux_video(video_path)

       # Prepare decode options
       decode_options = {
           "width": target_size[0],
           "height": target_size[1],
           "pix_fmt": "rgb",
       }

       # Add crop if specified
       if crop is not None:
           left, right, top, bottom = crop
           decode_options.update({
               "crop_left": left,
               "crop_right": right,
               "crop_top": top,
               "crop_bottom": bottom,
           })

       # Decode using hardware acceleration (bitstream filter applied automatically)
       buffer = spdl.io.decode_packets_nvdec(
           packets,
           device_config=device_config,
           **decode_options
       )

       # Convert to PyTorch tensor (zero-copy)
       tensor = spdl.io.to_torch(buffer)

       return tensor

   # Usage
   video_tensor = decode_video_gpu(
       "video.mp4",
       device_index=0,
       target_size=(224, 224),
       crop=(100, 100, 50, 50)  # Crop before resize
   )

   print(f"Shape: {video_tensor.shape}")  # (N, 3, 224, 224)
   print(f"Device: {video_tensor.device}")  # cuda:0
   print(f"Dtype: {video_tensor.dtype}")  # torch.uint8

Performance Considerations
--------------------------

Hardware Limitations
~~~~~~~~~~~~~~~~~~~~

- **Decoder count**: GPUs have a limited number of hardware decoders (typically 3-5 per GPU)
- **Concurrent decoding**: Limit concurrent decoding operations to the number of available decoders
- **Resolution limits**: Check `NVIDIA's decoder support matrix <https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>`_

When to Use Hardware-Accelerated Decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hardware-accelerated decoding is beneficial when:**

- Decoded frames are used for GPU operations (inference, training)
- Processing high-resolution videos (4K, 8K)
- Decoding entire videos sequentially
- Memory bandwidth is a bottleneck

**CPU decoding may be better when:**

- Only sampling a few frames from long videos
- Need FFmpeg filter support (complex transformations)
- High concurrency is required (more CPU threads than GPU decoders)
- Decoded frames are used on CPU

Benchmarking
~~~~~~~~~~~~

Compare CPU vs hardware-accelerated decoding for your use case:

.. code-block:: python

   import time
   import spdl.io

   def benchmark_cpu_decode(video_path: str, width: int, height: int) -> float:
       t0 = time.time()
       buffer = spdl.io.load_video(
           video_path,
           filter_desc=spdl.io.get_video_filter_desc(
               scale_width=width,
               scale_height=height,
               pix_fmt="rgb24"
           )
       )
       elapsed = time.time() - t0
       num_frames = spdl.io.to_numpy(buffer).shape[0]
       return num_frames / elapsed

   def benchmark_hardware_decode(video_path: str, width: int, height: int) -> float:
       t0 = time.time()
       packets = spdl.io.demux_video(video_path)
       buffer = spdl.io.decode_packets_nvdec(
           packets,
           device_config=spdl.io.cuda_config(device_index=0),
           scale_width=width,
           scale_height=height,
           pix_fmt="rgb"
       )
       elapsed = time.time() - t0
       num_frames = spdl.io.to_torch(buffer).shape[0]
       return num_frames / elapsed

   # Run benchmarks
   video = "test_video.mp4"
   cpu_fps = benchmark_cpu_decode(video, 224, 224)
   hw_fps = benchmark_hardware_decode(video, 224, 224)

   print(f"CPU: {cpu_fps:.1f} FPS")
   print(f"Hardware: {hw_fps:.1f} FPS")
   print(f"Speedup: {hw_fps / cpu_fps:.2f}x")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue: "Odd width/height not supported"**

NVDEC requires even dimensions:

.. code-block:: python

   # Wrong
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=device_config,
       scale_width=225,  # Odd number!
       scale_height=225
   )

   # Correct
   buffer = spdl.io.decode_packets_nvdec(
       packets,
       device_config=device_config,
       scale_width=224,  # Even number
       scale_height=224
   )

**Issue: "Bitstream filter required"**

For H.264/HEVC in MP4 containers, ensure bitstream filtering is applied.
Use :py:func:`~spdl.io.decode_packets_nvdec` which applies it automatically.

**Issue: Out of memory**

For long videos, use streaming:

.. code-block:: python

   # Instead of loading entire video
   # buffer = spdl.io.decode_packets_nvdec(packets, ...)

   # Use streaming
   streamer = spdl.io.streaming_load_video_nvdec(
       video_path,
       device_config,
       num_frames=32  # Process in batches
   )

**Issue: "No decoder available"**

Too many concurrent decoding operations. Limit concurrency to the number of hardware decoders available.

Checking NVDEC Support
~~~~~~~~~~~~~~~~~~~~~~~

Verify NVDEC is available:

.. code-block:: python

   import spdl.io.utils

   # Check if SPDL was built with NVDEC support
   if spdl.io.utils.built_with_nvcodec():
       print("NVDEC support is available")
   else:
       print("NVDEC support not available")

   # Check FFmpeg configuration
   config = spdl.io.utils.get_ffmpeg_config()
   if "nvdec" in config.lower():
       print("FFmpeg has NVDEC support")

See Also
--------

- :doc:`basic` - High-level loading functions
- :doc:`decoding_overview` - Understanding the decoding pipeline
- :py:func:`spdl.io.decode_packets_nvdec` - Hardware-accelerated decoding function
- :py:func:`spdl.io.streaming_load_video_nvdec` - Streaming hardware-accelerated decoding
- :py:func:`spdl.io.nv12_to_rgb` - NV12 to RGB conversion
- :py:class:`spdl.io.BSF` - Bitstream filter class
- `NVIDIA Video Codec SDK <https://developer.nvidia.com/nvidia-video-codec-sdk>`_ - Official NVIDIA documentation
