Decoding Process Overview
==========================

This section explains how media data is processed internally in SPDL.
Understanding this multi-stage process will help you customize decoding effectively.

The Decoding Process
--------------------

Media decoding in SPDL follows a multi-stage process based on FFmpeg.
The decoding process consists of the following stages:

1. **Demuxing**: Extract packets from the source
2. **Decoding**: Decode packets into frames
3. **Filtering**: Apply transformations to frames (optional but commonly used)
4. **Buffer Conversion**: Merge frames into a contiguous array

The following diagram illustrates this process:

.. mermaid::

   flowchart TD
    subgraph Demuxing
      direction LR
      b(Byte String) --> |Demuxing| p1(Packet)
      p1 --> |Bitstream Filtering
      &#40Optional&#41| p2(Packet)
    end
    subgraph Decoding
      direction LR
      p3(Packet) --> |Decoding| f1(Frame)
      f1 --> |Filtering| f2(Frame)

    end
    subgraph c["Buffer Conversion"]
      subgraph ff[" "]
        f3(Frame)
        f4(Frame)
        f5(Frame)
      end
      ff  --> |Buffer Conversion| b2(Buffer)
    end
    Demuxing --> Decoding --> c

Low-Level Functions
~~~~~~~~~~~~~~~~~~~

While the high-level loading functions (:py:func:`spdl.io.load_audio`, :py:func:`spdl.io.load_video`, :py:func:`spdl.io.load_image`) provide a simple interface for common use cases, SPDL also exposes low-level functions that give you fine-grained control over each stage of the decoding process.

These low-level functions are:

1. **Demuxing functions**: Extract packets from the source

   - :py:func:`spdl.io.demux_audio`
   - :py:func:`spdl.io.demux_video`
   - :py:func:`spdl.io.demux_image`

2. **Decoding function**: Decode packets into frames

   - :py:func:`spdl.io.decode_packets`

3. **Buffer conversion function**: Convert frames into a contiguous buffer

   - :py:func:`spdl.io.convert_frames`

4. **Transfer function**: Transfer buffer to GPU (optional)

   - :py:func:`spdl.io.transfer_buffer`

The relationship between high-level and low-level functions can be expressed as:

.. code-block:: python

   # load_audio is equivalent to:
   packets = spdl.io.demux_audio(src, timestamp=timestamp, demux_config=demux_config)
   frames = spdl.io.decode_packets(packets, decode_config=decode_config, filter_desc=filter_desc)
   buffer = spdl.io.convert_frames(frames)
   # Optionally:
   buffer = spdl.io.transfer_buffer(buffer, device_config=device_config)

   # load_video is equivalent to:
   packets = spdl.io.demux_video(src, timestamp=timestamp, demux_config=demux_config)
   frames = spdl.io.decode_packets(packets, decode_config=decode_config, filter_desc=filter_desc)
   buffer = spdl.io.convert_frames(frames)
   # Optionally:
   buffer = spdl.io.transfer_buffer(buffer, device_config=device_config)

   # load_image is equivalent to:
   packets = spdl.io.demux_image(src, demux_config=demux_config)
   frames = spdl.io.decode_packets(packets, decode_config=decode_config, filter_desc=filter_desc)
   buffer = spdl.io.convert_frames(frames)
   # Optionally:
   buffer = spdl.io.transfer_buffer(buffer, device_config=device_config)

Using low-level functions is useful when you need to:

- Inspect or modify packets/frames between steps
- Apply custom processing logic
- Integrate with other libraries or custom decoders
- Debug decoding issues
- Reuse demuxed packets with different decoding parameters

Stage 1: Demuxing
------------------

`Demuxing <https://en.wikipedia.org/wiki/Demultiplexer_(media_file)>`_ (short for demultiplexing) is the process of splitting input data into smaller chunks called **packets**.

Media files typically contain multiple streams (e.g., one audio stream and one video stream) interleaved together.
Demuxing identifies the boundaries between these packets and extracts them one by one.

.. mermaid::

   block-beta
        columns 1
        b["0101010101100101...................................."]
        space
        block:demuxed
            p0[["Header"]]
            p1(["Audio 0"])
            p2["Video 0"]
            p3["Video 1"]
            p4["Video 2"]
            p5["Video 3"]
            p6(["Audio 1"])
            p7["Video 4"]
            p8["Video 5"]
        end
        b-- "demuxing" -->demuxed

**In SPDL:**

- :py:func:`spdl.io.demux_audio` - Demux audio packets
- :py:func:`spdl.io.demux_video` - Demux video packets
- :py:func:`spdl.io.demux_image` - Demux image packets

**Example:**

.. code-block:: python

   import spdl.io

   # Demux video packets from a file
   packets = spdl.io.demux_video("video.mp4")

   # Demux audio packets for a specific time window
   packets = spdl.io.demux_audio("audio.mp3", timestamp=(5.0, 10.0))

Stage 2: Decoding
------------------

Decoding is the process of decompressing packets to recover the original media data.
Media files are typically encoded (compressed) to reduce file size.
The decoder reverses this process to produce **frames**.

**Frames** contain the actual media samples:

- For audio: waveform samples
- For video/image: pixel data for each frame

**In SPDL:**

- :py:func:`spdl.io.decode_packets` - Decode packets into frames

**Example:**

.. code-block:: python

   import spdl.io

   # Demux and decode
   packets = spdl.io.demux_video("video.mp4")
   frames = spdl.io.decode_packets(packets)

Stage 3: Filtering
-------------------

Filtering is a versatile stage that can apply various transformations to frames.
This is where format conversion, resizing, cropping, and other preprocessing operations occur.

FFmpeg provides a rich set of filters through its `filter graph system <https://ffmpeg.org/ffmpeg-filters.html>`_.
In SPDL, filtering is controlled by the ``filter_desc`` parameter.

**Common filtering operations:**

- **Format conversion**: Convert pixel format (e.g., YUV to RGB) or audio sample format
- **Resizing**: Scale video/image to different dimensions
- **Cropping**: Extract a region of interest
- **Frame rate adjustment**: Change video frame rate
- **Trimming**: Remove frames outside a specified time window
- **Augmentation**: Apply random transformations for data augmentation

**In SPDL:**

The ``filter_desc`` parameter in :py:func:`spdl.io.decode_packets` controls filtering.
Helper functions generate filter descriptions:

- :py:func:`spdl.io.get_audio_filter_desc`
- :py:func:`spdl.io.get_video_filter_desc`
- :py:func:`spdl.io.get_filter_desc`

**Example:**

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")

   # Create a filter description
   filter_desc = spdl.io.get_video_filter_desc(
       scale_width=256,
       scale_height=256,
       pix_fmt="rgb24"
   )

   # Decode with filtering
   frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)

See :doc:`filtering` for detailed information about filter customization.

Stage 4: Buffer Conversion
---------------------------

Buffer conversion is the final stage where multiple frames are merged into a single contiguous memory region.
This creates an array-like buffer that can be easily converted to NumPy arrays, PyTorch tensors, or other array types.

**In SPDL:**

- :py:func:`spdl.io.convert_frames` - Convert frames to a buffer

**Example:**

.. code-block:: python

   import spdl.io

   packets = spdl.io.demux_video("video.mp4")
   frames = spdl.io.decode_packets(packets)
   buffer = spdl.io.convert_frames(frames)

   # Convert to PyTorch tensor
   tensor = spdl.io.to_torch(buffer)

Optional: Bitstream Filtering
------------------------------

Bitstream filtering is an optional stage that modifies packets before decoding.
This is less commonly used but necessary for certain scenarios.

**Common use cases:**

- Converting H.264/HEVC packets to Annex B format for hardware-accelerated decoding
- Extracting specific data from packets
- Modifying packet metadata

**In SPDL:**

- :py:class:`spdl.io.BSF` - Bitstream filter class
- :py:func:`spdl.io.apply_bsf` - Apply bitstream filtering to packets

**Example:**

.. code-block:: python

   import spdl.io

   # For hardware-accelerated video decoding, H.264 packets need conversion
   packets = spdl.io.demux_video("video.mp4")
   packets = spdl.io.apply_bsf(packets, "h264_mp4toannexb")

   # Now decode with hardware decoder
   frames = spdl.io.decode_packets_nvdec(
       packets,
       device_config=spdl.io.cuda_config(device_index=0)
   )

Complete Example
----------------

Here's a complete example showing all stages:

.. code-block:: python

   import spdl.io

   # Source file
   src = "video.mp4"

   # Stage 1: Demuxing
   packets = spdl.io.demux_video(src, timestamp=(0.0, 5.0))
   print(f"Demuxed {len(packets)} packets")

   # Stage 2 & 3: Decoding with Filtering
   filter_desc = spdl.io.get_video_filter_desc(
       scale_width=224,
       scale_height=224,
       pix_fmt="rgb24",
       num_frames=30
   )
   frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
   print(f"Decoded {len(frames)} frames")

   # Stage 4: Buffer Conversion
   buffer = spdl.io.convert_frames(frames)
   print(f"Buffer shape: {buffer.shape}")

   # Convert to array
   tensor = spdl.io.to_torch(buffer)
   print(f"Tensor shape: {tensor.shape}")  # (30, 224, 224, 3)

.. note::

   The high-level :py:func:`spdl.io.load_video` function performs all four stages (demuxing, decoding,
   filtering, and buffer conversion) automatically. The equivalent call would be:

   .. code-block:: python

      import spdl.io

      buffer = spdl.io.load_video(
          "video.mp4",
          timestamp=(0.0, 5.0),
          filter_desc=spdl.io.get_video_filter_desc(
              scale_width=224,
              scale_height=224,
              pix_fmt="rgb24",
              num_frames=30
          )
      )
      tensor = spdl.io.to_torch(buffer)

   This performs stages 1-4 internally and returns the final buffer ready for conversion.

Performance Considerations
--------------------------

While this multi-stage design provides great flexibility and features, it comes with overhead due to the complexity of FFmpeg-based processing.
For certain formats, specialized libraries or direct byte manipulation can be significantly faster.

For example, WAV format stores raw audio samples without compression. Instead of going through the full demux-decode-filter-convert process,
it's more efficient to simply reinterpret the incoming bytes directly as an array.

**In SPDL:** :py:func:`spdl.io.load_wav` is optimized for this use case and bypasses the full FFmpeg-based process for better performance when working with WAV files.

When choosing between :py:func:`spdl.io.load_audio` and :py:func:`spdl.io.load_wav`:

- Use :py:func:`spdl.io.load_wav` for WAV files when you need maximum performance and don't require complex preprocessing
- Use :py:func:`spdl.io.load_audio` when you need:

  - Support for multiple formats (MP3, FLAC, AAC, etc.)
  - Complex filtering and preprocessing
  - Timestamp-based seeking
  - Consistent API across different formats
