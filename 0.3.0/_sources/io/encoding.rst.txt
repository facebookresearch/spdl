Media Encoding
==============

This section explains how to encode media data (audio and video) using SPDL.
Encoding is the process of converting array data into compressed media formats like MP4, WAV, PNG, etc.

.. note::

   This section covers audio and video encoding. For image encoding, use the video encoding
   workflow with a single frame. Providing array data of one frame with proper configuration
   will save the data as an image file (e.g., PNG, JPEG). See :py:func:`spdl.io.save_image`
   for a convenient wrapper.

The Encoding Process
---------------------

Media encoding in SPDL follows a multi-stage process:

1. **Create Reference Frames**: Reinterpret array data as frame objects without copying
2. **Filter Frames** (Optional): Apply transformations like scaling or color correction
3. **Encode Frames**: Compress frames into packets
4. **Mux Packets**: Write packets to an output file

The following diagram illustrates this process:

.. mermaid::

   flowchart LR
    a[Array Data] --> |create_reference_frame| f[Frame]
    f --> |Optional: FilterGraph| f2[Filtered Frame]
    f2 -.-> f
    f --> |Encoder| p[Packet]
    p --> |Muxer| file[Output File]

Creating Reference Frames
--------------------------

The first step in encoding is to create reference frames from your array data. This process reinterprets
the contiguous array data into a format compatible with the encoding system **without copying the data**.
The resulting frame objects hold metadata (format, dimensions, timestamps) and reference the original
array's memory, making this operation very efficient.

SPDL provides two functions for creating reference frames:

Creating Audio Frames
~~~~~~~~~~~~~~~~~~~~~

Use :py:func:`spdl.io.create_reference_audio_frame` to create audio frames from array data:

.. code-block:: python

   import numpy as np
   import spdl.io

   sample_rate = 44100
   num_channels = 2
   duration = 3

   # Create audio data (3 seconds of stereo audio)
   shape = (sample_rate * duration, num_channels)
   audio_data = np.random.randint(-32768, 32767, size=shape, dtype=np.int16)

   # Create audio frame
   frames = spdl.io.create_reference_audio_frame(
       array=audio_data,
       sample_fmt="s16",      # 16-bit signed integer
       sample_rate=sample_rate,
       pts=0,                 # Presentation timestamp
   )

For detailed parameter descriptions, see :py:func:`spdl.io.create_reference_audio_frame`.

Creating Video Frames
~~~~~~~~~~~~~~~~~~~~~

Use :py:func:`spdl.io.create_reference_video_frame` to create video frames from array data:

.. code-block:: python

   import numpy as np
   import spdl.io

   height, width = 240, 320
   frame_rate = (30000, 1001)  # ~29.97 fps
   num_frames = 90

   # Create video data (90 frames of RGB video)
   shape = (num_frames, height, width, 3)
   video_data = np.random.randint(0, 255, size=shape, dtype=np.uint8)

   # Create video frames
   frames = spdl.io.create_reference_video_frame(
       array=video_data,
       pix_fmt="rgb24",
       frame_rate=frame_rate,
       pts=0,
   )

For detailed parameter descriptions, see :py:func:`spdl.io.create_reference_video_frame`.

Using Encoders
--------------

Encoders compress frame data into packets using codecs like H.264, AAC, or PCM. The encoding process
applies compression algorithms to reduce file size while maintaining quality. Encoders are created
through a :py:class:`spdl.io.Muxer` object and configured with parameters like bit rate, quality,
and codec-specific settings.

Audio Encoding
~~~~~~~~~~~~~~

Here's a complete example of encoding audio to a WAV file:

.. code-block:: python

   import numpy as np
   import spdl.io

   sample_rate = 44100
   duration = 3
   num_channels = 2

   # Create audio data
   shape = (sample_rate * duration, num_channels)
   audio_data = np.random.randint(-32768, 32767, size=shape, dtype=np.int16)

   # Create muxer and encoder
   muxer = spdl.io.Muxer("output.wav")
   encoder = muxer.add_encode_stream(
       config=spdl.io.audio_encode_config(
           num_channels=num_channels,
           sample_fmt="s16",
           sample_rate=sample_rate,
       ),
       encoder="pcm_s16le",  # Optional: specify encoder
   )

   # Encode and write
   with muxer.open():
       # Create frames
       frames = spdl.io.create_reference_audio_frame(
           array=audio_data,
           sample_fmt="s16",
           sample_rate=sample_rate,
           pts=0,
       )

       # Encode frames
       if (packets := encoder.encode(frames)) is not None:
           muxer.write(0, packets)

       # Flush encoder
       if (packets := encoder.flush()) is not None:
           muxer.write(0, packets)

Video Encoding
~~~~~~~~~~~~~~

Here's a complete example of encoding video to an MP4 file:

.. code-block:: python

   import numpy as np
   import spdl.io

   height, width = 240, 320
   frame_rate = (30000, 1001)
   duration = 3
   batch_size = 32

   num_frames = int(frame_rate[0] / frame_rate[1] * duration)
   shape = (num_frames, height, width, 3)
   video_data = np.random.randint(0, 255, size=shape, dtype=np.uint8)

   # Create muxer and encoder
   muxer = spdl.io.Muxer("output.mp4")
   encoder = muxer.add_encode_stream(
       config=spdl.io.video_encode_config(
           height=height,
           width=width,
           pix_fmt="rgb24",
           frame_rate=frame_rate,
       ),
   )

   # Encode and write in batches
   with muxer.open():
       for start in range(0, num_frames, batch_size):
           # Create frames for this batch
           frames = spdl.io.create_reference_video_frame(
               array=video_data[start:start + batch_size, ...],
               pix_fmt="rgb24",
               frame_rate=frame_rate,
               pts=start,
           )

           # Encode frames
           if (packets := encoder.encode(frames)) is not None:
               muxer.write(0, packets)

       # Flush encoder
       if (packets := encoder.flush()) is not None:
           muxer.write(0, packets)

Using the Muxer
---------------

The muxer is the final stage that writes encoded packets to an output file. It handles the container
format (e.g., MP4, WAV, MKV) and ensures packets are properly interleaved and timestamped. The muxer
can write multiple streams (audio, video, subtitles) into a single file.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   # Create muxer for output file
   muxer = spdl.io.Muxer("output.mp4")

   # Add encoding stream(s)
   encoder = muxer.add_encode_stream(
       config=spdl.io.video_encode_config(
           height=240,
           width=320,
           pix_fmt="rgb24",
           frame_rate=(30, 1),
       ),
   )

   # Open muxer and write data
   with muxer.open():
       # ... encode and write packets
       muxer.write(0, packets)

The muxer automatically flushes and closes when used as a context manager.

Multiple Streams (Audio + Video)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can write both audio and video streams to a single file:

.. code-block:: python

   import numpy as np
   import spdl.io

   # Create audio and video data
   audio_data = np.random.randint(-32768, 32767, size=(44100 * 3, 2), dtype=np.int16)
   video_data = np.random.randint(0, 255, size=(90, 240, 320, 3), dtype=np.uint8)

   # Create muxer with both audio and video streams
   muxer = spdl.io.Muxer("output.mp4")
   audio_encoder = muxer.add_encode_stream(
       config=spdl.io.audio_encode_config(
           num_channels=2, sample_rate=44100, sample_fmt="s16"
       ),
       encoder="aac",
   )
   video_encoder = muxer.add_encode_stream(
       config=spdl.io.video_encode_config(
           height=240, width=320, pix_fmt="rgb24", frame_rate=(30, 1)
       ),
   )

   with muxer.open():
       # Write audio to stream 0
       audio_frames = spdl.io.create_reference_audio_frame(
           array=audio_data, sample_fmt="s16", sample_rate=44100, pts=0
       )
       if (packets := audio_encoder.encode(audio_frames)) is not None:
           muxer.write(0, packets)

       # Write video to stream 1
       video_frames = spdl.io.create_reference_video_frame(
           array=video_data, pix_fmt="rgb24", frame_rate=(30, 1), pts=0
       )
       if (packets := video_encoder.encode(video_frames)) is not None:
           muxer.write(1, packets)

       # Flush both encoders
       if (packets := audio_encoder.flush()) is not None:
           muxer.write(0, packets)
       if (packets := video_encoder.flush()) is not None:
           muxer.write(1, packets)

Remuxing (Copying Streams)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also remux (copy) streams without re-encoding:

.. code-block:: python

   import spdl.io

   # Open source file
   demuxer = spdl.io.Demuxer("input.mp4")

   # Create output muxer
   muxer = spdl.io.Muxer("output.mp4")
   muxer.add_remux_stream(demuxer.video_codec)

   # Copy packets
   with muxer.open():
       for packets in demuxer.streaming_demux(duration=1):
           muxer.write(0, packets)

Customizing Encoders
--------------------

SPDL provides configuration functions to customize encoding behavior.

Video Encode Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :py:func:`spdl.io.video_encode_config` to customize video encoding:

.. code-block:: python

   import spdl.io

   config = spdl.io.video_encode_config(
       height=1080,
       width=1920,
       pix_fmt="yuv420p",
       frame_rate=(30, 1),
       bit_rate=5000000,           # 5 Mbps
       gop_size=30,                # GOP size
       max_b_frames=2,             # Max B-frames
       compression_level=5,        # Compression level
       colorspace="bt709",         # Color space
       color_primaries="bt709",    # Color primaries
       color_trc="bt709",          # Transfer characteristics
   )

   muxer = spdl.io.Muxer("output.mp4")
   encoder = muxer.add_encode_stream(config=config)

For detailed parameter descriptions, see :py:func:`spdl.io.video_encode_config`.

Audio Encode Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :py:func:`spdl.io.audio_encode_config` to customize audio encoding:

.. code-block:: python

   import spdl.io

   config = spdl.io.audio_encode_config(
       num_channels=2,
       sample_rate=48000,
       sample_fmt="fltp",
       bit_rate=192000,        # 192 kbps
       compression_level=5,
   )

   muxer = spdl.io.Muxer("output.aac")
   encoder = muxer.add_encode_stream(config=config)

For detailed parameter descriptions, see :py:func:`spdl.io.audio_encode_config`.

Applying Filters to Reference Frames
-------------------------------------

You can apply filters to reference frames before encoding using :py:class:`spdl.io.FilterGraph`.
This is useful for preprocessing (e.g., scaling, color correction, audio normalization).

.. code-block:: python

   import numpy as np
   import spdl.io

   height, width = 240, 320
   frame_rate = (30, 1)
   video_data = np.random.randint(0, 255, size=(90, height, width, 3), dtype=np.uint8)

   # Create reference frames
   frames = spdl.io.create_reference_video_frame(
       array=video_data, pix_fmt="rgb24", frame_rate=frame_rate, pts=0
   )

   # Apply scaling filter
   filter_desc = (
       f"buffer=width={width}:height={height}:pix_fmt=rgb24:"
       f"time_base={frame_rate[1]}/{frame_rate[0]}:sar=1/1,"
       "scale=640:480,buffersink"
   )
   filter_graph = spdl.io.FilterGraph(filter_desc)
   filter_graph.add_frames(frames)
   filter_graph.flush()
   filtered_frames = filter_graph.get_frames()

   # Encode filtered frames
   muxer = spdl.io.Muxer("output.mp4")
   encoder = muxer.add_encode_stream(
       config=spdl.io.video_encode_config(
           height=480, width=640, pix_fmt="rgb24", frame_rate=frame_rate
       )
   )
   with muxer.open():
       if (packets := encoder.encode(filtered_frames)) is not None:
           muxer.write(0, packets)
       if (packets := encoder.flush()) is not None:
           muxer.write(0, packets)

For more details on filtering, including helper functions like :py:func:`spdl.io.get_buffer_desc`
and :py:func:`spdl.io.get_abuffer_desc`, see :doc:`filtering` and :doc:`advanced_filtering`.

See Also
--------

- :doc:`filtering` - More details on using FilterGraph
- :doc:`advanced_filtering` - Advanced filtering techniques
- :doc:`decoding_overview` - Understanding the decoding process
- :py:class:`spdl.io.Muxer` - Muxer API reference
- :py:class:`spdl.io.AudioEncoder` - Audio encoder API reference
- :py:class:`spdl.io.VideoEncoder` - Video encoder API reference
- :py:func:`spdl.io.create_reference_audio_frame` - Create audio frames
- :py:func:`spdl.io.create_reference_video_frame` - Create video frames
- :py:func:`spdl.io.audio_encode_config` - Audio encoding configuration
- :py:func:`spdl.io.video_encode_config` - Video encoding configuration
