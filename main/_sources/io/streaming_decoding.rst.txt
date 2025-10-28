Streaming Decoding
==================

This section explains how to decode media data in a streaming fashion,
processing data chunk by chunk instead of loading everything into memory at once.

Overview
--------

The high-level loading functions (:py:func:`spdl.io.load_audio`, :py:func:`spdl.io.load_video`, :py:func:`spdl.io.load_image`)
load entire media files into memory before returning. While this is convenient for many use cases, it can be
inefficient or impractical for:

- **Long media files**: Files that are too large to fit in memory
- **Real-time processing**: Applications that need to start processing before the entire file is loaded
- **Memory-constrained environments**: Systems with limited available memory
- **Streaming applications**: Live video/audio processing pipelines

SPDL provides streaming APIs that allow you to process media data incrementally, chunk by chunk.

Basic Streaming Pattern
-----------------------

The basic streaming workflow follows these steps:

1. **Create a Demuxer**: Open the media source
2. **Create a Decoder**: Initialize the decoder for the codec
3. **Stream packets**: Use an iterator to get packets in chunks
4. **Decode incrementally**: Process each chunk of packets
5. **Flush**: Don't forget to flush remaining buffered frames

Here's a minimal example:

.. code-block:: python

   import spdl.io

   # Step 1: Create demuxer
   with spdl.io.Demuxer("video.mp4") as demuxer:
       # Step 2: Create decoder
       decoder = spdl.io.Decoder(demuxer.video_codec)

       # Step 3 & 4: Stream and decode packets
       for packets in demuxer.streaming_demux(demuxer.video_stream_index, num_packets=10):
           frames = decoder.decode(packets)
           if frames is not None:
               buffer = spdl.io.convert_frames(frames)
               # Process buffer here...
               tensor = spdl.io.to_torch(buffer)

       # Step 5: Flush remaining frames
       if (frames := decoder.flush()) is not None:
           buffer = spdl.io.convert_frames(frames)
           tensor = spdl.io.to_torch(buffer)

.. warning::

   Always call ``decoder.flush()`` at the end of streaming to retrieve any buffered frames.
   Many codecs buffer frames internally, and failing to flush will result in incomplete data.

Streaming Video
---------------

Use :py:meth:`spdl.io.Demuxer.streaming_demux` to stream video packets in chunks:

.. code-block:: python

   import spdl.io

   with spdl.io.Demuxer("video.mp4") as demuxer:
       decoder = spdl.io.Decoder(demuxer.video_codec)

       # Process 30 packets at a time
       for packets in demuxer.streaming_demux(demuxer.video_stream_index, num_packets=30):
           print(f"Processing {len(packets)} packets")
           frames = decoder.decode(packets)
           if frames is not None:
               buffer = spdl.io.convert_frames(frames)
               # Process frames...

       # Always flush
       if (frames := decoder.flush()) is not None:
           buffer = spdl.io.convert_frames(frames)

Streaming Audio
---------------

Audio streaming works similarly to video streaming:

.. code-block:: python

   import spdl.io

   with spdl.io.Demuxer("audio.mp3") as demuxer:
       decoder = spdl.io.Decoder(demuxer.audio_codec)

       # Stream audio packets
       for packets in demuxer.streaming_demux(demuxer.audio_stream_index, num_packets=20):
           frames = decoder.decode(packets)
           if frames is not None:
               buffer = spdl.io.convert_frames(frames)
               array = spdl.io.to_numpy(buffer)
               # Process audio chunk...

       # Flush
       if (frames := decoder.flush()) is not None:
           buffer = spdl.io.convert_frames(frames)

Multi-Stream Decoding
---------------------

For files containing multiple streams (e.g., audio and video), use :py:meth:`spdl.io.Demuxer.streaming_demux`
with stream indices.

You can chunk streams by duration (e.g., 5-second chunks) or by packet count (e.g., 50 packets at a time):

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("movie.mp4")

   video_index = demuxer.video_stream_index
   audio_index = demuxer.audio_stream_index

   audio_decoder = spdl.io.Decoder(demuxer.audio_codec)
   video_decoder = spdl.io.Decoder(demuxer.video_codec)

   # Process 5-second chunks (alternatively, use num_packets=50 for packet-based chunking)
   packet_stream = demuxer.streaming_demux(
       indices=[video_index, audio_index],
       duration=5.0
   )

   for packets in packet_stream:
       # Process audio if present in this chunk
       if audio_index in packets:
           frames = audio_decoder.decode(packets[audio_index])
           buffer = spdl.io.convert_frames(frames)
           # Process audio...

       # Process video if present in this chunk
       if video_index in packets:
           frames = video_decoder.decode(packets[video_index])
           buffer = spdl.io.convert_frames(frames)
           # Process video...

   # Flush both decoders
   if (frames := audio_decoder.flush()) is not None:
       buffer = spdl.io.convert_frames(frames)

   if (frames := video_decoder.flush()) is not None:
       buffer = spdl.io.convert_frames(frames)

.. note::

   When using ``duration`` parameter, each iteration yields approximately the specified duration
   worth of packets from each stream. Alternatively, use ``num_packets`` to chunk by packet count.

Streaming with Filtering
-------------------------

You can apply filters during streaming decoding by providing a ``filter_desc`` to the decoder.

This example shows how to apply a common preprocessing filter that resizes video frames to 256x256 and converts them to RGB format during streaming:

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("video.mp4")

   # Create filter description
   filter_desc = spdl.io.get_video_filter_desc(
       scale_width=256,
       scale_height=256,
       pix_fmt="rgb24"
   )

   # Create decoder with filter
   decoder = spdl.io.Decoder(demuxer.video_codec, filter_desc=filter_desc)

   # Stream with filtering applied
   for packets in demuxer.streaming_demux_video(num_packets=20):
       frames = decoder.decode(packets)  # Frames are already filtered
       buffer = spdl.io.convert_frames(frames)
       tensor = spdl.io.to_torch(buffer)
       # Process filtered tensor...

   # Flush
   if (frames := decoder.flush()) is not None:
       buffer = spdl.io.convert_frames(frames)

Advanced: Custom Filter Graph Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more complex filtering scenarios, use :py:class:`spdl.io.FilterGraph` to manually control the filter graph.

This example demonstrates adding a watermark overlay to video frames during streaming decoding. The overlay filter composites a logo image on top of the video:

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("video.mp4")
   codec = demuxer.video_codec

   # Create decoder without filter
   decoder = spdl.io.Decoder(codec, filter_desc=None)

   # Load watermark image
   watermark_buffer = spdl.io.load_image("logo.png")
   watermark_desc = spdl.io.get_vbuffer_desc(watermark_buffer)

   # Create filter graph with overlay
   # Format: buffer (main video) -> scale -> overlay <- buffer (watermark)
   filter_desc = (
       f"{spdl.io.get_vbuffer_desc(codec)}[main];"
       f"{watermark_desc}[logo];"
       "[main]scale=1280:720[scaled];"
       "[scaled][logo]overlay=W-w-10:H-h-10"  # Position logo at bottom-right with 10px margin
   )
   filter_graph = spdl.io.FilterGraph(filter_desc)

   # Add watermark to filter graph (only once)
   filter_graph.add_frames(watermark_buffer)

   # Stream with manual filter graph control
   for packets in demuxer.streaming_demux_video(num_packets=10):
       frames = decoder.decode(packets)
       if frames is not None:
           # Add video frames to filter graph
           filter_graph.add_frames(frames)

           # Get filtered frames with watermark applied
           output_frames = filter_graph.get_frames()
           if len(output_frames):
               buffer = spdl.io.convert_frames(output_frames)
               # Process watermarked buffer...

   # Flush decoder
   if (frames := decoder.flush()) is not None:
       filter_graph.add_frames(frames)

   # Flush filter graph
   filter_graph.flush()
   output_frames = filter_graph.get_frames()
   if len(output_frames):
       buffer = spdl.io.convert_frames(output_frames)
