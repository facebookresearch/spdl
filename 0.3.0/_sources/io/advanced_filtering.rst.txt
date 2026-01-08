Advanced Filter Graphs
======================

This section covers advanced filter graph usage, including complex graphs with multiple inputs and outputs,
and direct use of the :py:class:`spdl.io.FilterGraph` class for fine-grained control.

When to Use FilterGraph Directly
---------------------------------

The high-level functions (:py:func:`~spdl.io.load_audio`, :py:func:`~spdl.io.load_video`, etc.) and
:py:func:`~spdl.io.decode_packets` handle filtering automatically using simple linear filter chains.

Use :py:class:`~spdl.io.FilterGraph` directly when you need:

- **Multiple inputs**: Combining multiple media streams (e.g., side-by-side video comparison)
- **Multiple outputs**: Splitting one stream into multiple processed versions
- **Streaming processing**: Processing media in chunks without loading everything into memory
- **Complex filter topologies**: Non-linear filter graphs with branches and merges
- **Fine-grained control**: Manual control over when frames are added and retrieved

FilterGraph Basics
------------------

The :py:class:`~spdl.io.FilterGraph` class provides a low-level interface to FFmpeg's filter graph system.

Basic Workflow
~~~~~~~~~~~~~~

1. **Create** a filter graph with a filter description
2. **Add frames** to input nodes using :py:meth:`~spdl.io.FilterGraph.add_frames`
3. **Get frames** from output nodes using :py:meth:`~spdl.io.FilterGraph.get_frames`
4. **Flush** the graph when done using :py:meth:`~spdl.io.FilterGraph.flush`

Input and Output Nodes
~~~~~~~~~~~~~~~~~~~~~~

Unlike simple filter chains, complex filter graphs require explicit input and output nodes:

- **Input nodes**: ``buffer`` (video/image) or ``abuffer`` (audio)
- **Output nodes**: ``buffersink`` (video/image) or ``abuffersink`` (audio)

Helper functions construct these nodes:

- :py:func:`spdl.io.get_buffer_desc` - Create video/image input node
- :py:func:`spdl.io.get_abuffer_desc` - Create audio input node

Simple FilterGraph Example
---------------------------

Here's a basic example using FilterGraph for a simple passthrough:

.. code-block:: python

   import spdl.io

   # Load source
   demuxer = spdl.io.Demuxer("video.mp4")
   codec = demuxer.video_codec
   decoder = spdl.io.Decoder(codec, filter_desc=None)

   # Create filter graph with explicit input/output nodes
   buffer_desc = spdl.io.get_buffer_desc(codec)
   filter_desc = f"{buffer_desc},scale=256:256,format=rgb24,buffersink"

   filter_graph = spdl.io.FilterGraph(filter_desc)
   print(filter_graph)  # Print graph structure

   # Process frames
   buffers = []
   for packets in demuxer.streaming_demux(duration=1):
       frames = decoder.decode(packets)

       # Add frames to filter graph
       filter_graph.add_frames(frames)

       # Get filtered frames
       filtered_frames = filter_graph.get_frames()
       if filtered_frames is not None:
           buffer = spdl.io.convert_frames(filtered_frames)
           buffers.append(spdl.io.to_numpy(buffer))

   # Flush remaining frames
   if (frames := decoder.flush()) is not None:
       filter_graph.add_frames(frames)

   filter_graph.flush()

   if (frames := filter_graph.get_frames()) is not None:
       buffer = spdl.io.convert_frames(frames)
       buffers.append(spdl.io.to_numpy(buffer))

   # Combine all buffers
   result = np.concatenate(buffers)

Multiple Input Graphs
----------------------

Complex filter graphs can accept multiple input streams. This is useful for:

- Side-by-side video comparison
- Video overlays
- Audio mixing
- Picture-in-picture effects

Labeling Input Nodes
~~~~~~~~~~~~~~~~~~~~~

To use multiple inputs, label each input node with a unique name:

.. code-block:: python

   # Create two input nodes with labels
   buffer0 = spdl.io.get_buffer_desc(codec, label="in0")
   buffer1 = spdl.io.get_buffer_desc(codec, label="in1")

   # Construct filter graph that stacks videos vertically
   filter_desc = f"{buffer0} [in0];{buffer1} [in1],[in0] [in1] vstack,buffersink"

The syntax breakdown:

- ``buffer@in0=...`` - Input node named "in0"
- ``[in0]`` - Label for the output of this node
- ``[in0] [in1] vstack`` - Stack the two labeled streams
- ``buffersink`` - Output node

Side-by-Side Video Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("video.mp4")
   codec = demuxer.video_codec
   decoder = spdl.io.Decoder(codec, filter_desc=None)

   # Create filter graph with two inputs stacked vertically
   buf0 = spdl.io.get_buffer_desc(codec, label="in0")
   buf1 = spdl.io.get_buffer_desc(codec, label="in1")
   filter_desc = f"{buf0} [in0];{buf1} [in1],[in0] [in1] vstack,buffersink"

   filter_graph = spdl.io.FilterGraph(filter_desc)

   buffers = []
   for packets in demuxer.streaming_demux(duration=1):
       frames = decoder.decode(packets)

       # Add the same frames to both inputs (creates duplicate)
       filter_graph.add_frames(frames.clone(), key="buffer@in0")
       filter_graph.add_frames(frames, key="buffer@in1")

       # Get stacked output
       filtered_frames = filter_graph.get_frames()
       if filtered_frames is not None:
           buffer = spdl.io.convert_frames(filtered_frames)
           buffers.append(spdl.io.to_numpy(buffer))

   # Flush
   if (frames := decoder.flush()) is not None:
       filter_graph.add_frames(frames.clone(), key="buffer@in0")
       filter_graph.add_frames(frames, key="buffer@in1")

   filter_graph.flush()

   if (frames := filter_graph.get_frames()) is not None:
       buffer = spdl.io.convert_frames(frames)
       buffers.append(spdl.io.to_numpy(buffer))

   result = np.concatenate(buffers)
   # result now contains frames stacked vertically (double height)

Common Multi-Input Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Horizontal stack (side-by-side):**

.. code-block:: python

   filter_desc = f"{buf0} [in0];{buf1} [in1],[in0] [in1] hstack,buffersink"

.. image:: ../../_static/data/io_multi_input_hstack.png

**Vertical stack (top-bottom):**

.. code-block:: python

   filter_desc = f"{buf0} [in0];{buf1} [in1],[in0] [in1] vstack,buffersink"

.. image:: ../../_static/data/io_multi_input_vstack.png

**Overlay (picture-in-picture):**

.. code-block:: python

   # Overlay second video on top of first at position (10, 10)
   filter_desc = ";".join(
       [
           f"{buf0} [main]",
           f"{buf1} [pip]",
           "[pip] scale=96:72 [pip_scaled]",
           "[main][pip_scaled] overlay=x=W-w-10:y=H-h-10 [overlaid]",
           "[overlaid] format=rgb24,buffersink",
       ]
   )

.. image:: ../../_static/data/io_overlay_pip.png


**Blend:**

.. code-block:: python

   # Blend two videos with 50% opacity each
   filter_desc = f"{buf0} [in0];{buf1} [in1],[in0] [in1] blend=all_mode=average,buffersink"

Multiple Output Graphs
-----------------------

Filter graphs can produce multiple output streams. This is useful for:

- Generating multiple resolutions simultaneously
- Creating different augmented versions
- Extracting different features from the same source

Labeling Output Nodes
~~~~~~~~~~~~~~~~~~~~~~

To use multiple outputs, label each output node:

.. code-block:: python

   filter_desc = ";".join([
       f"{spdl.io.get_buffer_desc(codec)} [in]",
       "[in] split [out0][out1]",
       "[out0] buffersink@out0",
       "[out1] buffersink@out1",
   ])

The syntax breakdown:

- ``[in] split [out0][out1]`` - Split input into two streams
- ``buffersink@out0`` - Output node named "out0"
- ``buffersink@out1`` - Output node named "out1"

Multi-Resolution Output Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("video.mp4")
   codec = demuxer.video_codec
   decoder = spdl.io.Decoder(codec, filter_desc=None)

   # Create filter graph with two outputs at different resolutions
   filter_desc = ";".join([
       f"{spdl.io.get_buffer_desc(codec)} [in]",
       "[in] split [tmp0][tmp1]",
       "[tmp0] scale=256:256 [out0]",
       "[tmp1] scale=128:128 [out1]",
       "[out0] buffersink@out0",
       "[out1] buffersink@out1",
   ])

   filter_graph = spdl.io.FilterGraph(filter_desc)

   buffers_256, buffers_128 = [], []

   for packets in demuxer.streaming_demux(duration=1):
       frames = decoder.decode(packets)
       filter_graph.add_frames(frames)

       # Get frames from first output (256x256)
       frames_256 = filter_graph.get_frames(key="buffersink@out0")
       if frames_256 is not None:
           buffer = spdl.io.convert_frames(frames_256)
           buffers_256.append(spdl.io.to_numpy(buffer))

       # Get frames from second output (128x128)
       frames_128 = filter_graph.get_frames(key="buffersink@out1")
       if frames_128 is not None:
           buffer = spdl.io.convert_frames(frames_128)
           buffers_128.append(spdl.io.to_numpy(buffer))

   # Flush
   if (frames := decoder.flush()) is not None:
       filter_graph.add_frames(frames)

   filter_graph.flush()

   if (frames := filter_graph.get_frames(key="buffersink@out0")) is not None:
       buffer = spdl.io.convert_frames(frames)
       buffers_256.append(spdl.io.to_numpy(buffer))

   if (frames := filter_graph.get_frames(key="buffersink@out1")) is not None:
       buffer = spdl.io.convert_frames(frames)
       buffers_128.append(spdl.io.to_numpy(buffer))

   result_256 = np.concatenate(buffers_256)  # Shape: (N, 256, 256, C)
   result_128 = np.concatenate(buffers_128)  # Shape: (N, 128, 128, C)


.. image:: ../../_static/data/io_multi_output.png

Common Multi-Output Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Different augmentations:**

.. code-block:: python

   filter_desc = ";".join([
       f"{spdl.io.get_buffer_desc(codec)} [in]",
       "[in] split [tmp0][tmp1]",
       "[tmp0] hflip [out0]",
       "[tmp1] vflip [out1]",
       "[out0] buffersink@out0",
       "[out1] buffersink@out1",
   ])

.. image:: ../../_static/data/io_multi_input_different_processing.png

**Different color spaces:**

.. code-block:: python

   filter_desc = ";".join([
       f"{spdl.io.get_buffer_desc(codec)} [in]",
       "[in] split [tmp0][tmp1]",
       "[tmp0] format=rgb24 [out0]",
       "[tmp1] format=gray [out1]",
       "[out0] buffersink@out0",
       "[out1] buffersink@out1",
   ])

Multimedia Filters
------------------

FFmpeg provides `multimedia filters <https://ffmpeg.org/ffmpeg-filters.html#Multimedia-Filters>`_
that can convert between audio and video streams.

Audio to Video Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``showwaves`` filter converts audio waveforms to video:

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("audio.mp3")
   codec = demuxer.audio_codec
   decoder = spdl.io.Decoder(codec, filter_desc=None)

   # Create filter graph: audio input -> video output
   abuffer_desc = spdl.io.get_abuffer_desc(codec)
   filter_desc = f"{abuffer_desc},showwaves,buffersink"

   filter_graph = spdl.io.FilterGraph(filter_desc)

   video_buffers = []
   for packets in demuxer.streaming_demux(duration=1):
       audio_frames = decoder.decode(packets)

       # Add audio frames
       filter_graph.add_frames(audio_frames)

       # Get video frames
       video_frames = filter_graph.get_frames()
       if video_frames is not None:
           buffer = spdl.io.convert_frames(video_frames)
           video_buffers.append(spdl.io.to_numpy(buffer))

   # Flush
   if (frames := decoder.flush()) is not None:
       filter_graph.add_frames(frames)

   filter_graph.flush()

   if (frames := filter_graph.get_frames()) is not None:
       buffer = spdl.io.convert_frames(frames)
       video_buffers.append(spdl.io.to_numpy(buffer))

   video_result = np.concatenate(video_buffers)
   # video_result contains visualization of audio waveform

.. image:: ../../_static/data/io_audio_to_video_showwaves.png

Other Multimedia Filters
~~~~~~~~~~~~~~~~~~~~~~~~~

**showspectrum** - Audio spectrum visualization:

.. code-block:: python

   filter_desc = f"{abuffer_desc},showspectrum,buffersink"

.. image:: ../../_static/data/io_audio_to_video_showspectrum.png

**showfreqs** - Frequency visualization:

.. code-block:: python

   filter_desc = f"{abuffer_desc},showfreqs,buffersink"

**avectorscope** - Stereo audio vectorscope:

.. code-block:: python

   filter_desc = f"{abuffer_desc},avectorscope,buffersink"

Complex Graph Examples
-----------------------

Example 1: Multi-Input with Different Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process two video streams differently and combine them:

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("video.mp4")
   codec = demuxer.video_codec
   decoder = spdl.io.Decoder(codec, filter_desc=None)

   # Create complex filter: apply different effects to each input
   buf0 = spdl.io.get_buffer_desc(codec, label="in0")
   buf1 = spdl.io.get_buffer_desc(codec, label="in1")

   filter_desc = ";".join([
       f"{buf0} [in0]",
       f"{buf1} [in1]",
       "[in0] hflip,scale=320:240 [left]",
       "[in1] vflip,scale=320:240 [right]",
       "[left][right] hstack",
       "buffersink"
   ])

   filter_graph = spdl.io.FilterGraph(filter_desc)

   buffers = []
   for packets in demuxer.streaming_demux(duration=1):
       frames = decoder.decode(packets)

       filter_graph.add_frames(frames.clone(), key="buffer@in0")
       filter_graph.add_frames(frames, key="buffer@in1")

       filtered_frames = filter_graph.get_frames()
       if filtered_frames is not None:
           buffer = spdl.io.convert_frames(filtered_frames)
           buffers.append(spdl.io.to_numpy(buffer))

   # Flush
   if (frames := decoder.flush()) is not None:
       filter_graph.add_frames(frames.clone(), key="buffer@in0")
       filter_graph.add_frames(frames, key="buffer@in1")

   filter_graph.flush()

   if (frames := filter_graph.get_frames()) is not None:
       buffer = spdl.io.convert_frames(frames)
       buffers.append(spdl.io.to_numpy(buffer))

   result = np.concatenate(buffers)
   # Result: horizontally stacked video with left side flipped horizontally,
   # right side flipped vertically

Example 2: Multi-Output with Branching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a thumbnail grid from a single video:

.. code-block:: python

   import spdl.io

   demuxer = spdl.io.Demuxer("video.mp4")
   codec = demuxer.video_codec
   decoder = spdl.io.Decoder(codec, filter_desc=None)

   # Create 2x2 grid of thumbnails with different effects
   filter_desc = ";".join([
       f"{spdl.io.get_buffer_desc(codec)} [in]",
       "[in] split=4 [tmp0][tmp1][tmp2][tmp3]",
       "[tmp0] scale=160:120 [tl]",
       "[tmp1] scale=160:120,hflip [tr]",
       "[tmp2] scale=160:120,vflip [bl]",
       "[tmp3] scale=160:120,hflip,vflip [br]",
       "[tl][tr] hstack [top]",
       "[bl][br] hstack [bottom]",
       "[top][bottom] vstack",
       "buffersink"
   ])

   filter_graph = spdl.io.FilterGraph(filter_desc)

   buffers = []
   for packets in demuxer.streaming_demux(duration=1):
       frames = decoder.decode(packets)
       filter_graph.add_frames(frames)

       filtered_frames = filter_graph.get_frames()
       if filtered_frames is not None:
           buffer = spdl.io.convert_frames(filtered_frames)
           buffers.append(spdl.io.to_numpy(buffer))

   # Flush
   if (frames := decoder.flush()) is not None:
       filter_graph.add_frames(frames)

   filter_graph.flush()

   if (frames := filter_graph.get_frames()) is not None:
       buffer = spdl.io.convert_frames(frames)
       buffers.append(spdl.io.to_numpy(buffer))

   result = np.concatenate(buffers)
   # Result: 320x240 video showing 2x2 grid of the same video with different flips

.. image:: ../../_static/data/io_thumbnail_grid_2x2.png

Debugging Filter Graphs
------------------------

Visualizing Graph Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`~spdl.io.FilterGraph` class provides a string representation showing the graph structure:

.. code-block:: python

   filter_graph = spdl.io.FilterGraph(filter_desc)
   print(filter_graph)

This outputs a text diagram showing:

- All nodes in the graph
- Connections between nodes
- Data formats at each connection

Example output:

.. code-block:: text

   +-----------------+
   | Parsed_buffer_0 |default--[320x240 1:1 yuv420p]--Parsed_scale_1:default
   |    (buffer)     |
   +-----------------+

                                                          +-----------------+
   Parsed_buffer_0:default--[320x240 1:1 yuv420p]--default| Parsed_scale_1  |default--[256x256 1:1 yuv420p]--Parsed_buffersink_2:default
                                                          |    (scale)      |
                                                          +-----------------+
