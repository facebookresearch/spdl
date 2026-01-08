Packets and Frames
==================

This page documents the conceptual base classes ``Packets`` and ``Frames`` that are common to all media types in SPDL.
While these base classes don't exist as concrete Python classes, their behavior and properties are inherited by all specific media type implementations.

Packets
-------

**Packets objects** represent the result of demuxing operations.

Internally, they hold a series of FFmpeg's ``AVPacket`` objects, which contain compressed media data.
Decode functions receive ``Packets`` objects and generate audio samples and visual frames.

The concrete implementations are:

- :py:class:`~spdl.io.AudioPackets` - Contains audio samples
- :py:class:`~spdl.io.VideoPackets` - Contains video frames
- :py:class:`~spdl.io.ImagePackets` - Contains an image frame

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   >>> src = "sample.mp4"
   >>> windows = [
   ...     (3, 5),
   ...     (7, 9),
   ...     (13, 15),
   ... ]
   >>>
   >>> demuxer = spdl.io.Demuxer(src)
   >>> for window in windows:
   ...     packets = demuxer.demux_video(window)
   ...     frames = decode_packets(packets)
   ...
   >>>

Lifetime of Packets Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   When packets objects are passed to a decode function, its ownership is
   also passed to the function. Therefore, accessing the packets object after
   it is passed to decode function will cause an error.

   .. code-block:: python

      >>> # Demux an image
      >>> packets = spdl.io.demux_image("foo.png")
      >>> packets  # this works.
      ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>
      >>>
      >>> # Decode the packets
      >>> frames = spdl.io.decode_packets(packets)
      >>> frames
      ImageFrames<pixel_format="rgb24", num_planes=1, width=320, height=240>
      >>>
      >>> # The packets object is no longer valid.
      >>> packets
      RuntimeWarning: nanobind: attempted to access an uninitialized instance of type 'spdl.lib._spdl_ffmpeg6.ImagePackets'!

      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      TypeError: __repr__(): incompatible function arguments. The following argument types are supported:
          1. __repr__(self) -> str

      Invoked with types: spdl.lib._spdl_ffmpeg6.ImagePackets
      >>>

   This design choice was made for two reasons;

   1. Decoding is performed in background thread, potentially long after
      since the job was created due to other decoding jobs. To ensure the
      existence of the packets, the decoding function should take the
      ownership of the packets, instead of a reference.

   2. An alternative approach to 1 is to share the ownership, however, in this
      approach, it is not certain when the Python variable holding the shared
      ownership of the Packets object is deleted. Python might keep the
      reference for a long time, or the garbage collection might kick-in when
      the execution is in critical code path. By passing the ownership to
      decoding function, the ``Packets`` object resource is also released in
      background.

Cloning Packets
~~~~~~~~~~~~~~~

To decode packets multiple times, use the ``clone`` method.

.. code-block:: python

   >>> packets = spdl.io.demux_image("foo.png")
   >>> # Decode the cloned packets
   >>> packets2 = packets.clone()
   >>> packets2
   ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>
   >>> frames = spdl.io.decode_packets(packets)
   >>>
   >>> # The original packets object is still valid
   >>> packets
   ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>

.. note::

   The underlying FFmpeg implementation employs reference counting for
   ``AVPacket`` object.

   Therefore, even though the method is called ``clone``, this method
   does not copy the frame data.

Frames
------

**Frames objects** represent the result of decoding and filtering operations.

Internally, they hold a series of FFmpeg's ``AVFrame`` objects, which contain uncompressed media data.
Note that these are not contiguous memory objects.

The concrete implementations are:

- :py:class:`~spdl.io.AudioFrames` - Contains audio samples
- :py:class:`~spdl.io.VideoFrames` - Contains video frames
- :py:class:`~spdl.io.ImageFrames` - Contains an image frame

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   >>> packets = spdl.io.demux_video("sample.mp4")
   >>> frames = spdl.io.decode_packets(packets)
   >>> frames
   VideoFrames<num_frames=120, pixel_format="rgb24", num_planes=1, width=640, height=480, timestamp=[0.000, 4.000]>

Cloning Frames
~~~~~~~~~~~~~~

To convert frames to buffer multiple times, use the ``clone`` method.

.. code-block:: python

   >>> frames = spdl.io.decode_packets(packets)
   >>> # Clone the frames
   >>> frames2 = frames.clone()
   >>> buffer1 = spdl.io.convert_frames(frames)
   >>> buffer2 = spdl.io.convert_frames(frames2)

.. note::

   The underlying FFmpeg implementation employs reference counting for
   ``AVFrame`` object.

   Therefore, even though the method is called ``clone``, this method
   does not copy the frame data.
