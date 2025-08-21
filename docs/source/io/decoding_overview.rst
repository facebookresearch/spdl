How media is processed
======================

In the previous section, we looked at how to load multi-media data into array format,
and customize its output format.
The output format is customized after the media is decoded.
In the next section, we will look into ways to customize the decoding process.

Before looking into customization APIs,
we look at the processes that media data goes through,
so that we have a good understanding of how and what to customize.

Thes processes are generic and not specific to SPDL.
The underlying implementation is provided by FFmpeg,
which is the industry-standard tool for processing media in
streaming fashion.

Decoding media comprises of two mondatory steps (demuxing and decoding) and
optional filtering steps.
Frames are allocated in the heap when decoding.
They are combined into one contiguous memory region to form an array.
We call this process the buffer conversion.

The following diagram illustrates this.

.. mermaid::

   flowchart TD
    subgraph Demuxing
      direction LR
      b(Bite String) --> |Demuxing| p1(Packet)
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


Demuxing
~~~~~~~~
    
The `demuxing <https://en.wikipedia.org/wiki/Demultiplexer_(media_file)>`_
(short for demultiplexing) is a process to split the input data into smaller chunks.

For example, a video has multiple media streams (typically one audio and one video) and
they are storead as a series of data called "packet".

The demuxing is a process to find data boundary and extract these packets one-by-one.

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

Decoding
~~~~~~~~

Multi-media files are usually encoded to reduce the file size.
The decoding is a process to recover the media from the encoded data.

The decoded data are called frames, and they contain waveform samples (audio)
or image frame samples (image/video).

Buffer Conversion
~~~~~~~~~~~~~~~~~

The buffer conversion is the step to merge multiple frames into one
contiguous memory so that we can handle them as an array data.

Bitstream Filtering
~~~~~~~~~~~~~~~~~~~

The bitstream filtering is a process to modify packets.
You can refer to
`FFmpeg Bitstream Filters Documentation <https://ffmpeg.org/ffmpeg-bitstream-filters.html>`_
for available operations.

In SPDL, the most relevant operations are ``h264_mp4toannexb`` and
``hevc_mp4toannexb``, which are necessary when using GPU video decoding.
See :ref:`gpu-video-decoder`

Filtering
~~~~~~~~~

The frame filtering is a versatile process.
It can apply many different operations.

Please refer to `FFmpeg Filters Documentation <https://ffmpeg.org/ffmpeg-filters.html>`_
for available filters.

By default it is used to change the output format.
You can also apply augmentation using filters.
See :ref:`augmentation` for the detail.
