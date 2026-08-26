Lifecycle Management of Packets and Frames
==========================================

.. note::

   This document describes the internal implementation details of libspdl.
   It is intended for developers who want to understand or contribute to
   the library's core design.

At the center of libspdl are data structures that hold packets and frames.
They are passed around different functions and across the boundary of C++ and
Python.

This note explains how the lifecycle of packets/frames are managed and how
they are exposed to Python.

AVPacket / AVFrame and RAII Wrappers
-------------------------------------

The most primitive structures that hold packets/frames data are
FFmpeg's `AVPacket <https://ffmpeg.org/doxygen/5.1/structAVPacket.html>`_ and
`AVFrame <https://ffmpeg.org/doxygen/5.1/structAVFrame.html>`_.

These structures are accompanied with helper functions for allocation and deallocation.

.. code-block:: c

   // Allocate a packet
   AVPacket* packet = av_packet_alloc();

   // Process the packet
   ...

   // Deallocate packet
   av_packet_free(&packet);

.. code-block:: c

   // Allocate frame
   AVFrame* frame = av_frame_alloc();

   // Process the frame
   ...

   // Deallocate the frame
   av_frame_free(&frame);

.. note::

   The actual decoded data in ``AVFrame*`` are managed via a separate
   reference counting mechanism, but it is omitted here for brevity.

Manually calling deallocation functions is tedious and error-prone. Since
we pass data across the boundary of C++/Python, it is important to make
the ownership of the data clear.

In C++, `RAII <https://en.cppreference.com/w/cpp/language/raii>`_ is a common pattern
to automate resource management, and the smart pointer, ``std::unique_ptr<T>``, makes the
ownership of underlying data clear.

We define custom deleters that properly clean up FFmpeg resources:

.. code-block:: c++

   struct AVPacketDeleter {
     void operator()(AVPacket* p);
   };

   using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;

   struct AVFrameDeleter {
     void operator()(AVFrame* p);
   };

   using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

Now the ownership of the data is clear.
Functions that take ``AVPacketPtr`` or ``AVFramePtr`` own the data.

.. code-block:: c++

   // decode function takes the ownership
   FramesPtr<media> decode(AVPacketPtr packet);

To pass data to such functions, ``std::move`` must be used:

.. code-block:: c++

   // call site must be explicit about giving up the ownership of the data
   auto frames = decode(std::move(packet));

The data owned by the ``packet`` instance is now owned by the ``decode`` function and
will be deallocated when no longer needed.

Helper functions that do not alter the lifetime of the data can take the
variable as reference or raw pointer. In libspdl, we take the latter approach
and pass raw pointers:

.. code-block:: c++

   // Helper function that requires temporal access to packet data.
   // It might alter the data, but does not alter the lifetime of the data.
   std::string get_codec_name(AVPacket* p);

   void some_func(AVPacketPtr packet) {
       auto codec_name = get_codec_name(packet.get());
   }

These RAII wrappers are defined in ``libspdl/core/detail/ffmpeg/wrappers.h``.

Handling a Series of Packets/Frames
------------------------------------

When decoding media streams such as audio and video, multiple packets/frames are
processed. The demuxing process generates heap-allocated ``AVPacket*`` objects, and the decoding
process generates heap-allocated ``AVFrame*`` objects.
We need to track these generated objects and pass them around as a chunk.

PacketSeries
~~~~~~~~~~~~

For packets, we use the ``PacketSeries`` class, which manages a ``std::vector<AVPacket*>``
and bulk deallocates the data when the instance is destructed:

.. code-block:: c++

   class PacketSeries {
     std::vector<AVPacket*> container_ = {};

    public:
     PacketSeries();
     ~PacketSeries();  // Bulk deallocate all AVPacket*

     void push(AVPacket* packet);
     const std::vector<AVPacket*>& get_packets() const;
   };

The ``PacketSeries`` class is copyable and movable, properly managing the lifecycle
of all contained packets.

Packets Structure
~~~~~~~~~~~~~~~~~

The ``Packets<media>`` template structure wraps ``PacketSeries`` along with metadata:

.. code-block:: c++

   template <MediaType media>
   struct Packets {
     uintptr_t id{};                           // Trace ID for debugging
     std::string src;                          // Source URI
     int stream_index;                         // Stream index in source

     PacketSeries pkts;                        // Series of compressed packets
     Rational time_base{};                     // Time base for timestamps
     std::optional<TimeWindow> timestamp;      // Optional time window
     std::optional<Codec<media>> codec;        // Codec information
   };

   using AudioPackets = Packets<MediaType::Audio>;
   using VideoPackets = Packets<MediaType::Video>;
   using ImagePackets = Packets<MediaType::Image>;

   template <MediaType media>
   using PacketsPtr = std::unique_ptr<Packets<media>>;

Frames Structure
~~~~~~~~~~~~~~~~

For frames, the ``Frames<media>`` class directly manages a ``std::vector<AVFrame*>``
with a custom destructor:

.. code-block:: c++

   template <MediaType media>
   class Frames {
     uintptr_t id_{0};                    // Trace ID
     Rational time_base_;                 // Time base
     std::vector<AVFrame*> frames_{};     // Series of decoded frames

    public:
     Frames(uintptr_t id, Rational time_base);
     ~Frames();  // Bulk deallocate all AVFrame*

     // No copy, only move
     Frames(const Frames&) = delete;
     Frames& operator=(const Frames&) = delete;
     Frames(Frames&&) noexcept;
     Frames& operator=(Frames&&) noexcept;

     void push_back(AVFrame* frame);
     const std::vector<AVFrame*>& get_frames() const;
   };

   using AudioFrames = Frames<MediaType::Audio>;
   using VideoFrames = Frames<MediaType::Video>;
   using ImageFrames = Frames<MediaType::Image>;

   template <MediaType media>
   using FramesPtr = std::unique_ptr<Frames<media>>;

The ``Frames`` class is move-only to ensure clear ownership semantics.

Exposing Structures to Python
------------------------------

We use `nanobind <https://nanobind.readthedocs.io/en/latest/>`_
to bind the C++ code to Python. Nanobind provides excellent support for
``std::unique_ptr`` as a holder type, allowing ownership transfer between
Python and C++.

.. code-block:: c++

   namespace nb = nanobind;

   void register_packets(nb::module_& m) {
     nb::class_<AudioPackets>(m, "AudioPackets")
       .def("clone", [](const AudioPackets& self) -> AudioPacketsPtr {
         return std::make_unique<AudioPackets>(self);
       });

     nb::class_<VideoPackets>(m, "VideoPackets")
       .def("clone", [](const VideoPackets& self) -> VideoPacketsPtr {
         return std::make_unique<VideoPackets>(self);
       });
   }

   void register_frames(nb::module_& m) {
     nb::class_<AudioFrames>(m, "AudioFrames", nb::dynamic_attr());
     nb::class_<VideoFrames>(m, "VideoFrames", nb::dynamic_attr());
     nb::class_<ImageFrames>(m, "ImageFrames");
   }

By default, nanobind uses ``std::unique_ptr`` as the holder type for classes,
which provides the following benefits:

1. **Clear ownership**: Functions can take ownership of data from Python
2. **Efficient resource management**: Data is deallocated when the unique_ptr goes out of scope
3. **Move semantics**: Ownership can be transferred without copying

Ownership Transfer from Python to C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a C++ function takes a ``PacketsPtr<media>`` or ``FramesPtr<media>`` parameter,
nanobind automatically transfers ownership from Python to C++:

.. code-block:: c++

   // C++ function that takes ownership
   FramesPtr<media> decode_and_flush(
       PacketsPtr<media> packets,
       int num_frames = -1);

   // Python binding
   m.def("decode_and_flush",
         &decode_and_flush,
         nb::arg("packets"),
         nb::arg("num_frames") = -1);

When called from Python, the packets object is moved into the C++ function,
and the Python variable becomes invalid:

.. code-block:: python

   packets = demuxer.demux_video()
   frames = decode_and_flush(packets)  # packets is moved, no longer valid

This design ensures that:

1. **Resource cleanup happens in worker threads**: When the C++ function completes,
   resources are deallocated in the worker thread, not in Python's main thread
   during garbage collection.

2. **No accidental reuse**: Once ownership is transferred, the Python variable
   cannot be used again, preventing bugs.

3. **Efficient pipeline processing**: Data flows through the pipeline with clear
   ownership at each stage.

Cloning for Multiple Uses
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to use the same packets or frames multiple times (e.g., decode
the same packets with different configurations), use the ``clone()`` method:

.. code-block:: python

   packets = demuxer.demux_video()

   # Clone for multiple decoding
   packets_copy = packets.clone()

   frames1 = decode_and_flush(packets)       # Original is consumed
   frames2 = decode_and_flush(packets_copy)  # Clone is consumed

The ``clone()`` method creates a deep copy of the packets/frames, including
all the underlying FFmpeg data structures.

Memory Management Benefits
---------------------------

This design provides several key benefits for high-performance media processing:

1. **Offloaded deallocation**: Heavy resource cleanup happens in worker threads,
   not in Python's main thread during garbage collection.

2. **Predictable lifecycle**: Resources are deallocated immediately when processing
   completes, not at some later garbage collection cycle.

3. **Clear ownership**: The type system enforces clear ownership semantics,
   preventing use-after-free bugs.

4. **Efficient pipelines**: Data flows through processing pipelines with minimal
   overhead and clear ownership transfer at each stage.

5. **Thread-safe processing**: Each processing stage owns its data, avoiding
   shared ownership complexities.

Implementation Details
----------------------

The key implementation files are:

- ``libspdl/core/detail/ffmpeg/wrappers.h``: RAII wrappers for FFmpeg types
- ``libspdl/core/packets.h``: PacketSeries and Packets structures
- ``libspdl/core/frames.h``: Frames class template
- ``spdl/io/lib/core/packets.cpp``: Python bindings for packets
- ``spdl/io/lib/core/frames.cpp``: Python bindings for frames

The design leverages modern C++ features (RAII, move semantics, smart pointers)
and nanobind's excellent support for these features to create a safe, efficient
interface between Python and C++.
