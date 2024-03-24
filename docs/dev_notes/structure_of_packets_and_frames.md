# How packets/frames are structured in C++ and exposed to Python

At the center of libspdl is data structures that hold packets and frames.
They are passed around different functions and across the boundary of C++ and
Python forth and back.

This note explains how the lifecycle of packets/frames are managed and how
they are exposed to Python.

If you wonder why structure like `FFmpegFramesWrapperPtr<MediaType::Audio>`
has layers of managed pointers (both `std::unique_ptr` and `std::shared_ptr`),
this note should help understand the purpose of each pointer types.

## AVPacket / AVFrame and RAII wrapper

The most primitive structures that hold packets/frames data are
FFmpeg's [`AVPacket`](https://ffmpeg.org/doxygen/5.1/structAVPacket.html) and
[`AVFrame`](https://ffmpeg.org/doxygen/5.1/structAVFrame.html).

These structures are accompanied with helper functions for allocation and deallocation.

```C
// Allocate a packet
AVPacket* packet = av_packet_alloc();

// Process the packet
...

// Deallocate packet
av_packet_free(&packet);
```

```C
// Allocate frame
AVFrame* frame = av_frame_alloc();

// Process the frame
...

// Deallocate the frame
av_frame_free(&frame);
```

!!! note

    The actual decoded data in `AVFrame*` are managed via through separate
    reference counting mechanism, but it is ommited here for brevity.

Manually calling deallocation functions is tedious and error-prone. And since
we will be passing the data across the boundary of C++/Python, it is
important to make the ownership of the data clear.

In C++, [RAII](https://en.cppreference.com/w/cpp/language/raii) is a common pattern
to automate the resource management, and the smart pointer, `std::unique_ptr<T>`, makes the
ownership of underlying data clear.

W can define a class that holds the packet/frame data, cleans up the resource when an
istance reaches its end of life.

```C++
struct Deleter{
  void operator()(AVPacket* p) {
    av_frame_free(&p);
  }
};

using PacketPtr = std::unique_ptr<AVPacket, Deleter>;
```

Now the ownership of the data is clear.
The function that takes the `PacketPtr` class owns the data.

```C++
// decode function takes the ownership
FramesPtr decode(PacketPtr packet);
```

To pass data to such function, `std::move` must be used.

```
// call site must be explicit about giving up the ownership of the data
auto frames = decode(std::move(packet));
```

The data owned by the `packet` instance is now owned by `decode` function and
it will be deallocated at the end of `decode` function.

Helper functions that do not alter the lifetime of the data can take the
variable as reference or raw pointer.

In libspdl, we take the later approach and pass the raw pointer.

```C++
// Helper function that requires temporal access to packet data.
// It might alter the data, but does not alter the lifetime of the data.
std::string get_codec_name(AVPacket* p);

void some_func(PacketPtr packet) {
    auto codec_name = get_codec_name(packet.get());
}
```

## Handling a series of packets/frames

When decoding media stream such as audio and video, multiple packets/frames are
processed.
The demuxing process generates heap-allocated `AVPacket*` objects, and the decoding
process generates heap-allocated `AVFrame*` objects.
We need to track these generated objects and we pass around them as chunk.

To do this, we create a new struct, which use `std::vector<AVFrame*>` as a container and
bulk deallocate the data when an instance is being destructed.

Such a class for `AVPacket` can be implemented like the following.

```C++
struct Packets {
  std::vector<AVPacket*> packets;

  void push(AVPacket*); // push new packet created by demuxing process

  ~Packets() {
    // Bulk deallocate the data
    std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
      av_packet_free(&p);
    });
  };
};
```

## Exposing the structure to Python

We use [PyBind11](https://pybind11.readthedocs.io/en/stable/index.html)
to bind the C++ code to Python.
PyBind11 by default uses
[`std::unique_ptr` as a holder type](https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html)
to expose a class structure.

```C++
py::class_<Packets, /*holder type*/std::unique_ptr<Packets>> _Packets(m, "Packets");
```

This approach on data management is aligned with the idea of clear ownership,
however, there are two problems when writing high-performance library.

1. It does not allow to move the ownership of data from Python to C++.
2. The lifetime of the data is tied to the lifetime of Python variables referencing the data.

As described in the PyBind11 documentation,
[binding objects with `std::unique_ptr` makes it impossible for Python to give up the resource ownership](https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#std-unique-ptr).
This means that any function that processes these data cannot take the ownership
of the data, thus the the data cannot be deallocated as part of the process.
Instead, the data is deallocated when the Python variables referencing the data are
all garbage collected.

As a result, it becomes impossible to controll when the data are deallocated, and
more importantly, the underlying data are deallocated in the main execution flow
of Python interpreter.

We want to offload the CPU heavy work and the related resource management from the main
thread to other threads, and the Python's main execution flow to focus on orchestrating
the different data process pipelines. However, if data are deallocated in Python's
main thread when garbage collection happens, it blocks the orchestrations ever so often.

To get around this, the C++ code needs to be able to take the ownership of data
from Python variables. We need to disconnect the lifecycle of data from the lifecycle
of Python variables.

## Disconnecting the data lifecycle from the lifecycle of Python variable

To take the ownership of data from Python to C++, while having the clear ownership of
data using `std::unique_ptr`, we introduce a wrapper class and bind the wrapper class
with `std::shared_ptr`.

```C++
struct PacketsWrapper {
  std::unique_ptr<Packets> packets;
};

// Use `std::shared_ptr` as holder class.
py::class_<PacketsWrapper, std::shared_ptr<PacketsWrapper>> _Packets(m, "Packets");
```

By doing so, we can write the functions that takes the ownership of the data, so that
they cleans up the resource when the process is completed.

```C++
// Now we can let the `decode` function to take the ownership again.
std::unique_ptr<Frames> decode(std::unique_ptr<Packets>);

// Binding code wraps/unwraps the data
m.def("decode", [](PacketsWrapper wrapper){
  auto frames = decode(std::move(wrapper.packets));
  return wrap(std::move(frames)); // wrap Frames to FramesWrapper
});
```

The downside of this approach is that now that Python code must be aware of the
implicit data lifecycle. Care must be taken so that functinos that take ownership of
data are not called multiple times on the data object.