# libspdl

A modern C++20 media codec library wrapping FFmpeg, with optional CUDA GPU
acceleration. It provides a composable pipeline for demuxing, decoding,
filtering, encoding, and muxing audio, video, and image data.

## Architecture

```
Source Data --> Adaptor --> DataInterface --> Demuxer --> Packets
                                                          |
                                                          v
                                            Decoder --> Frames --> CPUBuffer
                                                       ^  |            |
                                                       |  v            v
                                                    FilterGraph   CUDABuffer (GPU)
                                                       |
                                                       v
                                            Encoder --> Packets --> Muxer --> Output
```

The FilterGraph transforms Frames into Frames. It is commonly used as part of
the decode pipeline for video/image resize and color space conversion, or audio
sample rate and channel layout manipulation. It can also feed into the Encoder
for transcoding workflows.

### Pipeline stages

| Stage | Header | Description |
|-------|--------|-------------|
| Adaptor | `core/adaptor.h` | Pluggable data sources (file, memory, custom) |
| Demuxer | `core/demuxing.h` | Extracts compressed packets from containers |
| Packets | `core/packets.h` | Type-safe compressed packet containers |
| Decoder | `core/decoder.h` | Converts packets to raw frames (CPU or NVDEC) |
| Frames | `core/frames.h` | Decoded frame containers with media-specific accessors |
| FilterGraph | `core/filter_graph.h` | FFmpeg filter chain processing |
| Encoder | `core/encoder.h` | Compresses frames back to packets |
| Muxer | `core/muxer.h` | Writes packets to output containers |
| Buffer | `core/buffer.h`, `cuda/buffer.h` | Contiguous N-D arrays (CPU or GPU) |

### Streaming support

The library uses C++20 coroutines (`core/generator.h`) for lazy evaluation.
`streaming_demux()` and `streaming_decode_packets()` yield results on demand
without buffering entire files in memory.

## Directory layout

```
libspdl/
  core/            CPU-side functionality, FFmpeg abstraction
    detail/        Internal implementation (not part of the public API)
      ffmpeg/      RAII wrappers around FFmpeg C structs
    adaptor/       Data source abstraction
  cuda/            GPU acceleration (CUDA, NVDEC, NVJPEG)
    detail/        Internal CUDA implementation
    nvjpeg/        NVIDIA JPEG decoder
  tests/           Tests and test data generation
```

## Type system

All core types are templated on `MediaType` (Audio, Video, Image). C++20
`requires` clauses restrict operations to valid media types at compile time:

- Audio-only: `get_sample_rate()`, `get_num_channels()`
- Video/Image: `get_width()`, `get_height()`, `get_num_planes()`
- Video-only: `slice()`, `streaming_decode_packets()`

The `Encoder` template is only instantiated for Audio and Video (encoding a
single image is handled through the Video path).

## Design decisions

### pImpl with raw pointers

Public headers (`decoder.h`, `encoder.h`, `demuxing.h`) use raw `pImpl_`
pointers with manual `new`/`delete` in the `.cpp` files. This is intentional:
`std::unique_ptr` requires the pointed-to type to be complete at the point where
the destructor is instantiated, which would force FFmpeg headers (via the
`detail::*Impl` classes) into the public API. The raw pointer pattern keeps
FFmpeg as a pure implementation detail, so downstream consumers never need
FFmpeg headers on their include path.

### Optional parameters in public APIs

Functions like `demux_window()` and `make_demuxer()` take `std::optional`
parameters with defaults. This is intentional — the common case is clean
(e.g., `demuxer->demux_window<MediaType::Video>()`), and a builder/config
pattern would add complexity for little gain.

### CUDAConfig simplicity

`CUDAConfig` requires only `device_index`; `stream` and `allocator` have
sensible defaults. A convenience wrapper is unnecessary — the struct is already
minimal: `CUDAConfig{.device_index = 0}`.

### NVDEC per-frame synchronization

In `cuda/nvdec/detail/buffer.cpp`, `cuMemcpy2DAsync` is immediately followed by
`cuStreamSynchronize` on every frame. This is intentional: the source pointer
comes from `cuvidMapVideoFrame` and is unmapped by the caller as soon as `push()`
returns. The sync ensures the async copy completes before the mapped frame is
released. Batching would require deferring unmap, which is a significant
architectural change.

### CUDA context cache locking

`cuda/detail/utils.cpp` uses double-checked locking with `shared_mutex` for the
CUDA context cache. The pattern is correct: the write-lock path re-validates
before inserting, preventing duplicate entries.

## Quick start

```cpp
#include <libspdl/core/demuxing.h>
#include <libspdl/core/decoder.h>
#include <libspdl/core/conversion.h>

using namespace spdl::core;

// Demux
auto demuxer = make_demuxer("input.mp4");
auto codec = demuxer->get_default_codec<MediaType::Video>();
auto packets = demuxer->demux_window<MediaType::Video>();

// Decode
Decoder<MediaType::Video> decoder(codec, std::nullopt, std::nullopt);
auto frames = decoder.decode_packets(std::move(packets));

// Convert to buffer
auto buffer = convert_frames(std::move(frames));
```

### With GPU acceleration

```cpp
#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/transfer.h>

using namespace spdl::cuda;

CUDAConfig cuda_cfg{.device_index = 0};
auto gpu_buffer = transfer_buffer(buffer, cuda_cfg);
```
