#pragma once
#include <libspdl/core/buffer.h>

namespace spdl::core {

/// Contiguous array data on a CUDA device.
/// This class is used to hold data decoded with NVDEC.
struct CUDABufferTracker {
  CUDABufferPtr buffer;

  // ``i`` keeps track of how many frames are written.
  // ``i`` < ``n``;
  // ``i`` is updated by writer.
  size_t n, c, h, w;

  size_t i{0};

  // For batch image / video
  CUDABufferTracker(int device_index, size_t n, size_t c, size_t h, size_t w);
  // For single image
  CUDABufferTracker(int device_index, size_t c, size_t h, size_t w);

  // Get the pointer to the head of the next frame.
  uint8_t* get_next_frame();
};

} // namespace spdl::core
