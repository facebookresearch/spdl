#include "libspdl/core/detail/nvdec/buffer.h"

#include "libspdl/core/detail/logging.h"

#include <fmt/core.h>

namespace spdl::core {

CUDABufferTracker::CUDABufferTracker(
    int index,
    size_t n_,
    size_t c_,
    size_t h_,
    size_t w_)
    : buffer(cuda_buffer({n_, c_, h_, w_}, 0, index)),
      n(n_),
      c(c_),
      h(h_),
      w(w_) {}

CUDABufferTracker::CUDABufferTracker(int index, size_t c_, size_t h_, size_t w_)
    : buffer(cuda_buffer({c_, h_, w_}, 0, index)), n(1), c(c_), h(h_), w(w_) {}

uint8_t* CUDABufferTracker::get_next_frame() {
  if (i >= n) {
    SPDL_FAIL_INTERNAL(fmt::format(
        "Attempted to write beyond the maximum number of frames. max_frames={}, n={}",
        n,
        i));
  }
  return (uint8_t*)(buffer->data()) + i * c * h * w;
}

} // namespace spdl::core
