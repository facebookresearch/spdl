#include <libspdl/core/buffer.h>

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

CUDABuffer2DPitch::CUDABuffer2DPitch(
    int index,
    size_t max_frames_,
    bool is_image_)
    : device_index(index), max_frames(max_frames_), is_image(is_image_) {
  TRACE_EVENT(
      "decoding",
      "CUDABuffer2DPitch::CUDABuffer2DPitch",
      perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));
}

CUDABuffer2DPitch::~CUDABuffer2DPitch() {
  TRACE_EVENT(
      "decoding",
      "CUDABuffer2DPitch::~CUDABuffer2DPitch",
      perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));
  if (p) {
    TRACE_EVENT("nvdec", "cuMemFree");
    auto status = cuMemFree(p);
    if (status != CUDA_SUCCESS) {
      XLOG(CRITICAL) << fmt::format(
          "Failed to free CUDA memory ({}: {})",
          spdl::core::detail::get_error_name(status),
          spdl::core::detail::get_error_desc(status));
    }
  }
}

void CUDABuffer2DPitch::allocate(size_t c_, size_t h_, size_t w_, size_t bpp_) {
  if (p) {
    SPDL_FAIL_INTERNAL("Arena is already allocated.");
  }
  c = c_, h = h_, w = w_, bpp = bpp_;

  width_in_bytes = w * bpp;
  size_t height = max_frames * c * h;

  TRACE_EVENT("nvdec", "cuMemAllocPitch");
  CHECK_CU(
      cuMemAllocPitch((CUdeviceptr*)&p, &pitch, width_in_bytes, height, 8),
      "Failed to allocate memory.");
}

std::vector<size_t> CUDABuffer2DPitch::get_shape() const {
  if (is_image) {
    return std::vector<size_t>{c, h, w};
  }
  return std::vector<size_t>{n, c, h, w};
}

uint8_t* CUDABuffer2DPitch::get_next_frame() {
  if (!p) {
    SPDL_FAIL_INTERNAL("Memory is not allocated.");
  }
  if (n >= max_frames) {
    SPDL_FAIL_INTERNAL(fmt::format(
        "Attempted to write beyond the maximum number of frames. max_frames={}, n={}",
        max_frames,
        n));
  }
  if (is_image && n == 1) {
    SPDL_FAIL_INTERNAL("Attempted to write multiple frames for image buffer.");
  }
  return (uint8_t*)p + n * c * h * pitch;
}

} // namespace spdl::core
