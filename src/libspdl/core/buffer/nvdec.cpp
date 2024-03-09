#include <libspdl/core/buffer.h>

#include <libspdl/core/detail/logging.h>
#include <libspdl/core/detail/tracing.h>

#include <libspdl/core/detail/cuda.h>

#include <fmt/core.h>

namespace spdl::core {

CUDABuffer2DPitch::CUDABuffer2DPitch(size_t max_frames_, bool is_image_)
    : max_frames(max_frames_), is_image(is_image_) {}

CUDABuffer2DPitch::~CUDABuffer2DPitch() {
  if (p) {
    TRACE_EVENT("nvdec", "cuMemFree");
    CHECK_CU(cuMemFree(p), "Failed to free memory.");
  }
}

void CUDABuffer2DPitch::allocate(
    size_t c_,
    size_t h_,
    size_t w_,
    size_t bpp_,
    bool channel_last_) {
  if (p) {
    SPDL_FAIL_INTERNAL("Arena is already allocated.");
  }
  channel_last = channel_last_;
  c = c_, h = h_, w = w_, bpp = bpp_;

  width_in_bytes = channel_last ? w * c * bpp : w * bpp;
  size_t height = channel_last ? max_frames * h : max_frames * c * h;

  TRACE_EVENT("nvdec", "cuMemAllocPitch");
  CHECK_CU(
      cuMemAllocPitch((CUdeviceptr*)&p, &pitch, width_in_bytes, height, 8),
      "Failed to allocate memory.");
}

std::vector<size_t> CUDABuffer2DPitch::get_shape() const {
  if (is_image) {
    return channel_last ? std::vector<size_t>{h, w, c}
                        : std::vector<size_t>{c, h, w};
  }
  return channel_last ? std::vector<size_t>{n, h, w, c}
                      : std::vector<size_t>{n, c, h, w};
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
  return channel_last ? (uint8_t*)p + n * h * pitch
                      : (uint8_t*)p + n * c * h * pitch;
}

} // namespace spdl::core
