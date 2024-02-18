#include <libspdl/core/detail/nvdec/wrapper.h>

#include <libspdl/core/detail/cuda.h>
#include <libspdl/core/detail/tracing.h>

#include <fmt/core.h>
#include <folly/logging/xlog.h>

#include <cuda.h>

#define WAN_IF_NOT_SUCCESS(expr, msg)                    \
  do {                                                   \
    auto __status = expr;                                \
    if (__status != CUDA_SUCCESS) {                      \
      XLOG(WARN) << fmt::format(                         \
          "{} ({}: {})",                                 \
          msg,                                           \
          spdl::core::detail::get_error_name(__status),  \
          spdl::core::detail::get_error_desc(__status)); \
    }                                                    \
  } while (0)

namespace spdl::core::detail {

void CUvideoparserDeleter::operator()(CUvideoparser p) {
  WAN_IF_NOT_SUCCESS(
      cuvidDestroyVideoParser(p), "Failed to destroy CUvideoparser.");
}

void CUvideodecoderDeleter::operator()(void* p) {
  WAN_IF_NOT_SUCCESS(
      cuvidDestroyDecoder((CUvideodecoder)p),
      "Failed to destroy CUvideodecoder.");
};

void CUvideoctxlockDeleter::operator()(void* p) {
  WAN_IF_NOT_SUCCESS(
      cuvidCtxLockDestroy((CUvideoctxlock)p),
      "Failed to create CUvideoctxlock.");
}

MapGuard::MapGuard(
    CUvideodecoder dec,
    CUVIDPROCPARAMS* proc_params,
    int picture_index)
    : decoder(dec) {
  TRACE_EVENT("nvdec", "cuvidMapVideoFrame");
  CHECK_CU(
      cuvidMapVideoFrame(decoder, picture_index, &frame, &pitch, proc_params),
      "Failed to map video frame.");
}

MapGuard::~MapGuard() {
  TRACE_EVENT("nvdec", "cuvidUnmapVideoFrame");
  CHECK_CU(
      cuvidUnmapVideoFrame(decoder, frame), "Failed to unmap video frame.");
}

} // namespace spdl::core::detail
