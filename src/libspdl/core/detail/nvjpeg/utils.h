
#pragma once

#include "libspdl/core/detail/logging.h"

#include <fmt/format.h>

#include <memory>
#include <optional>
#include <string>

#include <nvjpeg.h>

namespace spdl::core::detail {

//////////////////////////////////////////////////////////////////////////////
// nvjpeg handle
//////////////////////////////////////////////////////////////////////////////

// TODO: Add support for cuda_allocator?
nvjpegHandle_t get_nvjpeg();

//////////////////////////////////////////////////////////////////////////////
// nvjpeg Jpeg state
//////////////////////////////////////////////////////////////////////////////
struct nvjpeg_state_deleter {
  void operator()(nvjpegJpegState*);
};

using nvjpegStatePtr = std::unique_ptr<nvjpegJpegState, nvjpeg_state_deleter>;

nvjpegStatePtr get_nvjpeg_jpeg_state(nvjpegHandle_t);

//////////////////////////////////////////////////////////////////////////////
// Misc
//////////////////////////////////////////////////////////////////////////////
std::string to_string(nvjpegStatus_t);
std::string to_string(nvjpegBackend_t);
std::string to_string(nvjpegOutputFormat_t);

nvjpegBackend_t get_nvjpeg_backend(const std::optional<std::string>&);
nvjpegOutputFormat_t get_nvjpeg_output_format(const std::string&);

} // namespace spdl::core::detail

#define CHECK_NVJPEG(expr, msg)                                     \
  do {                                                              \
    auto _status = expr;                                            \
    if (_status != NVJPEG_STATUS_SUCCESS) {                         \
      SPDL_FAIL(fmt::format(                                        \
          "{} ({})", msg, spdl::core::detail::to_string(_status))); \
    }                                                               \
  } while (0)
