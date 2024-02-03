#pragma once

extern "C" {
#include <libavutil/attributes.h>
#include <libavutil/error.h>
}

#include <fmt/core.h>
#include <folly/logging/xlog.h>
#include <libspdl/core/logging.h>

namespace spdl {

////////////////////////////////////////////////////////////////////////////////
// Logging
////////////////////////////////////////////////////////////////////////////////

// Replacement of av_err2str, which causes
// `error: taking address of temporary array`
// https://github.com/joncampbell123/composite-video-simulator/issues/5
av_always_inline std::string av_err2string(int errnum) {
  char str[AV_ERROR_MAX_STRING_SIZE];
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}

template <typename... Args>
inline std::string av_error(int errnum, std::string_view tmp, Args&&... args) {
  return fmt::format(
      "{} ({})",
      fmt::vformat(tmp, fmt::make_format_args(std::forward<Args>(args)...)),
      av_err2string(errnum));
}

#define CHECK_AVERROR(expression, ...)                      \
  if (int _errnum = expression; _errnum < 0) [[unlikely]] { \
    SPDL_FAIL(av_error(_errnum, __VA_ARGS__));              \
  }

#define CHECK_AVERROR_NUM(errnum, ...)        \
  if (errnum < 0) [[unlikely]] {              \
    SPDL_FAIL(av_error(errnum, __VA_ARGS__)); \
  }

#define CHECK_AVALLOCATE(expression)                    \
  [&]() {                                               \
    auto* p = expression;                               \
    if (!p) [[unlikely]] {                              \
      SPDL_FAIL("Allocation failed (" #expression ")"); \
    }                                                   \
    return p;                                           \
  }()

} // namespace spdl
