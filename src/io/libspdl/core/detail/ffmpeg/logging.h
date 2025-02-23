/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libspdl/core/detail/logging.h"

#include <fmt/core.h>

extern "C" {
#include <libavutil/attributes.h>
#include <libavutil/error.h>
}

namespace spdl::core::detail {
namespace {
// Replacement of av_err2str, which causes
// `error: taking address of temporary array`
// https://github.com/joncampbell123/composite-video-simulator/issues/5
av_always_inline std::string av_err2string(int errnum) {
  char str[AV_ERROR_MAX_STRING_SIZE];
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
} // namespace

template <typename... Args>
inline std::string av_error(int errnum, std::string_view tmp, Args&&... args) {
  return fmt::format(
      "{} ({})",
      fmt::vformat(tmp, fmt::make_format_args(std::forward<Args>(args)...)),
      av_err2string(errnum));
}
} // namespace spdl::core::detail

#define CHECK_AVERROR(expression, ...)                               \
  if (int _errnum = expression; _errnum < 0) [[unlikely]] {          \
    SPDL_FAIL(::spdl::core::detail::av_error(_errnum, __VA_ARGS__)); \
  }

#define CHECK_AVERROR_NUM(errnum, ...)                              \
  if (errnum < 0) [[unlikely]] {                                    \
    SPDL_FAIL(::spdl::core::detail::av_error(errnum, __VA_ARGS__)); \
  }

#define CHECK_AVALLOCATE(expression)                    \
  [&]() {                                               \
    auto* p = expression;                               \
    if (!p) [[unlikely]] {                              \
      SPDL_FAIL("Allocation failed (" #expression ")"); \
    }                                                   \
    return p;                                           \
  }()
