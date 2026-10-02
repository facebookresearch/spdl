/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <nppdefs.h>

#include <fmt/core.h>
#include <glog/logging.h>

#include <cstdint>

namespace spdl::cuda::detail {

const char* to_string(NppStatus);

NppStreamContext get_npp_stream_context(
    uintptr_t stream_handle,
    int device_index);

} // namespace spdl::cuda::detail

#define CHECK_NPP(expr, msg)                                              \
  do {                                                                    \
    auto _status = expr;                                                  \
    if (_status < 0) {                                                    \
      SPDL_FAIL(fmt::format("{} ({})", msg, detail::to_string(_status))); \
    } else if (_status > 0) {                                             \
      SPDL_WARN(                                                          \
          fmt::format("[NPP WARNING] ({})", detail::to_string(_status))); \
    }                                                                     \
  } while (0)
