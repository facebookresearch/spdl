/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <glog/logging.h>

#include <stdexcept>
#include <string_view>
#include <version>

// CUDA interferes with <source_location> header
// https://forums.developer.nvidia.com/t/c-20s-source-location-compilation-error-when-using-nvcc-12-1/258026
#if defined SPDL_USE_CUDA && __has_include(<experimental/source_location>)
#include <experimental/source_location>
#elif defined __cpp_lib_source_location
#include <source_location>
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
#else
#error \
    "Neither <source_location> or <experimental/source_location> is available."
#endif

namespace spdl::common {

#if defined SPDL_USE_CUDA && __has_include(<experimental/source_location>)
using std::experimental::source_location;
#elif defined __cpp_lib_source_location
using std::source_location;
#else
using std::experimental::source_location;
#endif

std::string get_err_str(
    const std::string_view msg,
    const source_location& location);

std::string get_internal_err_str(
    const std::string_view msg,
    const source_location& location);

/// Exception thrown when unexpected internal error occurs.
class InternalError : public std::logic_error {
  using std::logic_error::logic_error;
};

} // namespace spdl::common

#define SPDL_FAIL(msg)           \
  throw std::runtime_error(      \
      spdl::common::get_err_str( \
          msg, spdl::common::source_location::current()))

#define SPDL_FAIL_INTERNAL(msg)           \
  throw spdl::common::InternalError(      \
      spdl::common::get_internal_err_str( \
          msg, spdl::common::source_location::current()))

#define SPDL_WARN(msg)                        \
  LOG(WARNING) << (spdl::common::get_err_str( \
      msg, spdl::common::source_location::current()))
