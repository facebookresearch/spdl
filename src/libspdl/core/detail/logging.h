/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <exception>
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

namespace spdl::core::detail {

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

} // namespace spdl::core::detail

#define SPDL_FAIL(msg)                                      \
  throw std::runtime_error(spdl::core::detail::get_err_str( \
      msg, spdl::core::detail::source_location::current()))

#define SPDL_FAIL_INTERNAL(msg)                                             \
  throw spdl::core::InternalError(spdl::core::detail::get_internal_err_str( \
      msg, spdl::core::detail::source_location::current()))

#define SPDL_WARN(msg)                              \
  LOG(WARNING) << (spdl::core::detail::get_err_str( \
      msg, spdl::core::detail::source_location::current()))
