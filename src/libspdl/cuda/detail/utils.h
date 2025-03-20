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

// Make sure CUDA_API_PER_THREAD_DEFAULT_STREAM is defined by compiler
// so that it is applied globally
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error CUDA_API_PER_THREAD_DEFAULT_STREAM must be defined.
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(expr, msg)            \
  do {                                   \
    auto _status = expr;                 \
    if (_status != cudaSuccess) {        \
      SPDL_FAIL(fmt::format(             \
          "{} ({}: {})",                 \
          msg,                           \
          cudaGetErrorName(_status),     \
          cudaGetErrorString(_status))); \
    }                                    \
  } while (0)

namespace spdl::cuda::detail {
const char* get_error_name(CUresult error);
const char* get_error_desc(CUresult error);
} // namespace spdl::cuda::detail

#define CHECK_CU(expr, msg)                              \
  do {                                                   \
    auto _status = expr;                                 \
    if (_status != CUDA_SUCCESS) {                       \
      SPDL_FAIL(fmt::format(                             \
          "{} ({}: {})",                                 \
          msg,                                           \
          spdl::cuda::detail::get_error_name(_status),   \
          spdl::cuda::detail::get_error_desc(_status))); \
    }                                                    \
  } while (0)

namespace spdl::cuda::detail {

// Get or create a CUcontext associated with the device.
// If the current context is associated with the device, then return it.
// Otherwise, create or fetch a floating primary context and return it.
//
// TODO: Test if this works if there is a context already created by others
CUcontext get_cucontext(CUdevice device);

// Set the current context to the primary context of the given device
void set_cuda_primary_context(int device_index);

} // namespace spdl::cuda::detail
