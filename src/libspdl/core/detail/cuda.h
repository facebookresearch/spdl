#pragma once

#include <libspdl/core/logging.h>

#include <fmt/core.h>

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

namespace spdl::core::detail {
const char* get_error_name(CUresult error);
const char* get_error_desc(CUresult error);
} // namespace spdl::core::detail

#define CHECK_CU(expr, msg)                              \
  do {                                                   \
    auto _status = expr;                                 \
    if (_status != CUDA_SUCCESS) {                       \
      SPDL_FAIL(fmt::format(                             \
          "{} ({}: {})",                                 \
          msg,                                           \
          spdl::core::detail::get_error_name(_status),   \
          spdl::core::detail::get_error_desc(_status))); \
    }                                                    \
  } while (0)
