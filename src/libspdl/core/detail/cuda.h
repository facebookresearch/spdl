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
