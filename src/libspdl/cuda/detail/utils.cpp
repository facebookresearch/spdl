/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/cuda/detail/utils.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace spdl::cuda::detail {

const char* get_error_name(CUresult error) {
  const char* p;
  if (cuGetErrorName(error, &p) == CUDA_SUCCESS) {
    return p;
  } else {
    return "UNKNOWN ERROR";
  }
}

const char* get_error_desc(CUresult error) {
  const char* p;
  if (cuGetErrorString(error, &p) == CUDA_SUCCESS) {
    return p;
  } else {
    return "Unknown error has occurred.";
  }
}

static std::string format_cuda_error(CUresult error) {
  return std::string{get_error_name(error)} + ": " + get_error_desc(error);
}

static std::shared_mutex CUCONTEXT_MUTEX;
static std::unordered_map<CUdevice, CUcontext> CUCONTEXT_CACHE;

CUcontext get_cucontext(CUdevice device) {
  {
    std::shared_lock<std::shared_mutex> lock(CUCONTEXT_MUTEX);
    if (CUCONTEXT_CACHE.contains(device)) {
      return CUCONTEXT_CACHE.at(device);
    }
  }
  std::lock_guard<std::shared_mutex> lock(CUCONTEXT_MUTEX);
  if (!CUCONTEXT_CACHE.contains(device)) {
    // If the current context is set, and is the same device, then
    // use it.
    CUcontext ctx = nullptr;
    TRACE_EVENT("nvdec", "cuCtxGetCurrent");
    CHECK_CU(cuCtxGetCurrent(&ctx), "Failed to get the current CUDA context.");
    if (ctx) {
      VLOG(5) << "Context found.";
      CUdevice dev;
      TRACE_EVENT("nvdec", "cuCtxGetDevice");
      CHECK_CU(
          cuCtxGetDevice(&dev),
          "Failed to get the device of the current CUDA context.");
      if (device == dev) {
        VLOG(5) << "The current context is the same device.";
        CUCONTEXT_CACHE.emplace(device, ctx);
        return ctx;
      }
    }
    VLOG(5) << "Context not found.";
    // Context is not set or different device, create floating one.
    TRACE_EVENT("nvdec", "cuDevicePrimaryCtxRetain");
    CHECK_CU(
        cuDevicePrimaryCtxRetain(&ctx, device),
        "Failed to retain the primary context.");

    CUCONTEXT_CACHE.emplace(device, ctx);
  }
  return CUCONTEXT_CACHE.at(device);
}

static CUcontext get_current_context() {
  CUcontext context = nullptr;
  CHECK_CU(
      cuCtxGetCurrent(&context), "Failed to get the current CUDA context.");
  return context;
}

CUDAContextPushGuard::CUDAContextPushGuard(int device_index)
    : context_{get_cucontext(device_index)},
      previous_context_{get_current_context()} {
  CHECK_CU(cuCtxPushCurrent(context_), "Failed to push the CUDA context.");
}

CUDAContextPushGuard::~CUDAContextPushGuard() noexcept {
  try {
    CUcontext popped_context = nullptr;
    const CUresult status = cuCtxPopCurrent(&popped_context);
    if (status != CUDA_SUCCESS) {
      LOG_EVERY_N(WARNING, 100) << "Failed to pop the CUDA context ("
                                << format_cuda_error(status) << ")";
      return;
    }
    if (popped_context == context_) {
      return;
    }

    CUcontext restored_context = nullptr;
    const CUresult current_status = cuCtxGetCurrent(&restored_context);
    if (current_status != CUDA_SUCCESS) {
      LOG_EVERY_N(WARNING, 100)
          << "Failed to inspect the restored CUDA context ("
          << format_cuda_error(current_status) << ")";
      return;
    }
    if (restored_context == previous_context_) {
      return;
    }

    LOG_EVERY_N(WARNING, 100)
        << "CUDA context stack was not restored after guarded scope "
        << "(expected current=" << previous_context_
        << ", actual current=" << restored_context << ", pushed=" << context_
        << ", popped=" << popped_context << ").";
  } catch (...) {
    // Context cleanup cannot safely replace an exception already in flight.
  }
}

} // namespace spdl::cuda::detail
