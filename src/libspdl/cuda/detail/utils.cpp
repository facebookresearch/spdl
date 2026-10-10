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

CUDAContextPushGuard::CUDAContextPushGuard(int device_index)
    : context_{get_cucontext(device_index)} {
  CHECK_CU(cuCtxPushCurrent(context_), "Failed to push the CUDA context.");
}

CUDAContextPushGuard::~CUDAContextPushGuard() noexcept {
  try {
    CUcontext popped_context = nullptr;
    const CUresult status = cuCtxPopCurrent(&popped_context);
    if (status != CUDA_SUCCESS) {
      LOG(WARNING) << "Failed to pop the CUDA context ("
                   << get_error_name(status) << ": " << get_error_desc(status)
                   << ")";
    } else if (popped_context != context_) {
      // Runtime APIs can legitimately replace the top context inside this
      // scope, while an unbalanced nested push has the same observable result.
      // Popping again could remove a caller-owned context, so leave the
      // restored stack alone and emit a bounded diagnostic.
      LOG_FIRST_N(WARNING, 1)
          << "cuCtxPopCurrent returned an unexpected CUDA context; the "
             "guarded scope replaced or unbalanced the context stack.";
    }
  } catch (...) {
    // Context cleanup cannot safely replace an exception already in flight.
  }
}

} // namespace spdl::cuda::detail
