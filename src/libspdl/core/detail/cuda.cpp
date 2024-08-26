/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace spdl::core::detail {

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
    return "Unknown error has occured.";
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
      VLOG(0) << "Context found.";
      CUdevice dev;
      TRACE_EVENT("nvdec", "cuCtxGetDevice");
      CHECK_CU(
          cuCtxGetDevice(&dev),
          "Failed to get the device of the current CUDA context.");
      if (device == dev) {
        VLOG(0) << "The current context is the same device.";
        CUCONTEXT_CACHE.emplace(device, ctx);
        return ctx;
      }
    }
    VLOG(0) << "Context not found.";
    // Context is not set or different device, create floating one.
    TRACE_EVENT("nvdec", "cuDevicePrimaryCtxRetain");
    CHECK_CU(
        cuDevicePrimaryCtxRetain(&ctx, device),
        "Failed to retain the primary context.");

    CUCONTEXT_CACHE.emplace(device, ctx);
  }
  return CUCONTEXT_CACHE.at(device);
}

void set_cuda_primary_context(int device_index) {
  CUcontext ctx = get_cucontext(device_index);
  CHECK_CU(cuCtxPushCurrent(ctx), "Failed to push the CUDA context.");
}

void ensure_cuda_initialized() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    TRACE_EVENT("nvdec", "cudaGetDeviceCount");
    int count;
    CHECK_CUDA(cudaGetDeviceCount(&count), "Failed to fetch the device count.");
    if (count == 0) {
      SPDL_FAIL("No CUDA device was found.");
    }
  });
}

} // namespace spdl::core::detail
