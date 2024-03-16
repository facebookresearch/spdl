#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

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
      XLOG(DBG) << "Context found.";
      CUdevice dev;
      TRACE_EVENT("nvdec", "cuCtxGetDevice");
      CHECK_CU(
          cuCtxGetDevice(&dev),
          "Failed to get the device of the current CUDA context.");
      if (device == dev) {
        XLOG(DBG) << "The current context is the same device.";
        CUCONTEXT_CACHE.emplace(device, ctx);
        return ctx;
      }
    }
    XLOG(DBG) << "Context not found.";
    // Context is not set or different device, create floating one.
    TRACE_EVENT("nvdec", "cuDevicePrimaryCtxRetain");
    CHECK_CU(
        cuDevicePrimaryCtxRetain(&ctx, device),
        "Failed to retain the primary context.");

    CUCONTEXT_CACHE.emplace(device, ctx);
  }
  return CUCONTEXT_CACHE.at(device);
}

CUdevice get_cuda_device_index(CUdeviceptr ptr) {
  CUcontext data;
  CHECK_CU(
      cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_CONTEXT, ptr),
      "Failed to fetch the CUDA context associated with a pointer.");
  CHECK_CU(
      cuCtxPushCurrent(data),
      "Failed to push the CUDA context associated with a pointer.");

  CUdevice device;
  auto result = cuCtxGetDevice(&device);

  CHECK_CU(
      cuCtxPopCurrent(&data),
      "Failed to pop the CUDA context associated with a pointer.");

  CHECK_CU(result, "Failed to fetch the CUDA device index from a pointer.");
  return device;
}

void set_current_cuda_context(CUdeviceptr ptr) {
  CUcontext data;
  CHECK_CU(
      cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_CONTEXT, ptr),
      "Failed to fetch the CUDA context associated with a pointer.");
  CHECK_CU(
      cuCtxPushCurrent(data),
      "Failed to push the CUDA context associated with a pointer.");
}

void init_cuda() {
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
