/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libspdl/core/detail/logging.h"
#include "libspdl/cuda/storage.h"

#include <fmt/format.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nvjpeg.h>

namespace spdl::cuda::detail {

//////////////////////////////////////////////////////////////////////////////
// nvjpeg handle
//////////////////////////////////////////////////////////////////////////////

// TODO: Add support for cuda_allocator?
nvjpegHandle_t get_nvjpeg();

//////////////////////////////////////////////////////////////////////////////
// nvjpeg Jpeg state
//////////////////////////////////////////////////////////////////////////////
struct nvjpeg_state_deleter {
  using DestroyFn = decltype(&nvjpegJpegStateDestroy);

  DestroyFn destroy = nvjpegJpegStateDestroy;

  void operator()(nvjpegJpegState*) const noexcept;
};

using nvjpegStatePtr = std::unique_ptr<nvjpegJpegState, nvjpeg_state_deleter>;

nvjpegStatePtr get_nvjpeg_jpeg_state(nvjpegHandle_t);

//////////////////////////////////////////////////////////////////////////////
// Misc
//////////////////////////////////////////////////////////////////////////////
std::string to_string(nvjpegStatus_t);
std::string to_string(nvjpegBackend_t);
std::string to_string(nvjpegOutputFormat_t);

nvjpegBackend_t get_nvjpeg_backend(const std::optional<std::string>&);
nvjpegOutputFormat_t get_nvjpeg_output_format(const std::string&);

struct NVJPEGImageLayout {
  size_t width;
  size_t height;
  size_t num_channels;
  bool interleaved;
};

void wrap_nvjpeg_image(
    void* data,
    const NVJPEGImageLayout& layout,
    nvjpegImage_t& image,
    size_t batch = 0);

void retain_cuda_storage_dependencies(
    CUDAStoragePtr& storage,
    std::vector<CUDAStoragePtr> dependencies);

class CUDAStreamSyncOnExceptionGuard {
 public:
  using SynchronizeFn = cudaError_t (*)(cudaStream_t);

  explicit CUDAStreamSyncOnExceptionGuard(
      uintptr_t stream,
      SynchronizeFn synchronize = cudaStreamSynchronize) noexcept;
  ~CUDAStreamSyncOnExceptionGuard() noexcept;

  CUDAStreamSyncOnExceptionGuard(const CUDAStreamSyncOnExceptionGuard&) =
      delete;
  CUDAStreamSyncOnExceptionGuard& operator=(
      const CUDAStreamSyncOnExceptionGuard&) = delete;
  CUDAStreamSyncOnExceptionGuard(CUDAStreamSyncOnExceptionGuard&&) = delete;
  CUDAStreamSyncOnExceptionGuard& operator=(CUDAStreamSyncOnExceptionGuard&&) =
      delete;

 private:
  uintptr_t stream_;
  SynchronizeFn synchronize_;
  int uncaught_exceptions_;
};

} // namespace spdl::cuda::detail

#define CHECK_NVJPEG(expr, msg)                                         \
  do {                                                                  \
    auto _status = expr;                                                \
    if (_status != NVJPEG_STATUS_SUCCESS) {                             \
      SPDL_FAIL(                                                        \
          fmt::format(                                                  \
              "{} ({})", msg, spdl::cuda::detail::to_string(_status))); \
    }                                                                   \
  } while (0)
