/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/cuda/nvdec/detail/wrapper.h"

#include "libspdl/core/detail/tracing.h"
#include "libspdl/cuda/utils.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <cuda.h>

#define WARN_IF_NOT_SUCCESS(expr, msg)                   \
  do {                                                   \
    auto __status = expr;                                \
    if (__status != CUDA_SUCCESS) {                      \
      LOG(WARNING) << fmt::format(                       \
          "{} ({}: {})",                                 \
          msg,                                           \
          spdl::cuda::detail::get_error_name(__status),  \
          spdl::cuda::detail::get_error_desc(__status)); \
    }                                                    \
  } while (0)

namespace spdl::cuda::detail {

void CUvideoparserDeleter::operator()(CUvideoparser p) {
  WARN_IF_NOT_SUCCESS(
      cuvidDestroyVideoParser(p), "Failed to destroy CUvideoparser.");
}

void CUvideodecoderDeleter::operator()(void* p) {
  WARN_IF_NOT_SUCCESS(
      cuvidDestroyDecoder((CUvideodecoder)p),
      "Failed to destroy CUvideodecoder.");
};

void CUvideoctxlockDeleter::operator()(void* p) {
  WARN_IF_NOT_SUCCESS(
      cuvidCtxLockDestroy((CUvideoctxlock)p),
      "Failed to create CUvideoctxlock.");
}

MapGuard::MapGuard(
    CUvideodecoder dec,
    CUVIDPROCPARAMS* proc_params,
    int picture_index)
    : decoder(dec) {
  TRACE_EVENT("nvdec", "cuvidMapVideoFrame");
  CHECK_CU(
      cuvidMapVideoFrame(decoder, picture_index, &frame, &pitch, proc_params),
      "Failed to map video frame.");
}

MapGuard::~MapGuard() {
  TRACE_EVENT("nvdec", "cuvidUnmapVideoFrame");
  auto status = cuvidUnmapVideoFrame(decoder, frame);

  if (status != CUDA_SUCCESS) {
    LOG(ERROR) << fmt::format(
        "Failed to unmap video frame ({}: {})",
        get_error_name(status),
        get_error_desc(status));
  }
}

} // namespace spdl::cuda::detail
