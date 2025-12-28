/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/types.h>
#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/color_conversion.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/cuda/detail/color_conversion.h"
#include "libspdl/cuda/detail/utils.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <vector>

namespace spdl::cuda {

using Nv2Converter =
    void (*)(CUstream, uint8_t*, int, uint8_t*, int, int, int, int, int);

namespace {
template <Nv2Converter Fn>
CUDABufferPtr nv12_to_rgb(
    const CUDABuffer& nv12_batch,
    const CUDAConfig& cfg,
    int matrix_coefficients,
    bool sync) {
  if (nv12_batch.shape.size() != 3) {
    SPDL_FAIL(
        fmt::format(
            "Expected 3D buffer [num_frames, h*1.5, width]. Found: {}D",
            nv12_batch.shape.size()));
  }

  size_t num_frames = nv12_batch.shape[0];

  auto h2 = nv12_batch.shape[1];
  auto width = nv12_batch.shape[2];

  if (h2 % 3 != 0) {
    SPDL_FAIL(
        fmt::format(
            "The height of NV12 image (h*1.5) must be divisible by 3. Found: {}",
            h2));
  }
  auto height = h2 / 3 * 2;

  auto ret = cuda_buffer({num_frames, 3, height, width}, cfg);

  auto* src = (uint8_t*)nv12_batch.data();
  auto* dst = (uint8_t*)ret->data();

  Fn((CUstream)cfg.stream,
     src,
     (int)width,
     dst,
     (int)width,
     (int)width,
     (int)height,
     (int)num_frames,
     matrix_coefficients);

  if (sync) {
    CHECK_CUDA(
        cudaStreamSynchronize((cudaStream_t)cfg.stream),
        "Failed to synchronize stream after batched color conversion.");
  }

  return ret;
}
} // namespace

CUDABufferPtr nv12_to_planar_rgb(
    const CUDABuffer& nv12_batch,
    const CUDAConfig& cfg,
    int matrix_coefficients,
    bool sync) {
  return nv12_to_rgb<detail::nv12_to_planar_rgb>(
      nv12_batch, cfg, matrix_coefficients, sync);
}

CUDABufferPtr nv12_to_planar_bgr(
    const CUDABuffer& nv12_batch,
    const CUDAConfig& cfg,
    int matrix_coefficients,
    bool sync) {
  return nv12_to_rgb<detail::nv12_to_planar_bgr>(
      nv12_batch, cfg, matrix_coefficients, sync);
}
} // namespace spdl::cuda
