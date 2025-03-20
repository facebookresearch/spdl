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

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <vector>

namespace spdl::cuda {
namespace {
void validate_shape_consistentcy(const std::vector<CUDABuffer>& frames) {
  if (!frames.size()) {
    SPDL_FAIL("The input must have at least one frame.");
  }
  const auto& f0 = frames.at(0);
  for (size_t i = 0; i < frames.size(); ++i) {
    const auto& f = frames.at(i);
    if (f.shape.size() != 2) {
      SPDL_FAIL(
          fmt::format("The input frame must be 2D. Found: {}", f.shape.size()));
    }
    if (f.depth != sizeof(uint8_t) ||
        f.elem_class != spdl::core::ElemClass::UInt) {
      SPDL_FAIL(fmt::format("The input must be uint8 type."));
    }

    if (f.shape != f0.shape) {
      SPDL_FAIL(fmt::format(
          "The shape of the buffer does not match. Found [{}] at 0, and [{}] at {}",
          fmt::join(f0.shape, ", "),
          fmt::join(f.shape, ", "),
          i));
    }

    if (f.device_index != f0.device_index) {
      SPDL_WARN(fmt::format(
          "The frames are in different devices. Frame 0 is on device {} and Frame {} is oon device {}",
          f0.device_index,
          i,
          f.device_index));
    }
  }
}

using Nv2Converter =
    void (*)(CUstream, uint8_t*, int, uint8_t*, int, int, int, int);

template <Nv2Converter Fn>
CUDABufferPtr nv12_to_rgb(
    const std::vector<CUDABuffer>& frames,
    const CUDAConfig& cfg,
    int matrix_coefficients) {
  validate_shape_consistentcy(frames);
  const auto& f0 = frames[0];
  auto height = f0.shape[0], width = f0.shape[1];
  if (height % 3) {
    SPDL_FAIL(fmt::format(
        "The height of NV12 image must be divisble by 3. Found: {}", height));
  }
  auto h0 = height / 3 * 2;

  auto ret = cuda_buffer({frames.size(), 3, h0, width}, cfg);

  auto* dst = (uint8_t*)ret->data();
  for (auto& frame : frames) {
    Fn((CUstream)cfg.stream,
       (uint8_t*)frame.data(),
       width,
       dst,
       width, // pitch
       width,
       h0,
       matrix_coefficients);
    dst += 3 * h0 * width;
  }
  return ret;
}

} // namespace

CUDABufferPtr nv12_to_planar_rgb(
    const std::vector<CUDABuffer>& frames,
    const CUDAConfig& cfg,
    int matrix_coefficients) {
  return nv12_to_rgb<detail::nv12_to_planar_rgb>(
      frames, cfg, matrix_coefficients);
}

CUDABufferPtr nv12_to_planar_bgr(
    const std::vector<CUDABuffer>& frames,
    const CUDAConfig& cfg,
    int matrix_coefficients) {
  return nv12_to_rgb<detail::nv12_to_planar_bgr>(
      frames, cfg, matrix_coefficients);
}
}; // namespace spdl::cuda
