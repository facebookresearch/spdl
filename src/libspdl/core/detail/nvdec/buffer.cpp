/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/nvdec/buffer.h"

#include "libspdl/core/detail/logging.h"

#include <fmt/core.h>

namespace spdl::core {

CUDABufferTracker::CUDABufferTracker(
    const CUDAConfig& cuda_config,
    const std::vector<size_t>& shape)
    : buffer(cuda_buffer(shape, cuda_config)) {
  switch (shape.size()) {
    case 3:
      n = 1, c = shape[0], h = shape[1], w = shape[2];
      break;
    case 4:
      n = shape[0], c = shape[1], h = shape[2], w = shape[3];
      break;
    default:
      SPDL_FAIL_INTERNAL("Only 3D and 4D are supported.");
  }
}

uint8_t* CUDABufferTracker::get_next_frame() {
  if (i >= n) {
    SPDL_FAIL_INTERNAL(fmt::format(
        "Attempted to write beyond the maximum number of frames. max_frames={}, n={}",
        n,
        i));
  }
  return (uint8_t*)(buffer->data()) + i * c * h * w;
}

} // namespace spdl::core
