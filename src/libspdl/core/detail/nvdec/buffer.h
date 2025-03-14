/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <libspdl/core/buffer.h>

#include <array>

namespace spdl::core {

/// Contiguous array data on a CUDA device.
/// This class is used to hold data decoded with NVDEC.
struct CUDABufferTracker {
  std::shared_ptr<CUDAStorage> storage;

  // ``i`` keeps track of how many frames are written.
  // ``i`` < ``n``;
  // ``i`` is updated by writer. (should we encapsulate it?)
  size_t n, c, h, w;

  size_t i{0};

  // For batch image / video
  CUDABufferTracker(
      std::shared_ptr<CUDAStorage>& storage,
      std::vector<size_t>& shape);

  // Get the pointer to the head of the next frame.
  uint8_t* get_next_frame();
};

} // namespace spdl::core
