/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <optional>

namespace spdl::cuda {

/// CUDA memory allocator function type.
/// @param size Size in bytes to allocate.
/// @param device_index CUDA device index.
/// @param stream CUDA stream handle.
/// @return Pointer to allocated memory as uintptr_t.
using cuda_allocator_fn = std::function<uintptr_t(size_t, int, uintptr_t)>;

/// CUDA memory deleter function type.
/// @param ptr Pointer to memory to free as uintptr_t.
using cuda_deleter_fn = std::function<void(uintptr_t)>;

/// CUDA allocator pair (allocator and deleter functions).
using cuda_allocator = std::pair<cuda_allocator_fn, cuda_deleter_fn>;

/// Configuration for CUDA operations.
struct CUDAConfig {
  /// CUDA device index to use.
  int device_index;
  /// CUDA stream handle. Default is per-thread-default stream (0x2).
  uintptr_t stream = 0x2;
  /// Optional custom allocator for GPU memory.
  std::optional<cuda_allocator> allocator;
};

/// Rectangle for cropping operations.
struct CropArea {
  /// Left edge offset in pixels.
  short left = 0;
  /// Top edge offset in pixels.
  short top = 0;
  /// Right edge offset in pixels.
  short right = 0;
  /// Bottom edge offset in pixels.
  short bottom = 0;
};

} // namespace spdl::cuda
