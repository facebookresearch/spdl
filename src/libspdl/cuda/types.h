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

using cuda_allocator_fn = std::function<uintptr_t(size_t, int, uintptr_t)>;
using cuda_deleter_fn = std::function<void(uintptr_t)>;
using cuda_allocator = std::pair<cuda_allocator_fn, cuda_deleter_fn>;

struct CUDAConfig {
  int device_index;
  uintptr_t stream = 0;
  std::optional<cuda_allocator> allocator;
};

struct CropArea {
  short left = 0;
  short top = 0;
  short right = 0;
  short bottom = 0;
};

} // namespace spdl::cuda
