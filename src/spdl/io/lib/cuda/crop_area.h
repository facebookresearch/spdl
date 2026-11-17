/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/cuda/types.h>

#include <fmt/format.h>

#include <climits>
#include <stdexcept>

namespace spdl::cuda {

// `CropArea` stores the offsets as `short`. Reject values that do not fit
// instead of letting them wrap to a different crop.
inline CropArea to_crop_area(int left, int top, int right, int bottom) {
  auto narrow = [](const char* name, int value) {
    if (value < SHRT_MIN || value > SHRT_MAX) {
      throw std::invalid_argument(
          fmt::format("`{}` is out of range: {}", name, value));
    }
    return static_cast<short>(value);
  };
  return CropArea{
      narrow("crop_left", left),
      narrow("crop_top", top),
      narrow("crop_right", right),
      narrow("crop_bottom", bottom)};
}

} // namespace spdl::cuda
