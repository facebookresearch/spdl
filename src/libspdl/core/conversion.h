/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <vector>

namespace spdl::core {

// The actual implementation is found in
// detail/ffmpeg/conversion.cpp
template <MediaType media>
CPUBufferPtr convert_frames(
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage = nullptr);

template <MediaType media>
CPUBufferPtr convert_frames(
    const Frames<media>* frames,
    std::shared_ptr<CPUStorage> storage = nullptr) {
  const std::vector<const Frames<media>*> batch{frames};
  // Use the same impl as batch conversion
  auto ret = convert_frames<media>(batch, std::move(storage));
  ret->shape.erase(ret->shape.begin()); // Trim the batch dim
  return ret;
}

VideoFramesPtr convert_rgb_array(
    const void* data,
    size_t num_frames,
    size_t height,
    size_t width,
    Rational time_base,
    int pts);
} // namespace spdl::core
