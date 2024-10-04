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
template <MediaType media_type>
CPUBufferPtr convert_frames(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    bool pin_memory = false);

template <MediaType media_type>
CPUBufferPtr convert_frames(
    const FFmpegFrames<media_type>* frames,
    bool pin_memory = false) {
  const std::vector<const FFmpegFrames<media_type>*> batch{frames};
  // Use the same impl as batch conversion
  auto ret = convert_frames<media_type>(batch, pin_memory);
  ret->shape.erase(ret->shape.begin()); // Trim the batch dim
  return ret;
}
} // namespace spdl::core
