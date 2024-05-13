#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>

#include <vector>

namespace spdl::core {

template <MediaType media_type>
CPUBufferPtr convert_frames(
    const std::vector<const FFmpegFrames<media_type>*>& batch);

template <MediaType media_type>
CPUBufferPtr convert_frames(const FFmpegFrames<media_type>* frames) {
  const std::vector<const FFmpegFrames<media_type>*> batch{frames};
  // Use the same impl as batch conversion
  auto ret = convert_frames<media_type>(batch);
  ret->shape.erase(ret->shape.begin()); // Trim the batch dim
  return ret;
}

} // namespace spdl::core
