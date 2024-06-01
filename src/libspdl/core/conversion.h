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

CUDABufferPtr convert_to_cuda(
    CPUBufferPtr buffer,
    int cuda_device_index,
    uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& allocator);

CPUStorage cp_to_cpu(const void* src, const std::vector<size_t> shape);

} // namespace spdl::core
