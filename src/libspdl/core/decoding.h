#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>
#include <string>

namespace spdl::core {

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder {
  struct Impl;
  Impl* pImpl;

  StreamingDecoder(
      PacketsPtr<media_type> packets,
      const std::optional<DecodeConfig> cfg = std::nullopt,
      const std::string filter_desc = {});

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames);
};

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> cfg = std::nullopt,
    const std::string filter_desc = {});

template <MediaType media_type>
CUDABufferPtr decode_packets_nvdec(
    PacketsPtr<media_type> packets,
    const int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator);

CUDABufferPtr decode_packets_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    const int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    bool strict,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator);

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    int cuda_device_index,
    const std::string& pix_fmt,
    const std::optional<cuda_allocator>& cuda_allocator);

} // namespace spdl::core
