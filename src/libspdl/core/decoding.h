#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>
#include <string>

namespace spdl::core {

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder;

template <MediaType media_type>
using DecoderPtr = std::unique_ptr<StreamingDecoder<media_type>>;

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder {
  struct Impl;
  Impl* pImpl;

  StreamingDecoder(
      PacketsPtr<media_type> packets,
      const std::optional<DecodeConfig> cfg = std::nullopt,
      const std::string filter_desc = {});

  ~StreamingDecoder();

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames);
};

template <MediaType media_type>
DecoderPtr<media_type> make_decoder(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::string& filter_desc);

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> cfg = std::nullopt,
    const std::string filter_desc = {});

template <MediaType media_type>
CUDABufferPtr decode_packets_nvdec(
    PacketsPtr<media_type> packets,
    const CUDAConfig cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt);

CUDABufferPtr decode_packets_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    const CUDAConfig cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    bool strict);

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig cuda_config,
    const std::string& pix_fmt);

} // namespace spdl::core
