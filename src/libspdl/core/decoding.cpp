#include <libspdl/core/decoding.h>
#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/decoder.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/nvdec/decoding.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

namespace spdl::core {
template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc) {
  TRACE_EVENT(
      "decoding",
      "decode_packets_ffmpeg",
      perfetto::Flow::ProcessScoped(packets->id));
  detail::Decoder decoder{packets->codecpar, packets->time_base, cfg};
  if constexpr (media_type != MediaType::Image) {
    packets->push(nullptr); // For flushing
  }
  auto filter = detail::get_filter<media_type>(
      decoder.codec_ctx.get(), filter_desc, packets->frame_rate);
  auto ret = std::make_unique<FFmpegFrames<media_type>>(
      packets->id, packets->time_base);

  auto gen = detail::decode_packets(packets->get_packets(), decoder, filter);
  while (gen) {
    ret->push_back(gen().release());
  }
  if (filter) {
    ret->time_base = filter->get_sink_time_base();
  }
  return ret;
}

template FFmpegAudioFramesPtr decode_packets_ffmpeg(
    AudioPacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc);

template FFmpegVideoFramesPtr decode_packets_ffmpeg(
    VideoPacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc);

template FFmpegImageFramesPtr decode_packets_ffmpeg(
    ImagePacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc);

#ifdef SPDL_USE_NVCODEC
namespace {
void validate_nvdec_params(
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height) {
  if (cuda_device_index < 0) {
    SPDL_FAIL(fmt::format(
        "cuda_device_index must be non-negative. Found: {}",
        cuda_device_index));
  }
  if (crop.left < 0) {
    SPDL_FAIL(
        fmt::format("crop.left must be non-negative. Found: {}", crop.left));
  }
  if (crop.top < 0) {
    SPDL_FAIL(
        fmt::format("crop.top must be non-negative. Found: {}", crop.top));
  }
  if (crop.right < 0) {
    SPDL_FAIL(
        fmt::format("crop.right must be non-negative. Found: {}", crop.right));
  }
  if (crop.bottom < 0) {
    SPDL_FAIL(fmt::format(
        "crop.bottom must be non-negative. Found: {}", crop.bottom));
  }
  if (width > 0 && width % 2) {
    SPDL_FAIL(fmt::format("width must be positive and even. Found: {}", width));
  }
  if (height > 0 && height % 2) {
    SPDL_FAIL(
        fmt::format("height must be positive and even. Found: {}", height));
  }
}
} // namespace
#endif

template <MediaType media_type>
CUDABufferPtr decode_packets_nvdec(
    PacketsPtr<media_type> packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt) {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  validate_nvdec_params(cuda_config.device_index, crop, width, height);
  if constexpr (media_type == MediaType::Video) {
    packets = apply_bsf(std::move(packets));
  }
  return detail::decode_nvdec<media_type>(
      std::move(packets), cuda_config, crop, width, height, pix_fmt);
#endif
}

template CUDABufferPtr decode_packets_nvdec(
    PacketsPtr<MediaType::Video> packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt);

template CUDABufferPtr decode_packets_nvdec(
    PacketsPtr<MediaType::Image> packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt);

CUDABufferPtr decode_packets_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    bool strict) {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  validate_nvdec_params(cuda_config.device_index, crop, width, height);
  return detail::decode_nvdec(
      std::move(packets), cuda_config, crop, width, height, pix_fmt, strict);
#endif
}

#ifdef SPDL_USE_NVJPEG
namespace detail {
CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt);
CUDABufferPtr decode_image_nvjpeg(
    const std::vector<std::string_view>& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt);
} // namespace detail
#endif

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt) {
#ifndef SPDL_USE_NVJPEG
  SPDL_FAIL("SPDL is not compiled with NVJPEG support.");
#else
  return detail::decode_image_nvjpeg(
      data, cuda_config, scale_width, scale_height, pix_fmt);
#endif
}

CUDABufferPtr decode_image_nvjpeg(
    const std::vector<std::string_view>& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt) {
#ifndef SPDL_USE_NVJPEG
  SPDL_FAIL("SPDL is not compiled with NVJPEG support.");
#else
  return detail::decode_image_nvjpeg(
      data, cuda_config, scale_width, scale_height, pix_fmt);
#endif
}

} // namespace spdl::core
