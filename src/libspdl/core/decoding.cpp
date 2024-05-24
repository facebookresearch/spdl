#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/logging.h"

namespace spdl::core {

template <MediaType media_type>
  requires(media_type != MediaType::Image)
StreamingDecoder<media_type>::StreamingDecoder(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> cfg,
    const std::string filter_desc)
    : pImpl(new StreamingDecoder<media_type>::Impl(
          std::move(packets),
          std::move(cfg),
          std::move(filter_desc))) {}

template <MediaType media_type>
  requires(media_type != MediaType::Image)
std::optional<FFmpegFramesPtr<media_type>> StreamingDecoder<media_type>::decode(
    int num_frames) {
  return pImpl->decode(num_frames);
}

template class StreamingDecoder<MediaType::Video>;

#ifndef SPDL_USE_NVJPEG
CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    int cuda_device_index,
    const std::string& pix_fmt,
    const std::optional<std::string>& backend,
    const std::optional<cuda_allocator>& cuda_allocator) {
  SPDL_FAIL("SPDL is not compiled with NVJPEG support.");
}
#endif

} // namespace spdl::core
