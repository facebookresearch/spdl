#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/decoding.h"

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

} // namespace spdl::core
