#pragma once

#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/generator.h"

#include <deque>

namespace spdl::core {
namespace detail {
// Wraps AVCodecContextPtr and provide convenient methods
struct Decoder {
  AVCodecContextPtr codec_ctx;

  Decoder(
      AVCodecParameters*,
      Rational time_base,
      const std::optional<DecodeConfig>& cfg = std::nullopt);

  Generator<AVFrame*> decode(AVPacket*, bool flush_null = false);
};

} // namespace detail

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder<media_type>::Impl {
  PacketsPtr<media_type> packets;
  detail::Decoder decoder;
  detail::FilterGraph filter_graph;

  detail::Generator<detail::AVFramePtr> gen;
  Impl(
      PacketsPtr<media_type> packets,
      const std::optional<DecodeConfig> cfg,
      const std::string filter_desc);

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames);
};

} // namespace spdl::core
