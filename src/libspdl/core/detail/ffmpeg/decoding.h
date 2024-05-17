#pragma once

#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

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

  void add_packet(AVPacket*);
  std::vector<AVFramePtr> get_frames(bool flush_null);
};

} // namespace detail

using spdl::core::detail::AVCodecContextPtr;
using spdl::core::detail::FilterGraph;

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder<media_type>::Impl {
  PacketsPtr<media_type> packets;
  detail::Decoder decoder;
  std::optional<FilterGraph> filter_graph;

  std::deque<FFmpegFramesPtr<media_type>> carry_overs;
  int packet_index = 0;
  Impl(
      PacketsPtr<media_type> packets,
      const std::optional<DecodeConfig> cfg,
      const std::string filter_desc);

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames);
};

} // namespace spdl::core
