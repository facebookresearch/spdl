#pragma once

#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <deque>

namespace spdl::core {

using spdl::core::detail::AVCodecContextPtr;
using spdl::core::detail::FilterGraph;

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder<media_type>::Impl {
  PacketsPtr<media_type> packets;
  AVCodecContextPtr codec_ctx;
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
