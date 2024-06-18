#pragma once

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/generator.h"

namespace spdl::core::detail {

// Wraps AVCodecContextPtr and provide convenient methods
struct Decoder {
  AVCodecContextPtr codec_ctx;

  Decoder(
      AVCodecParameters*,
      Rational time_base,
      const std::optional<DecodeConfig>& cfg = std::nullopt);

  Generator<AVFramePtr> decode(AVPacket*, bool flush_null = false);
};

Generator<AVFramePtr> decode_packets(
    const std::vector<AVPacket*>& packets,
    Decoder& decoder,
    std::optional<FilterGraph>& filter);

} // namespace spdl::core::detail
