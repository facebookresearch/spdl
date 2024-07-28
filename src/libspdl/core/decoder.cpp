/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/decoder.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core {
/////////////////////////////////////////////////////////////////////////////////
// StreamingDecoder::Impl
/////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder<media_type>::Impl {
  PacketsPtr<media_type> packets;
  detail::Decoder decoder;
  std::optional<detail::FilterGraph> filter_graph;

  detail::Generator<detail::AVFramePtr> gen;
  Impl(
      PacketsPtr<media_type> packets_,
      const std::optional<DecodeConfig>& cfg_,
      const std::optional<std::string>& filter_desc_)
      : packets(std::move(packets_)),
        decoder(packets->codecpar, packets->time_base, cfg_),
        filter_graph(detail::get_filter<media_type>(
            decoder.codec_ctx.get(),
            filter_desc_,
            packets->frame_rate)),
        gen(detail::decode_packets(
            packets->get_packets(),
            decoder,
            filter_graph)) {}

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames) {
    if (num_frames <= 0) {
      SPDL_FAIL("the `num_frames` must be positive.");
    }

    if (!gen) {
      return {};
    }

    TRACE_EVENT("decoding", "StreamingDecoder::decode");
    auto ret = std::make_unique<FFmpegFrames<media_type>>(
        packets->id, packets->time_base);
    for (int i = 0; gen && (i < num_frames); ++i) {
      ret->push_back(gen().release());
    }
    return ret;
  }
};

/////////////////////////////////////////////////////////////////////////////////
// StreamingDecoder
/////////////////////////////////////////////////////////////////////////////////

template <MediaType media_type>
  requires(media_type != MediaType::Image)
StreamingDecoder<media_type>::StreamingDecoder(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc)
    : pImpl(new StreamingDecoder<media_type>::Impl(
          std::move(packets),
          cfg,
          filter_desc)) {}

template <MediaType media_type>
  requires(media_type != MediaType::Image)
StreamingDecoder<media_type>::~StreamingDecoder() {
  TRACE_EVENT("decoding", "StreamingDecoder::~StreamingDecoder");
  delete pImpl;
}

template <MediaType media_type>
DecoderPtr<media_type> make_decoder(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::optional<std::string>& filter_desc) {
  TRACE_EVENT("decoding", "make_decoder");
  return std::make_unique<spdl::core::StreamingDecoder<media_type>>(
      std::move(packets), decode_cfg, filter_desc);
}

template DecoderPtr<MediaType::Video> make_decoder(
    PacketsPtr<MediaType::Video> packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::optional<std::string>& filter_desc);

template <MediaType media_type>
  requires(media_type != MediaType::Image)
std::optional<FFmpegFramesPtr<media_type>> StreamingDecoder<media_type>::decode(
    int num_frames) {
  return pImpl->decode(num_frames);
}

template struct StreamingDecoder<MediaType::Video>;

} // namespace spdl::core
