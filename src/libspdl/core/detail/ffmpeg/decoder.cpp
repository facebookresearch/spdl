/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/decoder.h"

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

namespace spdl::core::detail {
namespace {
Generator<AVPacket*> _stream_packet(const std::vector<AVPacket*>& packets) {
  for (auto& packet : packets) {
    co_yield packet;
  }
  co_yield nullptr;
}

#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

Generator<AVFramePtr> _decode_packet(
    AVCodecContextPtr& codec_ctx,
    AVPacket* packet,
    bool flush_null) {
  VLOG(9)
      << ((!packet) ? fmt::format(" -- flush decoder")
                    : fmt::format(
                          "{:21s} {:.3f} ({})",
                          " -- packet:",
                          TS(packet, codec_ctx->pkt_timebase),
                          packet->pts));

  int errnum;

  {
    TRACE_EVENT("decoding", "avcodec_send_packet");
    errnum = avcodec_send_packet(codec_ctx.get(), packet);
  }
  if (errnum < 0) {
    LOG(WARNING) << av_error(errnum, "Failed to pass a frame to decoder.");
    co_return;
  }

  while (errnum >= 0) {
    auto frame = AVFramePtr{CHECK_AVALLOCATE(av_frame_alloc())};

    {
      TRACE_EVENT("decoding", "avcodec_receive_frame");
      errnum = avcodec_receive_frame(codec_ctx.get(), frame.get());
    }

    switch (errnum) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF: {
        if (flush_null) {
          co_yield nullptr;
        }
        co_return;
      }
      default: {
        if (errnum < 0) {
          LOG(WARNING) << av_error(errnum, "Failed to decode a packet.");
          co_return;
        }

        VLOG(9) << fmt::format(
            "{:21s} {:.3f} ({})",
            " --- raw frame:",
            TS(frame, codec_ctx->pkt_timebase),
            frame->pts);

        co_yield std::move(frame);
      }
    }
  }
}

#undef TS

Generator<AVFramePtr> decode_packets(
    AVCodecContextPtr& codec_ctx,
    const std::vector<AVPacket*>& packets,
    std::optional<FilterGraph>& filter) {
  auto packet_stream = _stream_packet(packets);
  if (!filter) {
    while (packet_stream) {
      auto decoding = _decode_packet(codec_ctx, packet_stream(), false);
      while (decoding) {
        co_yield decoding();
      }
    }
  } else {
    while (packet_stream) {
      auto packet = packet_stream();
      auto decoding = _decode_packet(codec_ctx, packet, !packet);
      while (decoding) {
        auto frame = decoding();
        auto filtering = filter->filter(frame.get());
        while (filtering) {
          co_yield filtering();
        }
      }
    }
  }
}
} // namespace

////////////////////////////////////////////////////////////////////////////////
// DecoderImpl
////////////////////////////////////////////////////////////////////////////////

template <MediaType media_type>
DecoderImpl<media_type>::DecoderImpl(
    const Codec<media_type>& codec,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc)
    : codec_ctx(get_decode_codec_ctx_ptr(
          codec.get_parameters(),
          codec.time_base,
          cfg ? cfg->decoder : std::nullopt,
          cfg ? cfg->decoder_options : std::nullopt)),
      filter_graph(get_filter<media_type>(
          codec_ctx.get(),
          filter_desc,
          codec.frame_rate)) {}

template <MediaType media_type>
Rational DecoderImpl<media_type>::get_output_time_base() const {
  if (filter_graph) {
    return filter_graph->get_sink_time_base();
  }
  return {codec_ctx->time_base.num, codec_ctx->time_base.den};
}

template <MediaType media_type>
FFmpegFramesPtr<media_type> DecoderImpl<media_type>::decode_and_flush(
    PacketsPtr<media_type> packets,
    int num_frames) {
  auto ret = std::make_unique<FFmpegFrames<media_type>>(
      packets->id, get_output_time_base());
  auto gen = decode_packets(codec_ctx, packets->get_packets(), filter_graph);
  int num_yielded = 0;
  while (gen) {
    ret->push_back(gen().release());
    num_yielded += 1;
    if (num_frames > 0 && num_yielded >= num_frames) {
      break;
    }
  }
  return ret;
}

template class DecoderImpl<MediaType::Audio>;
template class DecoderImpl<MediaType::Video>;
template class DecoderImpl<MediaType::Image>;
} // namespace spdl::core::detail
