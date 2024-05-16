#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
namespace detail {
namespace {

#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

std::vector<AVFramePtr>
decode_packet(AVCodecContext* codec_ctx, AVPacket* packet, bool flush_null) {
  assert(codec_ctx);
  XLOG(DBG9)
      << ((!packet) ? fmt::format(" -- flush decoder")
                    : fmt::format(
                          "{:21s} {:.3f} ({})",
                          " -- packet:",
                          TS(packet, codec_ctx->pkt_timebase),
                          packet->pts));

  int errnum;
  {
    TRACE_EVENT("decoding", "avcodec_send_packet");
    errnum = avcodec_send_packet(codec_ctx, packet);
  }
  std::vector<AVFramePtr> ret;
  while (errnum >= 0) {
    AVFramePtr frame{CHECK_AVALLOCATE(av_frame_alloc())};
    {
      TRACE_EVENT("decoding", "avcodec_receive_frame");
      errnum = avcodec_receive_frame(codec_ctx, frame.get());
    }
    switch (errnum) {
      case AVERROR(EAGAIN):
        break;
      case AVERROR_EOF:
        if (flush_null) {
          ret.emplace_back(AVFramePtr{nullptr});
        }
        break;
      default: {
        if (frame->key_frame) {
          TRACE_EVENT_INSTANT("decoding", "key_frame");
        }
        CHECK_AVERROR_NUM(errnum, "Failed to decode a frame.");

        double ts = TS(frame, codec_ctx->pkt_timebase);
        XLOG(DBG9) << fmt::format(
            "{:21s} {:.3f} ({})", " --- raw frame:", ts, frame->pts);

        ret.emplace_back(frame.release());
      }
    }
  }
  return ret;
}

template <MediaType media_type>
FilterGraph get_filter(
    AVCodecContext* codec_ctx,
    const std::string& filter_desc,
    std::optional<Rational> frame_rate) {
  if constexpr (media_type == MediaType::Audio) {
    return get_audio_filter(filter_desc, codec_ctx);
  }
  if constexpr (media_type == MediaType::Video) {
    return get_video_filter(filter_desc, codec_ctx, *frame_rate);
  }
  if constexpr (media_type == MediaType::Image) {
    return get_image_filter(filter_desc, codec_ctx);
  }
}

template <MediaType media_type>
FFmpegFramesPtr<media_type> get_frame(DemuxedPackets<media_type>* packets) {
  return std::make_unique<FFmpegFrames<media_type>>(
      packets->id, packets->time_base);
}

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_with_filter(
    DemuxedPackets<media_type>* packets,
    AVCodecContext* codec_ctx,
    FilterGraph& filter) {
  auto frames = get_frame(packets);
  for (auto& packet : packets->get_packets()) {
    for (auto& raw_frame : decode_packet(codec_ctx, packet, true)) {
      for (auto& filtered_frame : filter_frame(filter, raw_frame.get())) {
        frames->push_back(filtered_frame.release());
      }
    }
  }
  frames->time_base = filter.get_sink_time_base();
  return frames;
}

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets(
    DemuxedPackets<media_type>* packets,
    AVCodecContext* codec_ctx) {
  auto frames = get_frame(packets);
  for (auto& packet : packets->get_packets()) {
    for (auto& frame : decode_packet(codec_ctx, packet, false)) {
      frames->push_back(frame.release());
    }
  }
  return frames;
}

template <MediaType media_type>
AVCodecContextPtr get_decode_ctx(
    PacketsPtr<media_type>& packets,
    const std::optional<DecodeConfig>& cfg) {
  return detail::get_codec_ctx_ptr(
      packets->codecpar,
      AVRational{packets->time_base.num, packets->time_base.den},
      cfg ? cfg->decoder : std::nullopt,
      cfg ? cfg->decoder_options : std::nullopt);
}

} // namespace
} // namespace detail

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc) {
  TRACE_EVENT(
      "decoding",
      "decode_packets_ffmpeg",
      perfetto::Flow::ProcessScoped(packets->id));
  auto codec_ctx = detail::get_decode_ctx(packets, cfg);
  if constexpr (media_type != MediaType::Image) {
    packets->push(nullptr); // For flushing
  }
  if (filter_desc.empty()) {
    return detail::decode_packets(packets.get(), codec_ctx.get());
  } else {
    auto filter = detail::get_filter<media_type>(
        codec_ctx.get(), filter_desc, packets->frame_rate);
    return detail::decode_packets_with_filter(
        packets.get(), codec_ctx.get(), filter);
  }
}

template FFmpegAudioFramesPtr decode_packets_ffmpeg(
    AudioPacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

template FFmpegVideoFramesPtr decode_packets_ffmpeg(
    VideoPacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

template FFmpegImageFramesPtr decode_packets_ffmpeg(
    ImagePacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

} // namespace spdl::core
