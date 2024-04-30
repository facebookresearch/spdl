#include "libspdl/core/detail/ffmpeg/decoding.h"

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

namespace spdl::core::detail {
////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
namespace {

#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

folly::coro::AsyncGenerator<AVFramePtr&&>
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
  while (errnum >= 0) {
    co_await folly::coro::co_safe_point;
    AVFramePtr frame{CHECK_AVALLOCATE(av_frame_alloc())};
    {
      TRACE_EVENT("decoding", "avcodec_receive_frame");
      errnum = avcodec_receive_frame(codec_ctx, frame.get());
    }
    switch (errnum) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF:
        if (flush_null) {
          co_yield nullptr;
        }
        co_return;
      default: {
        if (frame->key_frame) {
          TRACE_EVENT_INSTANT("decoding", "key_frame");
        }
        CHECK_AVERROR_NUM(errnum, "Failed to decode a frame.");

        double ts = TS(frame, codec_ctx->pkt_timebase);
        XLOG(DBG9) << fmt::format(
            "{:21s} {:.3f} ({})", " --- raw frame:", ts, frame->pts);

        co_yield std::move(frame);
      }
    }
  }
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
FFmpegFramesPtr<media_type> get_frame(PacketsPtr<media_type>& packets) {
  return std::make_unique<FFmpegFrames<media_type>>(
      packets->id, packets->time_base);
}

template <MediaType media_type>
folly::coro::Task<FFmpegFramesPtr<media_type>> decode_pkts_with_filter(
    PacketsPtr<media_type> packets,
    AVCodecContextPtr codec_ctx,
    std::string filter_desc) {
  auto frames = get_frame(packets);
  auto filter =
      get_filter<media_type>(codec_ctx.get(), filter_desc, packets->frame_rate);
  for (auto& packet : packets->get_packets()) {
    co_await folly::coro::co_safe_point;
    auto decoding = decode_packet(codec_ctx.get(), packet, true);
    while (auto raw_frame = co_await decoding.next()) {
      co_await folly::coro::co_safe_point;
      auto filtering = filter_frame(filter, *raw_frame);
      while (auto filtered_frame = co_await filtering.next()) {
        co_await folly::coro::co_safe_point;
        frames->push_back(filtered_frame->release());
      }
    }
  }
  frames->time_base = filter.get_sink_time_base();
  co_return std::move(frames);
}

template <MediaType media_type>
folly::coro::Task<FFmpegFramesPtr<media_type>> decode_pkts(
    PacketsPtr<media_type> packets,
    AVCodecContextPtr codec_ctx) {
  auto frames = get_frame(packets);
  for (auto& packet : packets->get_packets()) {
    co_await folly::coro::co_safe_point;
    auto decoding = decode_packet(codec_ctx.get(), packet, false);
    while (auto frame = co_await decoding.next()) {
      co_await folly::coro::co_safe_point;
      frames->push_back(frame->release());
    }
  }
  co_return std::move(frames);
}

} // namespace

template <MediaType media_type>
folly::coro::Task<FFmpegFramesPtr<media_type>> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc) {
  TRACE_EVENT(
      "decoding",
      "decode_packets_ffmpeg",
      perfetto::Flow::ProcessScoped(packets->id));
  auto codec_ctx = get_codec_ctx_ptr(
      packets->codecpar,
      AVRational{packets->time_base.num, packets->time_base.den},
      cfg ? cfg->decoder : std::nullopt,
      cfg ? cfg->decoder_options : std::nullopt);
  if constexpr (media_type != MediaType::Image) {
    packets->push(nullptr); // For flushing
  }
  if (filter_desc.empty()) {
    co_return co_await decode_pkts(std::move(packets), std::move(codec_ctx));
  }
  co_return co_await decode_pkts_with_filter(
      std::move(packets), std::move(codec_ctx), std::move(filter_desc));
}

template folly::coro::Task<FFmpegAudioFramesPtr> decode_packets_ffmpeg(
    AudioPacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

template folly::coro::Task<FFmpegVideoFramesPtr> decode_packets_ffmpeg(
    VideoPacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

template folly::coro::Task<FFmpegImageFramesPtr> decode_packets_ffmpeg(
    ImagePacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

} // namespace spdl::core::detail
