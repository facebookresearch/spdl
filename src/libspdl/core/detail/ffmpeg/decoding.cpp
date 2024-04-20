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

folly::coro::AsyncGenerator<AVFramePtr&&> filter_frame(
    AVFrame* frame,
    AVFilterContext* src_ctx,
    AVFilterContext* sink_ctx) {
  int errnum;
  {
    TRACE_EVENT("decoding", "av_buffersrc_add_frame_flags");
    errnum = av_buffersrc_add_frame_flags(
        src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
  }

  CHECK_AVERROR_NUM(errnum, "Failed to pass a frame to filter.");

  AVFrameAutoUnref frame_ref{frame};
  while (errnum >= 0) {
    co_await folly::coro::co_safe_point;
    AVFramePtr frame2{CHECK_AVALLOCATE(av_frame_alloc())};
    {
      TRACE_EVENT("decoding", "av_buffersrc_get_frame");
      errnum = av_buffersink_get_frame(sink_ctx, frame2.get());
    }
    switch (errnum) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF:
        co_return;
      default: {
        CHECK_AVERROR_NUM(errnum, "Failed to filter a frame.");
        co_yield std::move(frame2);
      }
    }
  }
}

folly::coro::AsyncGenerator<AVFramePtr&&> decode_packet(
    AVCodecContext* codec_ctx,
    AVPacket* packet) {
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
        co_yield nullptr;
        co_return;
      default: {
        if (frame->key_frame) {
          TRACE_EVENT_INSTANT("decoding", "key_frame");
        }
        CHECK_AVERROR_NUM(errnum, "Failed to decode a frame.");
        co_yield std::move(frame);
      }
    }
  }
}

template <MediaType media_type>
folly::coro::Task<void> decode_pkts(
    PacketsPtr<media_type> packets,
    AVCodecContextPtr codec_ctx,
    FFmpegFrames<media_type>* frames) {
  for (auto& packet : packets->get_packets()) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto frame = co_await decoding.next()) {
      co_await folly::coro::co_safe_point;
      AVFramePtr f = *frame;
      if (f) {
        double ts = TS(f, codec_ctx->pkt_timebase);
        XLOG(DBG9) << fmt::format(
            "{:21s} {:.3f} ({})", " --- raw frame:", ts, f->pts);
        frames->push_back(f.release());
      }
    }
  }
}

template <MediaType media_type>
folly::coro::Task<void> decode_pkts_with_filter(
    PacketsPtr<media_type> packets,
    AVCodecContextPtr codec_ctx,
    std::string filter_desc,
    FFmpegFrames<media_type>* frames) {
  auto filter_graph = [&]() {
    if constexpr (media_type == MediaType::Audio) {
      return get_audio_filter(filter_desc, codec_ctx.get());
    }
    if constexpr (media_type == MediaType::Video) {
      return get_video_filter(
          filter_desc, codec_ctx.get(), packets->frame_rate);
    }
    if constexpr (media_type == MediaType::Image) {
      return get_video_filter(filter_desc, codec_ctx.get());
    }
  }();
  AVFilterContext* src_ctx = filter_graph->filters[0];
  AVFilterContext* sink_ctx = filter_graph->filters[1];
  assert(strcmp(src_ctx->name, "in") == 0);
  assert(strcmp(sink_ctx->name, "out") == 0);

  XLOG(DBG5) << describe_graph(filter_graph.get());

  for (auto& packet : packets->get_packets()) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto raw_frame = co_await decoding.next()) {
      co_await folly::coro::co_safe_point;
      AVFramePtr rf = *raw_frame;

      XLOG(DBG9)
          << (rf ? fmt::format(
                       "{:21s} {:.3f} ({})",
                       " --- raw frame:",
                       TS(rf, src_ctx->outputs[0]->time_base),
                       rf->pts)
                 : fmt::format(" --- flush filter graph"));

      auto filtering = filter_frame(rf.get(), src_ctx, sink_ctx);
      while (auto frame = co_await filtering.next()) {
        AVFramePtr f = *frame;
        if (f) {
          double ts = TS(f, sink_ctx->inputs[0]->time_base);
          XLOG(DBG9) << fmt::format(
              "{:21s} {:.3f} ({})", " ---- filtered frame:", ts, f->pts);
          frames->push_back(f.release());
        }
      }
    }
  }
}
} // namespace

template <MediaType media_type>
folly::coro::Task<FFmpegFramesPtr<media_type>> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& cfg,
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
  auto frames = std::make_unique<FFmpegFrames<media_type>>(
      packets->id, packets->time_base);

  if constexpr (media_type != MediaType::Image) {
    packets->push(nullptr); // For flushing
  }
  if (filter_desc.empty()) {
    co_await decode_pkts(
        std::move(packets), std::move(codec_ctx), frames.get());
  } else {
    co_await decode_pkts_with_filter(
        std::move(packets),
        std::move(codec_ctx),
        std::move(filter_desc),
        frames.get());
  }
  co_return std::move(frames);
}

template folly::coro::Task<FFmpegAudioFramesPtr> decode_packets_ffmpeg(
    AudioPacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    std::string filter_desc);

template folly::coro::Task<FFmpegVideoFramesPtr> decode_packets_ffmpeg(
    VideoPacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    std::string filter_desc);

template folly::coro::Task<FFmpegImageFramesPtr> decode_packets_ffmpeg(
    ImagePacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    std::string filter_desc);

} // namespace spdl::core::detail
