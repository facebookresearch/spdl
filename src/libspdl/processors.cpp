#include <fmt/core.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/logging/xlog.h>

#include <libspdl/detail/demuxer.h>
#include <libspdl/detail/executors.h>
#include <libspdl/detail/ffmpeg/ctx_utils.h>
#include <libspdl/detail/ffmpeg/filter_graph.h>
#include <libspdl/detail/ffmpeg/logging.h>
#include <libspdl/detail/packets.h>
#include <libspdl/logging.h>
#include <libspdl/processors.h>
#include <cstddef>
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
}

using namespace spdl::detail;

template <typename T>
using Task = folly::coro::Task<T>;

template <typename T>
using Generator = folly::coro::AsyncGenerator<T>;

using folly::coro::collectAllTryRange;

namespace spdl {
namespace {
//////////////////////////////////////////////////////////////////////////////
// Decoder and filter
//////////////////////////////////////////////////////////////////////////////

#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

Generator<AVFramePtr&&> filter_frame(
    AVFrame* frame,
    AVFilterContext* src_ctx,
    AVFilterContext* sink_ctx) {
  int errnum =
      av_buffersrc_add_frame_flags(src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);

  CHECK_AVERROR_NUM(errnum, "Failed to pass a frame to filter.");

  AVFrameAutoUnref frame_ref{frame};
  while (errnum >= 0) {
    AVFramePtr frame2{CHECK_AVALLOCATE(av_frame_alloc())};
    errnum = av_buffersink_get_frame(sink_ctx, frame2.get());
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

Generator<AVFramePtr&&> decode_packet(
    AVCodecContext* codec_ctx,
    AVPacket* packet) {
  assert(codec_ctx);
  XLOG(DBG)
      << ((!packet) ? fmt::format(" -- flush decoder")
                    : fmt::format(
                          "{:21s} {:.3f} ({})",
                          " -- packet:",
                          TS(packet, codec_ctx->pkt_timebase),
                          packet->pts));

  int errnum = avcodec_send_packet(codec_ctx, packet);
  while (errnum >= 0) {
    AVFramePtr frame{CHECK_AVALLOCATE(av_frame_alloc())};
    errnum = avcodec_receive_frame(codec_ctx, frame.get());
    switch (errnum) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF:
        co_yield nullptr;
        co_return;
      default: {
        CHECK_AVERROR_NUM(errnum, "Failed to decode a frame.");
        co_yield std::move(frame);
      }
    }
  }
}

Task<FrameContainer> decode_packets(
    PackagedAVPackets packets,
    AVCodecContextPtr codec_ctx) {
  auto [start, end] = packets.timestamp;
  FrameContainer frames{
      codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO ? MediaType::Audio
                                                  : MediaType::Video};
  for (auto& packet : packets.packets) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto frame = co_await decoding.next()) {
      AVFramePtr f = *frame;
      if (f) {
        double ts = TS(f, codec_ctx->pkt_timebase);
        if (start <= ts && ts <= end) {
          XLOG(DBG) << fmt::format(
              "{:21s} {:.3f} ({})", " --- raw frame:", ts, f->pts);
          frames.frames.push_back(f.release());
        }
      }
    }
  }
  co_return frames;
}

Task<FrameContainer> decode_packets(
    PackagedAVPackets packets,
    AVCodecContextPtr codec_ctx,
    AVFilterGraphPtr filter_graph) {
  // XLOG(DBG) << describe_graph(filter_graph.get());

  auto [start, end] = packets.timestamp;

  assert(filter_graph);
  AVFilterContext* src_ctx = filter_graph->filters[0];
  AVFilterContext* sink_ctx = filter_graph->filters[1];
  assert(strcmp(src_ctx->name, "in") == 0);
  assert(strcmp(sink_ctx->name, "out") == 0);

  FrameContainer frames{get_output_media_type(filter_graph.get())};
  for (auto& packet : packets.packets) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto raw_frame = co_await decoding.next()) {
      AVFramePtr rf = *raw_frame;

      XLOG(DBG)
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
          if (start <= ts && ts <= end) {
            XLOG(DBG) << fmt::format(
                "{:21s} {:.3f} ({})", " ---- filtered frame:", ts, f->pts);

            frames.frames.push_back(f.release());
          }
        }
      }
    }
  }
  co_return frames;
}

Task<FrameContainer> get_decode_task(
    PackagedAVPackets packets,
    const std::string& filter_desc,
    const DecodeConfig& cfg) {
  auto codec_ctx = get_codec_ctx(
      packets.codecpar,
      packets.time_base,
      cfg.decoder,
      cfg.decoder_options,
      cfg.cuda_device_index);

  if (filter_desc.empty()) {
    return decode_packets(std::move(packets), std::move(codec_ctx));
  }

  auto filter_graph = [&]() {
    switch (codec_ctx->codec_type) {
      case AVMEDIA_TYPE_AUDIO:
        return get_audio_filter(filter_desc, codec_ctx.get());
      case AVMEDIA_TYPE_VIDEO:
        return get_video_filter(
            filter_desc, codec_ctx.get(), packets.frame_rate);
      default:
        SPDL_FAIL_INTERNAL(fmt::format(
            "Unexpected media type was given: {}",
            av_get_media_type_string(codec_ctx->codec_type)));
    }
  }();

  return decode_packets(
      std::move(packets), std::move(codec_ctx), std::move(filter_graph));
}

Task<std::vector<FrameContainer>> stream_decode(
    const enum AVMediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const std::string filter_desc,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg) {
  auto demuxer = stream_demux(type, src, timestamps, io_cfg);
  std::vector<folly::SemiFuture<FrameContainer>> futures;
  auto exec = getDecoderThreadPoolExecutor();
  while (auto packets = co_await demuxer.next()) {
    auto task = get_decode_task(*packets, filter_desc, decode_cfg);
    futures.emplace_back(std::move(task).scheduleOn(exec).start());
  }
  XLOG(DBG) << "Waiting for decode jobs to finish";
  std::vector<FrameContainer> results;
  size_t i = 0;
  for (auto& result : co_await collectAllTryRange(std::move(futures))) {
    if (result.hasValue()) {
      results.emplace_back(std::move(result.value()));
    } else {
      XLOG(ERR) << fmt::format(
          "Failed to decode video clip. Error: {} (Source: {}, timestamp: {}, {})",
          result.exception().what(),
          src,
          std::get<0>(timestamps[i]),
          std::get<1>(timestamps[i]));
    }
    ++i;
  };
  if (results.size() != timestamps.size()) {
    SPDL_FAIL("Failed to decode some video clips. Check the error log.");
  }
  co_return results;
}
} // namespace

std::vector<FrameContainer> decode_video(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::string& filter_desc,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg) {
  auto job = stream_decode(
      AVMEDIA_TYPE_VIDEO, src, timestamps, filter_desc, io_cfg, decode_cfg);
  return folly::coro::blockingWait(
      std::move(job).scheduleOn(getDemuxerThreadPoolExecutor()));
}

std::vector<FrameContainer> decode_audio(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::string& filter_desc,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg) {
  auto job = stream_decode(
      AVMEDIA_TYPE_AUDIO, src, timestamps, filter_desc, io_cfg, decode_cfg);
  return folly::coro::blockingWait(
      std::move(job).scheduleOn(getDemuxerThreadPoolExecutor()));
}

} // namespace spdl
