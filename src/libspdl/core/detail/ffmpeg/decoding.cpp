#include <libspdl/core/detail/ffmpeg/ctx_utils.h>
#include <libspdl/core/detail/ffmpeg/decoding.h>
#include <libspdl/core/detail/ffmpeg/filter_graph.h>
#include <libspdl/core/detail/ffmpeg/logging.h>
#include <libspdl/core/detail/ffmpeg/wrappers.h>
#include <libspdl/core/interface.h>
#include <libspdl/core/logging.h>

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace spdl::detail {
////////////////////////////////////////////////////////////////////////////////
// Demuxer
////////////////////////////////////////////////////////////////////////////////
namespace {
inline int parse_fmt_ctx(AVFormatContext* fmt_ctx, enum MediaType type_) {
  CHECK_AVERROR(
      avformat_find_stream_info(fmt_ctx, nullptr),
      "Failed to find stream information.");

  AVMediaType type =
      type_ == MediaType::Video ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;
  int idx = av_find_best_stream(fmt_ctx, type, -1, -1, nullptr, 0);
  if (idx < 0) {
    SPDL_FAIL(
        fmt::format("No {} stream was found.", av_get_media_type_string(type)));
  }
  return idx;
}
} // namespace

folly::coro::AsyncGenerator<PackagedAVPackets&&> stream_demux(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const IOConfig cfg) {
  auto dp =
      get_data_provider(src, cfg.format, cfg.format_options, cfg.buffer_size);
  AVFormatContext* fmt_ctx = dp->get_fmt_ctx();
  int idx = parse_fmt_ctx(fmt_ctx, type);
  AVStream* stream = fmt_ctx->streams[idx];

  // Disable other streams
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (i != idx) {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }

  for (auto& timestamp : timestamps) {
    auto [start, end] = timestamp;

    int64_t t = static_cast<int64_t>(start * AV_TIME_BASE);
    CHECK_AVERROR(
        av_seek_frame(fmt_ctx, -1, t, AVSEEK_FLAG_BACKWARD),
        "Failed to seek to the timestamp {} [sec]",
        start);

    PackagedAVPackets package{
        fmt_ctx->url,
        timestamp,
        stream->codecpar,
        stream->time_base,
        av_guess_frame_rate(fmt_ctx, stream, nullptr)};
    double packet_ts = 0;
    do {
      AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
      int errnum = av_read_frame(fmt_ctx, packet.get());
      if (errnum == AVERROR_EOF) {
        break;
      }
      CHECK_AVERROR_NUM(errnum, "Failed to process packet.");
      if (packet->stream_index != idx) {
        continue;
      }
      packet_ts = packet->pts * av_q2d(stream->time_base);
      package.packets.push_back(packet.release());
      // Note:
      // Since the video frames can be non-chronological order, so we add small
      // margin to end
    } while (packet_ts < end + 0.3);
    XLOG(DBG) << fmt::format(" - Sliced {} packets", package.packets.size());
    // For flushing
    package.packets.push_back(nullptr);
    co_yield std::move(package);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
namespace {

#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

folly::coro::AsyncGenerator<AVFramePtr&&> filter_frame(
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

folly::coro::AsyncGenerator<AVFramePtr&&> decode_packet(
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

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_pkts(
    PackagedAVPackets packets,
    const DecodeConfig cfg) {
  auto codec_ctx = get_codec_ctx(
      packets.codecpar,
      packets.time_base,
      cfg.decoder,
      cfg.decoder_options,
      cfg.cuda_device_index);

  auto [start, end] = packets.timestamp;
  auto container = std::make_unique<FrameContainer>(
      codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO ? MediaType::Audio
                                                  : MediaType::Video);
  for (auto& packet : packets.packets) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto frame = co_await decoding.next()) {
      AVFramePtr f = *frame;
      if (f) {
        double ts = TS(f, codec_ctx->pkt_timebase);
        if (start <= ts && ts <= end) {
          XLOG(DBG) << fmt::format(
              "{:21s} {:.3f} ({})", " --- raw frame:", ts, f->pts);
          container->frames.push_back(f.release());
        }
      }
    }
  }
  co_return container;
}

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_pkts(
    PackagedAVPackets packets,
    const std::string filter_desc,
    const DecodeConfig cfg) {
  auto codec_ctx = get_codec_ctx(
      packets.codecpar,
      packets.time_base,
      cfg.decoder,
      cfg.decoder_options,
      cfg.cuda_device_index);

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
  // XLOG(DBG) << describe_graph(filter_graph.get());

  auto [start, end] = packets.timestamp;

  AVFilterContext* src_ctx = filter_graph->filters[0];
  AVFilterContext* sink_ctx = filter_graph->filters[1];
  assert(strcmp(src_ctx->name, "in") == 0);
  assert(strcmp(sink_ctx->name, "out") == 0);

  auto container = std::make_unique<FrameContainer>(
      get_output_media_type(filter_graph.get()));
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

            container->frames.push_back(f.release());
          }
        }
      }
    }
  }
  co_return container;
}
} // namespace

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_packets(
    PackagedAVPackets&& packets,
    const std::string filter_desc,
    const DecodeConfig cfg) {
  if (filter_desc.empty()) {
    return decode_pkts(std::move(packets), cfg);
  }
  return decode_pkts(std::move(packets), filter_desc, cfg);
}

////////////////////////////////////////////////////////////////////////////////
// PackagedAVPackets
////////////////////////////////////////////////////////////////////////////////
namespace {
inline AVCodecParameters* copy(const AVCodecParameters* src) {
  auto dst = CHECK_AVALLOCATE(avcodec_parameters_alloc());
  CHECK_AVERROR(
      avcodec_parameters_copy(dst, src), "Failed to copy codec parameters.");
  return dst;
}
} // namespace

PackagedAVPackets::PackagedAVPackets(
    std::string src_,
    std::tuple<double, double> timestamp_,
    AVCodecParameters* codecpar_,
    AVRational time_base_,
    AVRational frame_rate_)
    : src(src_),
      timestamp(timestamp_),
      codecpar(copy(codecpar_)),
      time_base(time_base_),
      frame_rate(frame_rate_){};

PackagedAVPackets::PackagedAVPackets(PackagedAVPackets&& other) noexcept {
  *this = std::move(other);
};

PackagedAVPackets& PackagedAVPackets::operator=(
    PackagedAVPackets&& other) noexcept {
  using std::swap;
  swap(src, other.src);
  swap(timestamp, other.timestamp);
  swap(codecpar, other.codecpar);
  swap(time_base, other.time_base);
  swap(frame_rate, other.frame_rate);
  swap(packets, other.packets);
  return *this;
};

PackagedAVPackets::~PackagedAVPackets() {
  std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
    if (p) {
      av_packet_unref(p);
      av_packet_free(&p);
    }
  });
  avcodec_parameters_free(&codecpar);
};

} // namespace spdl::detail
