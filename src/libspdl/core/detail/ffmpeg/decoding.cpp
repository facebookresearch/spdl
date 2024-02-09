#include <libspdl/core/detail/ffmpeg/decoding.h>

#include <libspdl/core/adoptor/basic.h>
#include <libspdl/core/detail/ffmpeg/ctx_utils.h>
#include <libspdl/core/detail/ffmpeg/filter_graph.h>
#include <libspdl/core/detail/ffmpeg/logging.h>
#include <libspdl/core/detail/ffmpeg/wrappers.h>
#include <libspdl/core/detail/tracing.h>
#include <libspdl/core/logging.h>

#include <folly/logging/xlog.h>

#include <random>

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace spdl::core::detail {
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

std::unique_ptr<DataInterface> get_interface(
    const std::string src,
    std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg) {
  if (!adoptor) {
    adoptor.reset(static_cast<SourceAdoptor*>(new BasicAdoptor{}));
  }
  return std::unique_ptr<DataInterface>(
      static_cast<DataInterface*>(adoptor->get(src, io_cfg)));
}

uint64_t random() {
  static thread_local std::random_device rd;
  static thread_local std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  return dis(gen);
}
} // namespace

folly::coro::AsyncGenerator<std::unique_ptr<PackagedAVPackets>> stream_demux(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg) {
  TRACE_EVENT("demuxing", "detail::stream_demux");
  auto interface = get_interface(src, adoptor, io_cfg);
  AVFormatContext* fmt_ctx = interface->get_fmt_ctx();
  int idx = parse_fmt_ctx(fmt_ctx, type);
  AVStream* stream = fmt_ctx->streams[idx];

  // Disable other streams
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (i != idx) {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
  for (auto& timestamp : timestamps) {
    TRACE_EVENT("demuxing", "detail::stream_demux - iter");
    auto [start, end] = timestamp;

    int64_t t = static_cast<int64_t>(start * AV_TIME_BASE);
    TRACE_EVENT_BEGIN("demuxing", "av_seek_frame");
    CHECK_AVERROR(
        av_seek_frame(fmt_ctx, -1, t, AVSEEK_FLAG_BACKWARD),
        "Failed to seek to the timestamp {} [sec]",
        start);
    TRACE_EVENT_END("demuxing");

    auto package = std::make_unique<PackagedAVPackets>(
        fmt_ctx->url,
        timestamp,
        stream->codecpar,
        stream->time_base,
        av_guess_frame_rate(fmt_ctx, stream, nullptr));
    double packet_ts = 0;
    do {
      AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
      TRACE_EVENT_BEGIN("demuxing", "av_read_frame");
      int errnum = av_read_frame(fmt_ctx, packet.get());
      TRACE_EVENT_END("demuxing");
      if (errnum == AVERROR_EOF) {
        break;
      }
      CHECK_AVERROR_NUM(errnum, "Failed to process packet.");
      if (packet->stream_index != idx) {
        continue;
      }
      packet_ts = packet->pts * av_q2d(stream->time_base);
      package->packets.push_back(packet.release());
      // Note:
      // Since the video frames can be non-chronological order, so we add small
      // margin to end
    } while (packet_ts < end + 0.3);
    XLOG(DBG9) << fmt::format(" - Sliced {} packets", package->packets.size());
    // For flushing
    package->packets.push_back(nullptr);
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
  TRACE_EVENT_BEGIN("decoding", "av_buffersrc_add_frame_flags");
  int errnum =
      av_buffersrc_add_frame_flags(src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
  TRACE_EVENT_END("decoding");

  CHECK_AVERROR_NUM(errnum, "Failed to pass a frame to filter.");

  AVFrameAutoUnref frame_ref{frame};
  while (errnum >= 0) {
    AVFramePtr frame2{CHECK_AVALLOCATE(av_frame_alloc())};
    TRACE_EVENT_BEGIN("decoding", "av_buffersrc_get_frame");
    errnum = av_buffersink_get_frame(sink_ctx, frame2.get());
    TRACE_EVENT_END("decoding");
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
  TRACE_EVENT("decoding", "detail::decode_packet");
  assert(codec_ctx);
  XLOG(DBG9)
      << ((!packet) ? fmt::format(" -- flush decoder")
                    : fmt::format(
                          "{:21s} {:.3f} ({})",
                          " -- packet:",
                          TS(packet, codec_ctx->pkt_timebase),
                          packet->pts));

  TRACE_EVENT_BEGIN("decoding", "avcodec_send_packet");
  int errnum = avcodec_send_packet(codec_ctx, packet);
  TRACE_EVENT_END("decoding");
  while (errnum >= 0) {
    AVFramePtr frame{CHECK_AVALLOCATE(av_frame_alloc())};
    TRACE_EVENT_BEGIN("decoding", "avcodec_receive_frame");
    errnum = avcodec_receive_frame(codec_ctx, frame.get());
    TRACE_EVENT_END("decoding");
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

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_pkts(
    std::unique_ptr<PackagedAVPackets> packets,
    AVCodecContextPtr codec_ctx) {
  auto [start, end] = packets->timestamp;
  auto container = std::make_unique<FrameContainer>(
      packets->id,
      codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO ? MediaType::Audio
                                                  : MediaType::Video);
  for (auto& packet : packets->packets) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto frame = co_await decoding.next()) {
      AVFramePtr f = *frame;
      if (f) {
        double ts = TS(f, codec_ctx->pkt_timebase);
        if (start <= ts && ts <= end) {
          XLOG(DBG9) << fmt::format(
              "{:21s} {:.3f} ({})", " --- raw frame:", ts, f->pts);
          container->frames.push_back(f.release());
        }
      }
    }
  }
  co_return container;
}

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_pkts_with_filter(
    std::unique_ptr<PackagedAVPackets> packets,
    AVCodecContextPtr codec_ctx,
    std::string filter_desc) {
  auto filter_graph = [&]() {
    switch (codec_ctx->codec_type) {
      case AVMEDIA_TYPE_AUDIO:
        return get_audio_filter(filter_desc, codec_ctx.get());
      case AVMEDIA_TYPE_VIDEO:
        return get_video_filter(
            filter_desc, codec_ctx.get(), packets->frame_rate);
      default:
        SPDL_FAIL_INTERNAL(fmt::format(
            "Unexpected media type was given: {}",
            av_get_media_type_string(codec_ctx->codec_type)));
    }
  }();
  AVFilterContext* src_ctx = filter_graph->filters[0];
  AVFilterContext* sink_ctx = filter_graph->filters[1];
  assert(strcmp(src_ctx->name, "in") == 0);
  assert(strcmp(sink_ctx->name, "out") == 0);

  XLOG(DBG5) << describe_graph(filter_graph.get());

  auto [start, end] = packets->timestamp;
  auto container = std::make_unique<FrameContainer>(
      packets->id, get_output_media_type(filter_graph.get()));

  for (auto& packet : packets->packets) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto raw_frame = co_await decoding.next()) {
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
          if (start <= ts && ts <= end) {
            XLOG(DBG9) << fmt::format(
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
    std::unique_ptr<PackagedAVPackets> packets,
    const DecodeConfig cfg,
    std::string filter_desc) {
  TRACE_EVENT(
      "decoding", "decode_packets", perfetto::Flow::ProcessScoped(packets->id));
  auto codec_ctx = get_codec_ctx_ptr(
      packets->codecpar,
      packets->time_base,
      cfg.decoder,
      cfg.decoder_options,
      cfg.cuda_device_index);
  if (filter_desc.empty()) {
    co_return co_await decode_pkts(std::move(packets), std::move(codec_ctx));
  }
  co_return co_await decode_pkts_with_filter(
      std::move(packets), std::move(codec_ctx), std::move(filter_desc));
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
    : id(random()),
      src(src_),
      timestamp(timestamp_),
      codecpar(copy(codecpar_)),
      time_base(time_base_),
      frame_rate(frame_rate_) {
  TRACE_EVENT(
      "decoding",
      "PackagedAVPackets::PackagedAVPackets",
      perfetto::Flow::ProcessScoped(id));
};

PackagedAVPackets::PackagedAVPackets(PackagedAVPackets&& other) noexcept {
  *this = std::move(other);
};

PackagedAVPackets& PackagedAVPackets::operator=(
    PackagedAVPackets&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(src, other.src);
  swap(timestamp, other.timestamp);
  swap(codecpar, other.codecpar);
  swap(time_base, other.time_base);
  swap(frame_rate, other.frame_rate);
  swap(packets, other.packets);
  return *this;
};

PackagedAVPackets::~PackagedAVPackets() {
  TRACE_EVENT(
      "decoding",
      "PackagedAVPackets::~PackagedAVPackets",
      perfetto::Flow::ProcessScoped(id));
  std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
    if (p) {
      av_packet_unref(p);
      av_packet_free(&p);
    }
  });
  avcodec_parameters_free(&codecpar);
};

} // namespace spdl::core::detail
